import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


class CytokineTransputerSparse(torch.nn.Module):
    def __init__(
        self,
        in_ch: int,  # 1 + proj_meta_dim + 1
        meta_dim: int,
        d_h: int = 64,
        edge_dim: int = 4,
        heads: int = 4,
        num_layers: int = 3,
    ):

        super().__init__()
        assert d_h % heads == 0

        self.meta_proj = torch.nn.Sequential(
            torch.nn.Linear(meta_dim, d_h),
            torch.nn.ReLU(),
            torch.nn.Linear(d_h, d_h),
        )

        # Initial node-feature MLP (just take in [conc, logdeg])
        self.node_proj = torch.nn.Linear(in_ch - meta_dim, d_h)

        # Build a stack of 'num_layers' sparse TransformerConv layers.
        # Each layer: Q, K, V dims = d_h, head x d' = d_h, edge_dim=K
        self.convs = torch.nn.ModuleList()
        # Build a tiny FiLM-MLP for each layer that spit out 2xd_h parameters (gamma and beta) from the shared meta embedding
        self.film_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = TransformerConv(
                in_channels=d_h,
                out_channels=d_h,
                heads=heads,
                concat=False,  # Output is [Bxn_nodes, d_h] not d_h*heads
                dropout=0.1,
                edge_dim=edge_dim,  # must match K=4 if K=4 is used
            )
            self.convs.append(conv)
            self.film_layers.append(torch.nn.Linear(d_h, 2 * d_h))

        # Final readout: from d_h -> 1 prediction per node
        self.readout = torch.nn.Linear(d_h, 1)

    def forward(self, x, edge_index, edge_attr, batch_vec):
        # x: [B*n-nodes, in_ch]
        # edge_index: [2, total_num_edges]
        # edge_attr: [total_num_edges, edge_dim=K] (Laplacian PE)
        # batch_vec: [B*n_nodes]
        # returns [B*n_nodes] predicted log-conc per node

        conc = x[:, 0:1]  # [B*n, 1]
        meta = x[:, 1 : 1 + self.meta_proj[0].in_features]  # [B*n, meta_dim]
        logdeg = x[:, -1:]  # [B*n, 1]

        # project metadata once and detach (so we don't ack propagate through it repeatedly)
        meta_emb = self.meta_proj(meta)  # [B*n, d_h]

        h = torch.cat([conc, logdeg], dim=1)
        h = self.node_proj(h)
        h = F.silu(h)

        for conv, film in zip(self.convs, self.film_layers):
            h = conv(h, edge_index, edge_attr)
            gamma, beta = film(meta_emb).chunk(2, dim=1)
            h = gamma * h + beta
            h = F.silu(h)

        out = self.readout(h).squeeze(-1)
        return out
