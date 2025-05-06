import torch
import torch.nn.functional as F
from torch_geometric.nn import NNConv


class CytokineImputer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        # edge MLP for conv 1: scalar weight -> matrix of shape [in->hidden]
        self.edge_net1 = torch.nn.Sequential(
            torch.nn.Linear(1, 32),
            torch.nn.SiLU(),
            torch.nn.Linear(32, in_channels * hidden_channels),
        )
        self.conv1 = NNConv(in_channels, hidden_channels, self.edge_net1, aggr="mean")

        # edge MLP for conv2: scalar weight -> matrix of shape [hidden-> hidden]
        self.edge_net2 = torch.nn.Sequential(
            torch.nn.Linear(1, 32),
            torch.nn.SiLU(),
            torch.nn.Linear(32, hidden_channels * hidden_channels),
        )
        self.conv2 = NNConv(hidden_channels, hidden_channels, self.edge_net2, aggr="mean")

        self.decoder = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_weight):
        e = edge_weight.unsqueeze(1)
        h = self.conv1(x, edge_index, e)
        h = F.silu(h)
        h = self.conv2(h, edge_index, e)
        h = F.silu(h)
        out = self.decoder(h)
        return out.squeeze(1)
