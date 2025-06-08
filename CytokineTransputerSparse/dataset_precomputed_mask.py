import os
import shutil
import torch
import numpy as np
import pandas as pd

from typing import List, Callable
from torch_geometric.data import Data, InMemoryDataset


class CytokinewithPEandMask(InMemoryDataset):
    # Attach random-walk PE (P) and node degree vector (d) to each graph
    # Precompute 10 maksed versions of each subject's graph, with mask-rates uniformly drawn from [0.1,0.99].
    # This avoids doing random masking every epoch and saves runtime

    def __init__(
        self,
        df: pd.DataFrame,
        cytokine_list: List[str],
        edge_index: torch.LongTensor,
        edge_weights: torch.Tensor,
        meta_enc_fn: Callable[[pd.Series], np.ndarray],
        root: str,
        force_reload=False,
        K: int = 4,  # length of random-walk PE
        n_variations: int = 10,
        in_meta_dim: int = 7,  # raw meta length ()
        proj_meta_dim: int = 16,
        transform=None,  # any post-mask transform (usually None)
        pre_transform=None,  # transforms to run on the unmasked "base" graph
    ):
        self.df = df.reset_index(drop=True)
        self.cytokine_list = cytokine_list
        self.edge_index = edge_index
        self.edge_weights = edge_weights
        self.meta_enc_fn = meta_enc_fn
        self.K = K
        self.n_variations = n_variations
        self.n_nodes = len(cytokine_list)

        # 1) Load precomputed P matrix an build deg_vec
        pe_path = os.path.join(root, f"PE_lap_{self.K}.pt")
        deg_path = os.path.join(root, "deg_vec.pt")
        if not os.path.exists(pe_path) or not os.path.exists(deg_path):
            raise FileNotFoundError(
                f"Cannot find {pe_path!r} or {deg_path!r}. " "Run the precompute_pe.py script first."
            )
        print(f"Loaded PE_lap_{self.K}.pt")
        self.P_matrix = torch.load(pe_path)
        self.deg_vec = torch.load(deg_path)

        # # ) Meta-projector MLP: raw meta_dim -> proj_meta_dim
        # self.meta_projector = torch.nn.Sequential(
        #     torch.nn.Linear(in_meta_dim, 32), torch.nn.ReLU(), torch.nn.Linear(32, proj_meta_dim)
        # )

        # 3) Precompute 'broadcasted' meta embeddings and log(1+deg) once for each subject:
        self.meta_broadcast_list = []
        self.logdeg_repeat_list = []

        for _, row in self.df.iterrows():
            raw_meta = self.meta_enc_fn(row)  # nd.array shape [meta_dim] = [7]
            raw_meta_t = torch.tensor(raw_meta, dtype=torch.float32)
            # embed_meta = self.meta_projector(raw_meta_t)
            # embed_meta = embed_meta.detach()  # detach from the graph so embed_meta.requires_grad == False
            # mb = embed_meta.unsqueeze(0).repeat(self.n_nodes, 1)  # [1,16] -> [55,16]
            mb = raw_meta_t.unsqueeze(0).repeat(self.n_nodes, 1)
            self.meta_broadcast_list.append(mb)

            logdeg = torch.log(self.deg_vec + 1.0).unsqueeze(1)  # [55,1]
            self.logdeg_repeat_list.append(logdeg)

        proc_dir = os.path.join(root, "processed")
        if force_reload:
            shutil.rmtree(proc_dir, ignore_errors=True)

        # 4) Build original (unmasked) dataset by calling super()
        super().__init__(root, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

        print("  graphs actually loaded into memory  :", len(self))
        print("  subjects in the dataframe           :", self.df.shape[0])
        assert len(self) == len(self.df), f"Expected {len(self.df)} graphs, found {len(self)}"

        # After super(), 'self.data' and 'self.slices' hold the processed graphs
        # but they do not YET include P_matrix or deg_vec or masking variations
        # 5) Now attach masking variations
        self._attach_p_and_masks()  # Rebuild the processed file here:

    @property
    def raw_file_names(self):
        # bypass raw  files (already have df in memory)
        return []

    @property
    def processed_file_names(self):
        name = f"data_LAPPE_K{self.K}_var{self.n_variations}.pt"
        return [name]

    def download(self):
        pass

    def process(self):  # run once to create a temporary list of UNMASKED graphs with std x and original y/mask.
        data_list = []
        num_edges = self.edge_index.size(1)
        edge_ids = torch.arange(num_edges, dtype=torch.long)

        for idx, row in self.df.iterrows():
            # 1) Node-wise cytokine values
            vals = row[self.cytokine_list].to_numpy(dtype=np.float32)  # [55]
            orig_mask = ~np.isnan(vals)  # True = observed; False = missing
            conc = np.nan_to_num(vals, nan=0.0).reshape(-1, 1)  # [55,1]

            # 2) Broadcast preprojected metadata + logdeg
            meta_broadcast = self.meta_broadcast_list[idx]  # [55, 7]
            logdeg = self.logdeg_repeat_list[idx]  # [55,1]

            # 3) Final node feature = [conc, meta, logeg]
            x = torch.cat(
                [torch.tensor(conc, dtype=torch.float32), meta_broadcast, logdeg], dim=1
            )  # [n_nodes, 1+16+1] = [55,18]

            # 4) per-edge static Lap-PE
            src, dst = self.edge_index
            per_edge_pe = self.P_matrix[src, dst]  # [E, K]

            g = Data(
                x=x,
                y=torch.tensor(vals).unsqueeze(1),  # original raw conc [n,1]
                edge_index=self.edge_index,
                edge_attr=per_edge_pe,  # self.edge_weights.unsqueeze(1)
                mask=torch.tensor(orig_mask),  # [55] True where actually measured, False where originally missing
                orig_mask=torch.tensor(orig_mask),  # [55] same as g.mask
                num_nodes=self.n_nodes,
            )
            g.edge_id = edge_ids
            data_list.append(g)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return None

    def _attach_p_and_masks(self):
        # After process() has run once, we now have:
        # Load the unmaksed graphs from disk
        # for each subject-graph, create 'n_variations' clones with different random mask rates
        # attach self.P_matrix and self.deg_vec to each clone
        # Collate them all and overwrite the processed file

        # 1) Load the unmasked graphs
        data_path = self.processed_paths[0]
        data_all, slices_all = torch.load(data_path, weights_only=False)

        # 2) Build a new list that will contain ALL masked variations
        new_list = []

        for idx in range(len(self.df)):
            g = self.get(idx).cpu()  # load the idx-th Data from disk
            g.P = self.P_matrix.clone()  # [n,n,K]
            g.d = self.deg_vec  # [n]

            # Originally "observed" mask (True of originally measured)
            orig_mask = g.orig_mask  # [n]

            for _ in range(self.n_variations):
                rate = float(np.random.uniform(0.1, 0.99))
                g2 = g.clone()

                known_idx = torch.nonzero(orig_mask).flatten()  # Find indices of orig_mask==True (known conc)
                num_known = known_idx.numel()
                k = int(rate * num_known)  # how many known nodes to drop

                if k > 0:
                    perm = torch.randperm(num_known)
                    drop = known_idx[perm[:k]]  # indices to mask

                    # 3) Zero out the std-conc for dropped nodes
                    x_clone = g2.x.clone()
                    x_clone[drop, 0] = 0.0  # channel 0 is the std-conc
                    g2.x = x_clone

                    # 4) Build a new mask vector: True where we want to impute (i.e. dropped)
                    mask_vec = torch.zeros_like(orig_mask, dtype=torch.bool)
                    mask_vec[drop] = True  # mask these nodes for trianing
                    g2.mask = mask_vec

                # If k = 0, g2.mask stays as the originally known
                new_list.append(g2)

        # 5) Collate ALL masked graphs and overwrite processed file
        data_new, slices_new = self.collate(new_list)
        torch.save((data_new, slices_new), data_path)
        self.data, self.slices = data_new, slices_new


class PerNodeZScoreTransform:
    def __init__(self, means: np.ndarray, stds: np.ndarray, eps: float = 1e-6):
        self.means = torch.tensor(means, dtype=torch.float32)  # [55]
        self.stds = torch.tensor(stds, dtype=torch.float32)
        self.eps = eps

    def __call__(self, data: Data) -> Data:
        x = data.x.clone()
        # select column 0
        x[:, 0] = (x[:, 0] - self.means) / (self.stds + self.eps)
        data.x = x
        return data


def subject_to_dsall_indices(subject_idx, n_variations):
    base = subject_idx * n_variations
    return list(range(base, base + n_variations))
