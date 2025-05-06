import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from metadata_utils import encode_metadata


# --------- Write a custom dataset class ---------
class CytokineSubjectDataset(Dataset):
    def __init__(self, df, cytokine_list, edge_index, edge_weights: int, meta_enc_fn):
        self.df = df.reset_index(drop=True)  # subject-level df (one row per subject)
        self.cytokine_list = cytokine_list  # List of cytokine to be extract as nodes
        self.edge_index = edge_index  # edges as PyTorch geomtric format
        self.edge_weights = edge_weights  # tensor of edge weights
        self.meta_enc_fn = meta_enc_fn  # function to convert subject metadata into a feature vector
        self.num_nodes = len(cytokine_list)
        self.node_feat_dim = 1 + len(meta_enc_fn(df.iloc[0]))  # 1 for the cytokine level + metadata features

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):  # returns a PyTorch Geometric Data object for one subject
        row = self.df.iloc[idx]

        cytokine_values = row[self.cytokine_list].to_numpy(dtype=np.float32)
        nan_mask = ~np.isnan(cytokine_values)  # True = known, False = missing

        # node-wise concentration vector
        conc = torch.tensor(np.nan_to_num(cytokine_values, nan=0.0), dtype=torch.float32).unsqueeze(
            1
        )  # Cytokine levels [N, 1]
        mask_tensor = torch.tensor(nan_mask, dtype=torch.bool).unsqueeze(1)  # [N, 1]

        # Broadcast subject metadata to everynode
        meta = torch.tensor(self.meta_enc_fn(row), dtype=torch.float32)  # Subject metadata [M]
        meta_broadcast = meta.repeat(self.num_nodes, 1)  # Repeat the metadata for each node [N, M]

        x = torch.cat([conc, meta_broadcast], dim=1)  # final node feature matrix [N, 1+M]

        data = Data(
            x=x, edge_index=self.edge_index, edge_attr=self.edge_weights.unsqueeze(1), mask=mask_tensor
        )  # [E,1]

        return data


def prepare_loaders(dataset, batch_size=16):
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    return (
        DataLoader(train_set, batch_size=16, shuffle=True),
        DataLoader(val_set, batch_size=16),
        DataLoader(test_set, batch_size=16),
    )
