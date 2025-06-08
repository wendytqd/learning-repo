import os
import torch
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

from graph_utils import edge_index, edge_weights

# SciPy-sparse adjacency (coo_matrix)
n_nodes = int(edge_index.max().item()) + 1

A = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)
src, tgt = edge_index
A[src, tgt] = 1  # directed graph

# Form the degree diagonal
deg = A.sum(dim=1)
deg_inv_sqrt = torch.diag(1 / torch.sqrt(deg + 1e-8))
I_n = torch.eye(n_nodes)
# L = I- D^{-1/2} A D^{-1/2}
L = I_n - (deg_inv_sqrt @ A @ deg_inv_sqrt)

# Compute the first K eigenvectors
eigenvals, eigenvecs = torch.linalg.eigh(L)  # eigenvec = [n,n]

K = 8
phi = eigenvecs[:, :K].clone()  # [n, K]

# uil the 'relative-PE" tensor P of shape [n,n,K];
phi_i = phi.unsqueeze(1)  # [n, 1, K]
phi_j = phi.unsqueeze(0)  # [1, n, K]

P = torch.abs(phi_i - phi_j)

deg_vec = deg.clone()

base_dir = os.path.dirname(os.path.abspath(__file__))
graph_dir = os.path.join(base_dir, "Precomputed_graphs")
pe_outpath = os.path.join(graph_dir, f"PE_lap_{K}.pt")
deg_outpath = os.path.join(graph_dir, f"deg_vec.pt")

torch.save(P, pe_outpath)
torch.save(deg_vec, deg_outpath)
print(f"Laplacian-PE shape: {P.shape} saved to {pe_outpath}")
print(f"Degree vector shape: {deg_vec.shape} saved to {deg_outpath}")
