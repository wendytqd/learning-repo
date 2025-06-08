import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import umap

base_dir = os.path.dirname(__file__)
vis_dir = os.path.join(base_dir, "Visuals")

# Load the datasets
elisa_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "Data/multiplexELISA_normalized.csv"))
cytolink_df = pd.read_excel(os.path.join(os.path.dirname(__file__), "Data/CytokineLink_HPA_cytokine2cytokine.xlsx"))

# -------- Filter the datasets to include only the relevant columns --------
elisa_cytokines = set(elisa_df.columns[5:])
cytolink_cytokines = set(cytolink_df["Source_cytokine_genename"]) | set(cytolink_df["Target_cytokine_genename"])
cytokines = elisa_cytokines.intersection(cytolink_cytokines)

# Remove CCL17 because it does not have enough data
cytokines.remove("CCL17")

cytokines = sorted(cytokines)  # Sort the cytokines for consistency

# --------- Keep metadata and the cytokines of interest ---------
metacols = ["race", "age", "gender"]
data = elisa_df[["subject_accession", *metacols, *cytokines]]  # Keep only the relevant columns


# -------- Compute the common edge set ---------
node2idx = {c: i for i, c in enumerate(cytokines)}  # Map cytokine names to indices
filtered_edges = cytolink_df[
    cytolink_df["Source_cytokine_genename"].isin(cytokines) & cytolink_df["Target_cytokine_genename"].isin(cytokines)
].copy()  # Keep only the relevant edges

# --------- Attach weights from the correlation matrix ---------
corr = pd.read_csv("Cytokine_corr_spearman.csv", index_col=0).loc[cytokines, cytokines].fillna(0)

edge_weights_df = pd.DataFrame(
    {
        "source": filtered_edges["Source_cytokine_genename"].values,
        "target": filtered_edges["Target_cytokine_genename"].values,
        "weight": [
            corr.loc[s, t]
            for s, t in zip(filtered_edges["Source_cytokine_genename"], filtered_edges["Target_cytokine_genename"])
        ],
    }
)
edge_weights_df = edge_weights_df.drop_duplicates()  # Remove duplicates (len = 722)
edge_weights_df = edge_weights_df[edge_weights_df["weight"] != 0]  # Remove edges with weight = 0 (len = 636)

# Map gene names -> integer indices required by PyTorch Geometric
src = edge_weights_df["source"].map(node2idx).tolist()
tgt = edge_weights_df["target"].map(node2idx).to_list()

edge_index = torch.tensor([src, tgt], dtype=torch.long)  # Create edge index tensor
edge_weights = torch.tensor(edge_weights_df["weight"].to_numpy(dtype=np.float32))  # Create edge weights tensor
# -------- Create the graph ----------
G = nx.from_pandas_edgelist(
    edge_weights_df,
    source="source",
    target="target",
    edge_attr="weight",
    create_using=nx.DiGraph(),
)


# # Compute silhouette score
# silhouette_scores = []
# k_range = range(2, 11)

# A = nx.to_scipy_sparse_array(G, weight="weight", format="csr")  # adjacency matrix with signed weights
# degree = np.abs(A).sum(axis=1).flatten()  # Degree matrix based on sum of abs(weights) (convert to flat array)
# D = csr_matrix(np.diag(degree))
# L = D - A  # signed Laplacian

# for k in k_range:
#     eigval, eigvec = eigsh(L, k=k, which="SM")
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     labels = kmeans.fit_predict(eigvec)
#     sil_score = silhouette_score(eigvec, labels)
#     silhouette_scores.append(sil_score)

# best_k = k_range[np.argmax(silhouette_scores)]

# eigval_sil, eigvec_sil = eigsh(L, k=best_k, which="SM")
# kmeans_sil = KMeans(n_clusters=best_k, random_state=0)
# labels_sil = kmeans_sil.fit_predict(eigvec_sil)

# node_list = list(G.nodes())

# # Visualise using UMAP
# umap_model = umap.UMAP(n_components=2, random_state=42)
# eigvec_umap = umap_model.fit_transform(eigvec_sil)

# pos_umap = {node_list[i]: eigvec_umap[i, :2] for i in range(eigvec_umap.shape[0])}

# num_clusters = len(set(labels_sil))
# cmap = ListedColormap(plt.cm.Set3.colors[:num_clusters])

# plt.figure(figsize=(20, 16))
# edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
# edge_width = [d["weight"] for _, _, d in G.edges(data=True)]

# nx.draw_networkx_nodes(G, pos_umap, node_size=400, node_color=labels_sil, cmap=cmap, alpha=0.9)
# nx.draw_networkx_edges(G, pos_umap, width=edge_width, edge_color="gray", arrowstyle="->", alpha=0.4)
# nx.draw_networkx_labels(G, pos_umap, font_size=7, font_family="Times New Roman", verticalalignment="center")
# nx.draw_networkx_edge_labels(G, pos_umap, edge_labels=edge_labels, font_size=4, font_family="Times New Roman")

# unique_labels = np.unique(labels_sil)
# handles = [mpatches.Patch(color=cmap(i), label=f"Cluster{i+1}") for i in unique_labels]
# plt.legend(handles=handles, loc="best")
# plt.title(f"Cytokine Interaction Graph (k={best_k})")
# plt.axis("off")
# plt.margins(0)
# plt.tight_layout(rect=[0, 0, 1, 0.92], pad=0.1)
# plt.savefig(os.path.join(vis_dir, f"Cytokine Network UMAP (k={best_k})"), bbox_inches="tight", pad_inches=0.05)
# plt.show()
