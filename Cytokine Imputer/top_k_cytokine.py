import os
import torch
import random
import numpy as np
import pandas as pd
import networkx as nx
from copy import deepcopy
import matplotlib.pyplot as plt

from torch.nn import MSELoss
from torch.utils.data import random_split, Subset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import AddLaplacianEigenvectorPE, Compose

from train import evaluate
from expert import expert_df
from model import CytokineImputer
from metadata_utils import encode_metadata
from data_loader import CytokineSubjectDataset, AppendLaplacianEigenvectorPE, RandomMaskTransform
from graph_utils import G, cytokines, corr, edge_index, edge_weights, best_k, data

base_dir = os.path.dirname(__file__)
vis_dir = os.path.join(base_dir, "Visuals")
result_dir = os.path.join(base_dir, "Results")
models_dir = os.path.join(base_dir, "Models")
graph_dir = os.path.join(base_dir, "Precomputed_graphs")

node_list = list(G.nodes())
W = nx.to_numpy_array(G, nodelist=node_list, weight="weight")  # shape [N, N]

# build the dataframe
elisa_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "multiplexELISA_normalized.csv"))
df = elisa_df[["subject_accession", "race", "age", "gender", *node_list]]

# ------- Degree Centrality -------
# compute weighted degree centrality
deg_centrality = {n: sum(abs(d["weight"]) for _, _, d in G.edges(n, data=True)) for n in G.nodes()}
# degree centrality = sum of absolute edge-weights incident
order_degree = sorted(deg_centrality, key=deg_centrality.get, reverse=True)


# ------- Greedy Coverage of Correlation -------
# pick nodes that together cover the largest total absolute correlation to all others
def greedy_cover(W: np.ndarray, nodes: list[str]):
    W_abs = np.abs(W)
    N = len(nodes)
    selected = []
    covered = np.zeros(N)

    for n in range(N):
        gains = [(i, (np.maximum(covered, W_abs[i]).sum() - covered.sum())) for i in range(N) if i not in selected]
        best_i, n = max(gains, key=lambda x: x[1])
        selected.append(best_i)
        covered = np.maximum(covered, W_abs[best_i])
    return [nodes[i] for i in selected]


order_cover = greedy_cover(W, node_list)

# ------- k-Core Decomposition -------
cores = nx.core_number(G)
order_k_core = sorted(cores, key=cores.get, reverse=True)

# ------- Spectral decomposition ------
L = nx.normalized_laplacian_matrix(G, weight="weight")
A_absolute = np.abs(nx.to_numpy_array(G, nodelist=node_list, weight="weight"))
eigvals, eigvecs = np.linalg.eigh(A_absolute)

# pick the eigenvector for the *largest* eigenvalue
idx_max = np.argmax(eigvals)
principal = np.real(eigvecs[:, idx_max])

# compute leverage scores: row-norms squaured of the eigenvector matrix
scores = {node_list[i]: float(principal[i]) for i in range(len(node_list))}
max_score = max(scores.values())
scores = {n: s / max_score for n, s in scores.items()}

order_spectral_decomposition = sorted(scores, key=scores.get, reverse=True)

# ------- PCA on correlation matrix -------
eigval_pca, eigvec_pca = np.linalg.eigh(corr.values)

order = np.argsort(eigval_pca)[::-1]  # desceinding order
pc1 = eigvec_pca[:, order[0]]  # pick the leading principal component

loadings = np.abs(pc1)
order_pca = [node_list[i] for i in np.argsort(loadings)[::-1]]

total_var = np.sum(eigval_pca)
exp_var_ratio = eigval_pca[order] / total_var
cumvar = np.cumsum(exp_var_ratio)
threshold = 0.8  # capture 80% of total variance in correlation
k_pca = (
    np.searchsorted(cumvar, threshold) + 1
)  # the number of latent dimensions that would be ekpt if a dimensionality reduction of the full correlation matrix is done

# -------- Expert --------
expert_cyto = list(expert_df.index)

# Collect all the orderings
rankings = {
    "Degree": order_degree,
    "Coverage": order_cover,
    "K-core": order_k_core,
    "Spectral decomposition": order_spectral_decomposition,
    "PCA": order_pca,
    "Expert": expert_cyto,
}

np.random.seed(42)
shuffled_cyto = list(np.random.permutation(node_list))
rankings["Random"] = shuffled_cyto


# Define dataset sizes
n = len(df)
t, v = int(0.7 * n), int(0.15 * n)
split = random_split(df, [t, v, n - t - v], generator=torch.Generator().manual_seed(42))
val_ids = set(df.iloc[split[1].indices]["subject_accession"])

# add Laplacian eigenvector as a node feature
pe = AddLaplacianEigenvectorPE(k=best_k)
append_pe = AppendLaplacianEigenvectorPE()
full_pre_transform = Compose([pe, append_pe])

# Pre-load the graphs
ds = CytokineSubjectDataset(
    df=df,
    cytokine_list=cytokines,
    edge_index=edge_index,
    edge_weights=edge_weights,
    meta_enc_fn=encode_metadata,
    root=graph_dir,
    mask=False,
    pre_transform=full_pre_transform,
    transform=None,  # RandomMaskTransform will be applied later
)
print("Loaded all graphs: ", len(ds))

val_idx = [i for i, row in enumerate(df.itertuples(False)) if row.subject_accession in val_ids]

ds_val = Subset(ds, val_idx)
print(f"Validation graphs: {len(ds_val)}")


# ======= Define the TopKMaskTransform =========
class TopKMaskTransform:
    def __init__(self, measured, node_list):
        self.measured_mask = torch.tensor([n in measured for n in node_list], dtype=torch.bool)

    def __call__(self, data):
        x = data.x.clone()
        to_predict = ~self.measured_mask.to(x.device)  # nodes NOT in measured (not k)
        x[to_predict, 0] = 0.0
        data.x = x
        data.mask = to_predict.unsqueeze(1)  # set data.mask: true where we want to impute
        return data


# ====== Find the top k cytokines ======
k_max = 20
k_list = list(range(1, k_max + 1))
# ======= Seed Averaging =========
# seeds = [42, 123, 456, 789, 1011]
methods = list(rankings.keys())
val_loss_seed = {m: [] for m in methods}
tolacc_seed = {m: [] for m in methods}
results = []


# for i, seed in enumerate(seeds):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)

num_total = len(node_list)
for m in methods:
    ordering = rankings[m]

    for k in range(1, k_max + 1):
        measured = ordering[:k]

        # mask and evaluate on the val set
        masked_graphs = []
        tm = TopKMaskTransform(measured, node_list)

        for g in ds_val:
            g2 = deepcopy(g)
            masked_graphs.append(tm(g2))

        loader = DataLoader(masked_graphs, batch_size=16, shuffle=False)

        # Define the model:
        device = torch.device("mps")
        in_channels = ds[0].x.size(1)
        model = CytokineImputer(in_channels=in_channels, k_pe=pe.k, hidden_channels=64).to(device)
        model.load_state_dict(
            torch.load(os.path.join(models_dir, "cyto_imputer_population.pth"), map_location=device, weights_only=True)
        )
        loss_fn = MSELoss()

        metrics = evaluate(model, loader, loss_fn, device)
        mse_per_imputed = metrics["MSE"]
        num_imputed = num_total - k
        sse = mse_per_imputed * num_imputed  # sum of sqaured errors
        norm_MSE = sse / num_total  # total squared error/nodes

        val_loss_seed[m].append(np.log(norm_MSE + 1e-8))
        tolacc_seed[m].append(metrics["TolAcc"])

        results.append(
            {
                "method": m,
                "k": k,
                "val_loss": np.log(metrics["MSE"] + 1e-8),  #
                "tolacc": metrics["TolAcc"],
            }
        )
    print(f"Done with {m}")

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(result_dir, "Topk_sweep_by_methods.csv"), index=False)

# Plot
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
for m in methods:
    plt.plot(k_list, val_loss_seed[m], marker="o", label=m)
plt.xticks(range(0, k_max + 1, 1))
plt.ylabel("Validation log-MSE")
plt.legend()

plt.subplot(1, 2, 2)
for m in methods:
    plt.plot(k_list, tolacc_seed[m], marker="o", label=m)
plt.xticks(range(0, k_max + 1, 1))
plt.ylabel("Tolerance Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(vis_dir, "Topk_by_methods.png"))
plt.show()


# val_loss_seed_means = {m: val_loss_seed[m].mean(axis=0) for m in methods}
# val_loss_seed_std = {m: val_loss_seed[m].std(axis=0) for m in methods}
# tolacc_seed_means = {m: tolacc_seed[m].mean(axis=0) for m in methods}
# tolacc_seed_std = {m: tolacc_seed[m].std(axis=0) for m in methods}

# # plot average over seeds
# x = np.arange(len(methods))
# width = 0.5
# plt.figure(figsize=(12, 8))
# plt.subplot(2, 1, 1)
# for m in methods:
#     plt.plot(k_list, val_loss_seed_means[m], marker="o", label=m)
#     plt.fill_between(
#         k_list, val_loss_seed_means[m] - val_loss_seed_std[m], val_loss_seed_means[m] + val_loss_seed_std[m], alpha=0.2
#     )
# plt.xticks(range(0, k_max + 1, 1))  # Set x-ticks from 0 to 20 with interval 1
# plt.ylabel("Validation log-MSE")
# plt.legend()

# plt.subplot(2, 1, 2)
# for m in methods:
#     plt.plot(k_list, tolacc_seed_means[m], marker="o", label=m)
#     plt.fill_between(
#         k_list, tolacc_seed_means[m] - tolacc_seed_std[m], tolacc_seed_means[m] + tolacc_seed_std[m], alpha=0.2
#     )
# plt.xticks(range(1, k_max + 1, 1))  # Set x-ticks from 0 to 20 with interval 1
# plt.yticks(np.arange(0, 1.1, 0.1), [f"{y:.1f}" for y in np.arange(0, 1.1, 0.1)])  # Set y-ticks with 1 decimal place
# plt.ylabel("Tolerance accuracy")
# plt.legend()
# plt.suptitle("Average over 5 seeds")
# plt.tight_layout()
# plt.savefig(os.path.join(vis_dir, "Average_Loss_TolAcc_by_Methods.png"))

aucs = {}
for method in methods:
    method_data = results_df[results_df["method"] == method]
    method_data = method_data.sort_values(by="k")
    tolacc_values = method_data["tolacc"].values
    mean_tol = method_data.groupby("k")["tolacc"].mean().reindex(k_list).values
    aucs[method] = np.trapz(mean_tol, k_list)  # compute mean tolerance per k, reindex to ensure missing ks are filled

best_method_by_auc = max(aucs, key=aucs.get)
ordered_auc = sorted(aucs.items(), key=lambda x: x[1], reverse=True)

plt.figure(figsize=(10, 6))
plt.bar([method for method, _ in ordered_auc], [auc for _, auc in ordered_auc], width=0.6)
plt.xticks(rotation=45)
plt.ylabel("AUC")
plt.title("AUC by Method")
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, "Topk_AUC.png"))
plt.show()

print(f"Best overall ordering by AUC: {best_method_by_auc}: {aucs[best_method_by_auc]:.3f}")
print("sorted by AUC:", ordered_auc)
