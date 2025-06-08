import os, sys
import time
import torch
import numpy as np
import pandas as pd
import multiprocessing
from copy import deepcopy
import matplotlib.pyplot as plt

from torch.optim import AdamW
from torch.nn import MSELoss
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from train_and_eval import train, evaluate
from metadata_utils import encode_metadata
from CytokineTransputerSparse import CytokineTransputerSparse
from graph_utils import data, cytokines, edge_index, edge_weights
from dataset_precomputed_mask import CytokinewithPEandMask, PerNodeZScoreTransform, subject_to_dsall_indices

# Ensure that child processes use fork on macOS to avoid spawn-related bootstrapping errors
multiprocessing.set_start_method("fork", force=True)
sys.stderr = open(os.devnull, "w")

# Directories
base_dir = os.path.dirname(__file__)
graph_dir = os.path.join(base_dir, "Precomputed_graphs")
vis_dir = os.path.join(base_dir, "Visuals")
result_dir = os.path.join(base_dir, "Results")
models_dir = os.path.join(base_dir, "Models")

# Precompute per-channel mu_log, sigma-_og
# Build a (num_subjects x num_channels) array of log(y+1)
all_vals = data[cytokines].to_numpy(dtype=np.float32)  # [n_subjects,55] = [968,55]
all_logs = np.log(all_vals + 1.0)
valid = np.isfinite(all_logs)
# Compute pre-channel mean and std on only the finite entries:
mu_log = np.nanmean(all_logs, axis=0)  # [55]
sigma_log = np.nanmean(all_logs, axis=0)  # [55]
# Convert to torch.tensor
mu_log_tensor = torch.tensor(mu_log, dtype=torch.float32)
sigma_log_tensor = torch.tensor(sigma_log, dtype=torch.float32)


# Standardise cytokine concentrations to zero-mean , unit-variance
means = data[cytokines].mean().to_numpy(dtype=np.float32)
stds = data[cytokines].std().to_numpy(dtype=np.float32)

# Build the dataset
ds = CytokinewithPEandMask(
    df=data,
    cytokine_list=cytokines,
    edge_index=edge_index,
    edge_weights=edge_weights,
    meta_enc_fn=encode_metadata,
    root=graph_dir,
    force_reload=True,
    K=4,
    n_variations=10,
    in_meta_dim=7,  # len(encode_metadata(row)),
    proj_meta_dim=16,
    pre_transform=PerNodeZScoreTransform(means, stds),
    transform=None,
)

print(f"Loaded {len(ds)} premasked graphs.")

# # Split into train/val
n = len(data)
t, v = int(0.7 * n), int(0.15 * n)
df_train, df_val, df_test = random_split(data, [t, v, n - t - v], generator=torch.Generator().manual_seed(42))
df_train = data.iloc[df_train.indices].reset_index(drop=True)
df_val = data.iloc[df_val.indices].reset_index(drop=True)

train_ids = set(df_train["subject_accession"])  # to filter out subjects in the training set
val_ids = set(df_val["subject_accession"])  # to filter out subjects in the validation set

# Maps each subject_accession back to its row index in data
subj_to_idx = {row.subject_accession: idx for idx, row in enumerate(data.itertuples(index=False))}

# Collect all the 10 corresponding indices in ds_all
train_idx = []
for sid in train_ids:
    subj_idx = subj_to_idx[sid]
    train_idx += subject_to_dsall_indices(subj_idx, n_variations=10)

val_idx = []
for sid in val_ids:
    subj_idx = subj_to_idx[sid]
    val_idx += subject_to_dsall_indices(subj_idx, n_variations=10)

ds_train = Subset(ds, train_idx)
ds_val = Subset(ds, val_idx)

train_loader = DataLoader(
    ds_train, batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(42), num_workers=3, prefetch_factor=4
)
val_loader = DataLoader(ds_val, batch_size=16, shuffle=False, num_workers=3, prefetch_factor=4)

mask_rates_train = [g.mask.sum().item() / g.orig_mask.sum().item() for g in ds_train]
mask_rates_val = [g.mask.sum().item() / g.orig_mask.sum().item() for g in ds_val]
print(
    f"Train-set mask rates: mean={np.mean(mask_rates_train):.2f}| min={np.min(mask_rates_train):.2f}| max={np.max(mask_rates_train):.2f}"
)
print(
    f"Val-set mask rates: mean={np.mean(mask_rates_val):.2f}| min={np.min(mask_rates_val):.2f}| max={np.max(mask_rates_val):.2f}"
)
# ======= CytokineTransputerSparse ========
device = torch.device("mps")
in_ch = 1 + 7 + 1  # [conc + proj_meta_dim+ log_deg]
E = ds_train[0].edge_index.size(1)
model_static = CytokineTransputerSparse(in_ch=in_ch, meta_dim=7, d_h=64, edge_dim=4, heads=4, num_layers=3).to(device)

# Optimizer and loss
optimizer = AdamW(model_static.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
loss_fn = torch.nn.SmoothL1Loss(beta=1.0)
epochs = 50

best_val_loss = float("inf")
best_val_tolacc = 0.0
best_path = os.path.join(models_dir, f"CytokineTransputerSparse_population_.pth")

train_logloss_runs = []
val_logloss_runs = []
mae = []
tolacc = []
r2 = []

start_time = time.time()
for epoch in range(1, epochs + 1):
    # Train
    train_loss = train(model_static, train_loader, mu_log_tensor, sigma_log_tensor, optimizer, loss_fn, device)
    train_logloss_runs.append(train_loss)

    # Validate
    metrics = evaluate(model_static, val_loader, mu_log_tensor, sigma_log_tensor, loss_fn, device)
    val_logloss = metrics["log-loss"]
    val_logloss_runs.append(val_logloss)
    tol_acc = metrics["TolAcc"]
    mae.append(metrics["MAE"])
    r2.append(metrics["R2"])
    tolacc.append(tol_acc)

    scheduler.step(val_logloss)

    print(f"Epoch {epoch}: train: {train_loss:.2f}| val: {val_logloss:.2f}| tolacc: {tol_acc:.2f}")

    if val_logloss < best_val_loss:
        best_val_loss = val_logloss
        best_val_tolacc = tol_acc
torch.save(model_static.state_dict(), best_path)
print(f"Epoch {epoch}: val_logloss {val_logloss:.2f} -> saved new best model")

print(f"Best validation MSE={best_val_loss:.2f}| Tolacc = {best_val_tolacc}")
# total_time = int(time.time() - start_time) / 60
# print(f"Total runtime: {total_time} minutes")
summary = {
    "Train loss": train_logloss_runs,
    "Val loss": val_logloss_runs,
    "MAE": mae,
    "R2": r2,
    "TolAcc": tolacc,
    "Best val loss": best_val_loss,
    "Best val tolacc": best_val_tolacc,
}
summary = pd.DataFrame(summary)
summary.to_csv(os.path.join(result_dir, "Population_CytokineTransputerSparse_meta.csv"), index=False)

# Extract the weight matrix of the first linear
W = model_static.meta_proj[0].weight.detach().abs()  # [d_h, meta_dim]
importances = W.sum(dim=0).cpu().numpy()  # [meta_dim]
covariates = ["White", "Other", "Asian", "Black or African American", "Male", "Female", "Age"]
df_imp = pd.DataFrame({"Covariate": covariates, "Importance": importances}).sort_values("Importance", ascending=False)
print(df_imp)


# ======= Grid search for delta of Huber Loss ========
# delta = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

# results = {}
# for d in delta:
#     results[f"result_{d}"] = {
#         "train_logloss_runs": [],
#         "val_logloss_runs": [],
#         "mae": [],
#         "tolacc": [],
#         "r2": [],
#     }

# for d in delta:
#     print(f"\n=== Training with Huber Δ={d} ===")
#     loss_fn = torch.nn.SmoothL1Loss(beta=d)

#     best_val = float("inf")
#     for epoch in range(1, epochs + 1):
#         train_loss = train(model_static, train_loader, mu_log_tensor, sigma_log_tensor, optimizer, loss_fn, device)
#         results[f"result_{d}"]["train_logloss_runs"].append(train_loss)

#         metrics = evaluate(model_static, val_loader, mu_log_tensor, sigma_log_tensor, loss_fn, device)
#         val_logloss = metrics["log-loss"]
#         results[f"result_{d}"]["val_logloss_runs"].append(val_logloss)
#         tol_acc = metrics["TolAcc"]
#         results[f"result_{d}"]["tolacc"].append(tol_acc)
#         mae_val = metrics["MAE"]
#         results[f"result_{d}"]["mae"].append(mae_val)
#         r2_val = metrics["R2"]
#         results[f"result_{d}"]["r2"].append(r2_val)

#         scheduler.step(val_logloss)

#         if epoch % 2 == 0:
#             print(f"Δ={d} Epoch {epoch}: train={train_loss:.3f}| val={val_logloss:.3f}| tolacc={tol_acc:.3f}")

#         best_val = min(best_val, val_logloss)
#     print(f"Best val loss @Δ={d}: {best_val:.3f}")

#     result_df = pd.DataFrame(
#         {
#             "Epoch": list(range(1, epochs + 1)),
#             "Train Loss": results[f"result_{d}"]["train_logloss_runs"],
#             "Val Loss": results[f"result_{d}"]["val_logloss_runs"],
#             "MAE": results[f"result_{d}"]["mae"],
#             "TolAcc": results[f"result_{d}"]["tolacc"],
#             "R2": results[f"result_{d}"]["r2"],
#         }
#     )
#     result_path = os.path.join(result_dir, f"Results_CytokineTransputerSparse_delta_{d}.csv")
#     result_df.to_csv(result_path, index=False)
#     print(f"Saved results for Δ={d} to {result_path}")

# # ====== Visualising delta grid-search ========
# x = np.arange(1, epochs + 1)
# fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=100)
# delta = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
# # Plot val loss
# for d in delta:
#     sum_df = pd.read_csv(os.path.join(result_dir, f"Results_CytokineTransputerSparse_delta_{d}.csv"))
#     axes[0, 0].plot(x, sum_df["Val Loss"], marker="o", markersize=5, label=f"Δ={d}")
#     axes[0, 0].fill_between(
#         x,
#         sum_df["Val Loss"] - 1.96 * sum_df["Val Loss"].std() / np.sqrt(len(sum_df["Val Loss"])),
#         sum_df["Val Loss"] + 1.96 * sum_df["Val Loss"].std() / np.sqrt(len(sum_df["Val Loss"])),
#         alpha=0.2,
#     )
# axes[0, 0].set_ylabel("Huber Loss on log(y+1)")
# axes[0, 0].set_title("Val Loss")
# axes[0, 0].legend()

# # Plot MAE
# for d in delta:
#     sum_df = pd.read_csv(os.path.join(result_dir, f"Results_CytokineTransputerSparse_delta_{d}.csv"))
#     axes[0, 1].plot(x, sum_df["MAE"], marker="o", markersize=5, label=f"Δ={d}")
# axes[0, 1].set_ylabel("MAE")
# axes[0, 1].set_title("Mean Absolute Error")
# axes[0, 1].legend()

# # Plot R2
# for d in delta:
#     sum_df = pd.read_csv(os.path.join(result_dir, f"Results_CytokineTransputerSparse_delta_{d}.csv"))
#     axes[1, 0].plot(x, sum_df["R2"], marker="o", markersize=5, label=f"Δ={d}")
# axes[1, 0].set_ylabel("R2")
# axes[1, 0].set_title("R2 Score")
# axes[1, 0].legend()

# # Plot TolAcc
# for d in delta:
#     sum_df = pd.read_csv(os.path.join(result_dir, f"Results_CytokineTransputerSparse_delta_{d}.csv"))
#     axes[1, 1].plot(x, sum_df["TolAcc"], marker="o", markersize=5, label=f"Δ={d}")
# axes[1, 1].set_ylabel("Tolerance Accuracy")
# axes[1, 1].set_title("Tolerance Accuracy")
# axes[1, 1].legend()

# fig.supxlabel("Epochs")
# fig.suptitle("Grid Search for Δ of Huber Loss", fontsize=16)
# fig.tight_layout()
# plt.savefig(os.path.join(vis_dir, "Grid_search_delta_huber_loss.png"), bbox_inches="tight")
# plt.show()
