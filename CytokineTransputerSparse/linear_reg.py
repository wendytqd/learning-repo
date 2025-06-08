import os, torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from torch.utils.data import random_split, Subset

from metadata_utils import encode_metadata
from dataset_precomputed_mask import CytokinewithPEandMask, PerNodeZScoreTransform
from graph_utils import data, cytokines, edge_index, edge_weights
from data_loader import subject_to_dsall_indices

# Directories
base_dir = os.path.dirname(__file__)
graph_dir = os.path.join(base_dir, "Precomputed_graphs")
vis_dir = os.path.join(base_dir, "Visuals")
result_dir = os.path.join(base_dir, "Results")
models_dir = os.path.join(base_dir, "Models")

# Precompute per-cytokine log mean and std
# Build a (num_subjects x num_channels) array of log(y+1)
all_vals = data[cytokines].to_numpy(dtype=np.float32)  # [n_subjects,55] = [968,55]
all_logs = np.log(all_vals + 1.0)
valid = np.isfinite(all_logs)
# Compute pre-channel mean and std on only the finite entries:
mu_log = np.nanmean(all_logs, axis=0)  # [55]
sigma_log = np.nanmean(all_logs, axis=0)  # [55]

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
train_idx, val_idx = [], []
for sid in train_ids:
    subj_idx = subj_to_idx[sid]
    train_idx += subject_to_dsall_indices(subj_idx, n_variations=10)

for sid in val_ids:
    subj_idx = subj_to_idx[sid]
    val_idx += subject_to_dsall_indices(subj_idx, n_variations=10)

ds_train = Subset(ds, train_idx)
ds_val = Subset(ds, val_idx)


def extract_tabular_data(dataset):
    X_list, y_list, c_idx_list = [], [], []

    for graph in dataset:
        x = graph.x  # [55]
        y = graph.y.squeeze(1)  # [55]
        mask = graph.mask  # [55]

        for i in range(len(mask)):
            if mask[i]:
                X_list.append(x[i].numpy())
                y_list.append(y[i].item())
                c_idx_list.append(i)
    return np.vstack(X_list), np.array(y_list), np.array(c_idx_list)


g = ds_train[0]

X_train, y_train, c_train = extract_tabular_data(ds_train)
print("LinReg sees masked X[:,0]:", X_train[:10, 0].tolist())
X_val, y_val, c_val = extract_tabular_data(ds_val)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

log_std_true = (np.log1p(y_val) - mu_log[c_val]) / (sigma_log[c_val] + 1e-8)
log_std_pred = (np.log1p(y_pred) - mu_log[c_val]) / (sigma_log[c_val] + 1e-8)


mse = mean_squared_error(log_std_true, log_std_pred)
r2 = r2_score(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)

rel_tol = 0.07
tolacc = np.abs(y_pred - y_val) <= rel_tol * np.abs(y_val)
tol_acc = tolacc.mean()


def huber_loss(true, pred, delta=1.0):
    error = pred - true
    small_error = np.abs(error) <= delta
    squared_loss = 0.5 * error**2
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.mean(np.where(small_error, squared_loss, linear_loss))


huber = huber_loss(log_std_true, log_std_pred, delta=1.0)

print(f"Validation MSE: {mse:.4f}")
print(f"Validation R²: {r2:.4f}")
print(f"Validation MAE: {mae:.4f}")
print(f"Validation Tolerance accuracy: {tol_acc:.4f}")
print(f"validation Huber Loss: {huber:.4f}")

result_dict = {
    "Actual": y_val,
    "Predicted": y_pred,
    "Stand. actual": log_std_true,
    "Stand. predicted": log_std_pred,
    "MSE": mse,
    "R2": r2,
    "MAE": mae,
    "TolAcc": tol_acc,
    "Huber": huber,
}
results = pd.DataFrame(result_dict)
results.to_csv(os.path.join(result_dir, "Linear_regression_results.csv"))
print(results.head())

plt.figure(figsize=(8, 5))
metrics_dict = {k: v for k, v in result_dict.items() if np.isscalar(v)}
labels = list(metrics_dict.keys())
values = list(metrics_dict.values())
plt.barh(labels, values)
for i, v in enumerate(values):
    plt.text(v + 0.01, i, f"{v:.4f}", va="center")
plt.xlabel("Score/Error")
plt.title("Linear Regression (log-standardised)")
plt.grid(True, axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, "Linear_regression.png"))

plt.figure(figsize=(6, 6))
plt.scatter(log_std_true, log_std_pred, alpha=0.5)
plt.plot([-1, 1.75], [-1, 1.75], "r--")  # assuming z-scored data
plt.xlabel("True (log-standardised)")
plt.ylabel("Predicted")
plt.title("Linear Regression: Predicted vs True Cytokine Concentrations")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, "linear_regression_scatter_standardised.png"))
plt.show()
