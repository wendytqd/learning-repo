import torch
from torch.nn import MSELoss
from torch_geometric.data import Batch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ignite.handlers import EarlyStopping
from data_loader import CytokineSubjectDataset, prepare_loaders
from model import CytokineImputer
from train import train, evaluate
from metadata_utils import encode_metadata
from graph_utils import data, cytokines, edge_index, edge_weights
import matplotlib.pyplot as plt

# Define dataset
dataset = CytokineSubjectDataset(
    df=data,
    cytokine_list=cytokines,
    edge_index=edge_index,
    edge_weights=edge_weights,
    meta_enc_fn=encode_metadata,
)

# Prepare loaders
train_loader, val_loader, test_loader = prepare_loaders(dataset)

# Define the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_channels = dataset.node_feat_dim
model = CytokineImputer(in_channels=in_channels, hidden_channels=64).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
loss_fn = MSELoss()

scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

epochs = 50
history = {"train_mse": [], "val_mse": [], "val_mae": [], "val_r2": [], "val_tolacc": []}

for epoch in range(1, epochs + 1):
    # training
    train_loss = train(model, train_loader, optimizer, loss_fn)
    history["train_mse"].append(train_loss)
    # validation
    val_metrics = evaluate(model, val_loader, loss_fn, abs_tol=None, rel_tol=0.1)
    val_mse = val_metrics["MSE"]
    val_mae = val_metrics["MAE"]
    val_r2 = val_metrics["R2"]
    val_tolacc = val_metrics["TolAcc"]

    history["val_mse"].append(val_mse)
    history["val_mae"].append(val_mae)
    history["val_r2"].append(val_r2)
    history["val_tolacc"].append(val_tolacc)

    scheduler.step(val_mse)

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch}/{epochs}:"
            f"Train MSE: {train_loss:.3f}| "
            f"Val MSE: {val_mse:.3f}| "
            f"MAE: {val_mae:.3f}| "
            f"R2: {val_r2:.6f}| "
            f"TolAcc: {val_tolacc:.3f}"
        )

plt.figure(figsize=(16, 10))
plt.subplot(2, 2, 1)
plt.plot(history["train_mse"], marker="o", label="Train")
plt.plot(history["val_mse"], marker="o", label="Val")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(history["val_mae"], marker="o")
plt.xlabel("Epochs")
plt.ylabel("MAE")

plt.subplot(2, 2, 3)
plt.plot(history["val_r2"], marker="o")
plt.xlabel("Epochs")
plt.ylabel("R^2")

plt.subplot(2, 2, 4)
plt.plot(history["val_tolacc"], marker="o")
plt.xlabel("Epochs")
plt.ylabel("Tolerace accuracy")

plt.tight_layout()
plt.show()
