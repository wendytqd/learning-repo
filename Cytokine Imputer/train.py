import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    used_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.edge_attr.squeeze(1))
        true = batch.x[:, 0]

        mask = batch.mask.squeeze(1)
        if mask.sum() == 0:
            continue

        loss = loss_fn(pred[mask], true[mask])
        loss.backward()
        optimizer.step()  # update weights

        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, loss_fn, abs_tol=None, rel_tol=0.1):  # change abs_tol/rel_tol based on research
    model.eval()
    total_mse = 0
    used_batches = 0

    all_pred, all_true = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            pred = model(batch.x, batch.edge_index, batch.edge_attr.squeeze(1))
            true = batch.x[:, 0]

            mask = batch.mask.squeeze(1)
            if mask.sum() == 0:  # Skip graphs that have no hidden values to score
                continue

            # MSE
            loss = loss_fn(pred[mask], true[mask])
            total_mse += loss.item()
            used_batches += 1

            # Stash for MAE/accuracy
            all_pred.append(pred[mask].cpu())
            all_true.append(true[mask].cpu())

    if used_batches == 0:  # count only batches that contained masked nodes
        return float("nan"), float("nan"), float("nan")
    # otherwise, avg_mse is biased when some graphs have nothing to impute.

    avg_mse = total_mse / used_batches

    all_pred = torch.cat(all_pred).numpy()
    all_true = torch.cat(all_true).numpy()

    mae = mean_absolute_error(all_true, all_pred)
    r2 = r2_score(all_true, all_pred)

    # Tolerace accuracy
    if abs_tol is None:
        tol_mask = np.abs(all_pred - all_true) <= rel_tol * np.abs(all_true)
    else:
        tol_mask = np.abs(all_pred - all_true) <= abs_tol
    acc_tol = tol_mask.mean()

    return {"MSE": avg_mse, "RMSE": np.sqrt(avg_mse), "MAE": mae, "R2": r2, "TolAcc": acc_tol}
