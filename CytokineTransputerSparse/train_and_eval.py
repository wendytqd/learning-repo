import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def train(model, loader, mu_log_tensor, sigma_log_tensor, optimizer, loss_fn, device, n_nodes=55):
    model.to(device)
    model.train()
    total_loss, batch_count = 0.0, 0.0

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        x_raw = batch.x  # [55, in_ch]
        edge_index = batch.edge_index  # [2, B*E]
        edge_attr = batch.edge_attr  # [B*E,K
        y_true = batch.y.squeeze(1)  # [55]  true concentration of all nodes # batch.y
        mask = batch.mask  # [55]

        pred_log = model(x_raw, edge_index, edge_attr, batch.batch)

        # Compute raw log for every node
        raw_log = torch.log(y_true + 1.0)  # [B*55]

        # Figure out channel index of each node in the batch
        N = raw_log.size(0)  # [B*N_nodes]
        node_ids = torch.arange(N, device=device)
        channel_idxs = (node_ids % n_nodes).long()

        # fetch that channel's mu and sigma
        mu_c = mu_log_tensor.to(device)[channel_idxs]
        sigma_c = sigma_log_tensor.to(device)[channel_idxs]

        # Standardise both prediction and truth
        pred_std = (pred_log - mu_c) / (sigma_c + 1e-8)
        true_std = (raw_log - mu_c) / (sigma_c + 1e-8)

        masked_idxs = mask.nonzero(as_tuple=False).view(-1)  # indices where mask == True
        if masked_idxs.numel() == 0:
            # no masked nodes in this batch
            continue

        t_pred_std = pred_std[masked_idxs]
        t_true_std = true_std[masked_idxs]

        # Compute Huber loss on standaridsed predictions vs truth
        optimizer.zero_grad()
        loss = loss_fn(t_pred_std, t_true_std)

        # # Compute loss only on masked nodes:
        # t_pred = pred_log[mask]  # predictions for masked nodes
        # t_true = torch.log(y_true[mask] + 1.0).squeeze(-1)  # ground truth log(y+1) for masked
        # loss = loss_fn(t_pred, t_true)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

    avg_loss = total_loss / batch_count if batch_count > 0 else float("nan")
    return avg_loss


def evaluate(
    model, loader, mu_log_tensor, sigma_log_tensor, loss_fn, device, n_nodes=55, abs_tol=None, rel_tol=0.07
):  # allow 7% error as the ground truth will becoming from the singleplex ELISA
    model.eval()
    total_loss = 0
    used_batches = 0

    all_pred, all_true = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            x_raw = batch.x
            # P = batch.P  # [55, 55, K]
            # d_vec = batch.d  # [2,E]
            edge_index = batch.edge_index
            edge_attr = batch.edge_attr
            y_true = batch.y.squeeze(1)  # [55,1] true concentration of all nodes
            drop_mask = batch.mask  # [55]
            orig_known = batch.orig_mask

            pred_log = model(x_raw, edge_index, edge_attr, batch.batch)

            # Compute raw log for every node
            raw_log = torch.log(y_true + 1.0)  # [B*55]

            # Figure out channel index of each node in the batch
            N = raw_log.size(0)  # [B*N_nodes]
            node_ids = torch.arange(N, device=device)
            channel_idxs = (node_ids % n_nodes).long()

            # fetch that channel's mu and sigma
            mu_c = mu_log_tensor.to(device)[channel_idxs]
            sigma_c = sigma_log_tensor.to(device)[channel_idxs]

            # Standardise both prediction and truth
            pred_std = (pred_log - mu_c) / (sigma_c + 1e-8)
            true_std = (raw_log - mu_c) / (sigma_c + 1e-8)

            eval_mask = drop_mask & orig_known  # use the original mask to evaluate the model
            if eval_mask.sum() == 0:  # Skip graphs that have no hidden values to score
                continue

            idxs = eval_mask.nonzero(as_tuple=False).view(-1)
            t_pred_std = pred_std[idxs]
            t_true_std = true_std[idxs]
            loss = loss_fn(t_pred_std, t_true_std)

            # MSE
            # t_pred = pred_log[eval_mask]
            # t_true = torch.log(y_true[eval_mask] + 1).squeeze(-1)
            # loss = loss_fn(t_pred, t_true)

            total_loss += loss.item()
            used_batches += 1

            # Stash for MAE/accuracy
            raw_pred_i = (torch.exp(pred_log[idxs]) - 1).cpu().numpy()  # back to the original concentration
            raw_true_i = y_true[idxs].cpu().numpy()

            all_pred.append(raw_pred_i)
            all_true.append(raw_true_i)

    if used_batches == 0:  # count only batches that contained masked nodes
        return {"log-MSE": float("nan"), "MAE": float("nan"), "R2": float("nan"), "TolAcc": float("nan")}

    avg_loss = total_loss / used_batches

    all_pred = np.concatenate(all_pred).ravel()  # ravel to flatten the predictions
    all_true = np.concatenate(all_true).ravel()  # ravel to flatten the true values
    valid = np.isfinite(all_true) & np.isfinite(all_pred)  # filter out NaNs in the true values
    all_pred = all_pred[valid]  # filter predictions
    all_true = all_true[valid]  # filter true values

    if all_true.size == 0:
        return {"log-loss": float("nan"), "MAE": float("nan"), "R2": float("nan"), "TolAcc": float("nan")}

    mae = mean_absolute_error(all_true, all_pred)
    r2 = r2_score(all_true, all_pred)

    # Tolerace accuracy
    if abs_tol is None:
        tol_mask = np.abs(all_pred - all_true) <= rel_tol * np.abs(all_true)
    else:
        tol_mask = np.abs(all_pred - all_true) <= abs_tol
    acc_tol = tol_mask.mean()

    return {"log-loss": avg_loss, "MAE": mae, "R2": r2, "TolAcc": acc_tol}
