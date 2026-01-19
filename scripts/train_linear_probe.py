import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import (
    DataLoader,
)
from sklearn.metrics import roc_curve, auc


def compute_mean_probe_activations(train_probe_dataloader: DataLoader) -> torch.Tensor:
    features = []
    with torch.no_grad():
        for activations_batch, _ in train_probe_dataloader:
            features.append(activations_batch.float().cpu())

    X = torch.cat(features, dim=0)  # X shape: (N, d_model)

    # Mean centering
    X_mean = X.mean(dim=0, keepdim=True)  # shape: (1, d_model)

    # X_mean shape: (1, d_model)
    return X_mean.float()


def train_steering_linear_probe(
    probe_model: nn.Module,
    train_probe_dataloader: DataLoader,
    val_probe_dataloader: DataLoader,
    test_probe_dataloader: DataLoader,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 15,
    layer: int = 18,
    use_calibrated_threshold: bool = True,
    checkpoint_path: str = "steering_probe_18_epoch_15.pt",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> tuple[nn.Module, float, float, float, float, torch.Tensor]:
    X_mean = compute_mean_probe_activations(
        train_probe_dataloader
    )  # shape: (1, d_model)
    X_mean = X_mean.to(device)

    optimizer = optim.AdamW(
        params=probe_model.parameters(), lr=lr, weight_decay=weight_decay
    )

    loss_fn = nn.BCEWithLogitsLoss()

    start_epoch = 0
    probe_threshold = 0.5

    if os.path.exists(checkpoint_path):
        print("Resuming training from checkpoint...")
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        probe_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        probe_threshold = checkpoint["threshold"]

    best_auc = 0.0
    best_val_acc = 0.0
    best_test_acc = 0.0

    def compute_auc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
        with torch.no_grad():
            # Approximate via rank statistic (Mann–Whitney U / ROC AUC equivalence)

            y_true = y_true.cpu().float()
            y_score = y_score.cpu().float()

            pos = y_true == 1
            neg = y_true == 0

            if pos.sum() == 0 or neg.sum() == 0:
                return 0.5

            # Ranks
            _, order = torch.sort(y_score)
            ranks = torch.empty_like(order, dtype=torch.float32)
            ranks[order] = torch.arange(1, len(y_score) + 1, dtype=torch.float32)

            ranks_pos = ranks[pos]

            n_pos = ranks_pos.numel()
            n_neg = (neg).sum().item()

            U = ranks_pos.sum().item() - n_pos * (n_pos + 1) / 2.0
            auc = U / (n_pos * n_neg + 1e-8)

            return float(auc)

    def per_class_accuracy(
        y_true: torch.Tensor, y_pred: torch.Tensor
    ) -> tuple[float, float]:
        y_true = y_true.int().cpu()
        y_pred = y_pred.int().cpu()

        true_positives = ((y_true == 1) & (y_pred == 1)).sum().item()
        true_negatives = ((y_true == 0) & (y_pred == 0)).sum().item()
        false_positives = ((y_true == 0) & (y_pred == 1)).sum().item()
        false_negatives = ((y_true == 1) & (y_pred == 0)).sum().item()

        harmful_accuracy = true_positives / (true_positives + false_negatives + 1e-8)
        benign_accuracy = true_negatives / (true_negatives + false_positives + 1e-8)

        return benign_accuracy, harmful_accuracy

    def plot_roc_with_thresholds(
        fpr: np.ndarray, tpr: np.ndarray, thr: np.ndarray, annotate_every: int = 12
    ) -> None:
        roc_auc = auc(fpr, tpr)

        plt.figure()

        # Color points by threshold value
        scatter = plt.scatter(fpr, tpr, c=thr, s=18)
        plt.plot(fpr, tpr, linewidth=1)

        # Diagonal reference
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

        # Annotate every Nth point with its threshold
        for i in range(0, len(thr), max(1, len(thr) // annotate_every)):
            plt.annotate(
                f"{thr[i]:.2f}",
                (fpr[i], tpr[i]),
                textcoords="offset points",
                xytext=(5, -6),
                fontsize=8,
            )

        colorbar = plt.colorbar(scatter)
        colorbar.set_label("Threshold")

        plt.xlim(0, 1)
        plt.ylim(0, 1.05)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve (AUC={roc_auc:.3f})")
        plt.show()

    def get_best_threshold_youden(
        fpr: np.ndarray, tpr: np.ndarray, thr: np.ndarray
    ) -> tuple[float, float, float, float]:
        J = tpr - fpr
        best_idx = np.argmax(J)

        return thr[best_idx], fpr[best_idx], tpr[best_idx], J[best_idx]

    for epoch in tqdm(range(start_epoch, epochs), desc=f"Training for {epochs} epochs"):
        probe_model.train()
        total_loss = 0.0

        for activations_batch, labels_batch in tqdm(
            train_probe_dataloader, desc=f"Training epoch {epoch + 1}"
        ):
            activations_batch, labels_batch = activations_batch.to(
                device
            ), labels_batch.to(device)

            activations_batch = activations_batch - X_mean

            logits = probe_model(activations_batch).squeeze(-1)  # shape: (B)
            loss = loss_fn(logits, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * activations_batch.size(0)

        # Validation
        probe_model.eval()
        all_labels = []
        all_probs = []

        with torch.inference_mode():
            for activations_batch, labels_batch in tqdm(
                val_probe_dataloader, desc=f"Validation epoch {epoch + 1}"
            ):
                activations_batch = activations_batch.to(device)

                activations_batch = activations_batch - X_mean

                logits = probe_model(activations_batch).squeeze(-1)
                probs = torch.sigmoid(logits)

                all_labels.append(labels_batch)
                all_probs.append(probs)

        all_labels = torch.cat(all_labels, dim=0)
        all_probs = torch.cat(all_probs, dim=0)
        val_auc = compute_auc(all_labels, all_probs)

        if use_calibrated_threshold:
            y_val_np = all_labels.cpu().numpy().astype(int)
            p_val_np = all_probs.cpu().numpy()

            fpr, tpr, thr = roc_curve(y_val_np, p_val_np)

            print(fpr, tpr, thr)

            plot_roc_with_thresholds(fpr, tpr, thr)

            probe_threshold, best_fpr, best_tpr, best_J = get_best_threshold_youden(
                fpr, tpr, thr
            )

            print(
                f"Using a calibrated threshold of {probe_threshold} with FPR: {best_fpr}, TPR: {best_tpr}, and J: {best_J}"
            )

        val_preds = (all_probs >= probe_threshold).float()

        # Calculate accuracy
        val_acc = (val_preds.cpu() == all_labels.cpu()).float().mean().item()
        val_benign_acc, val_harmful_acc = per_class_accuracy(all_labels, val_preds)

        # Test
        probe_model.eval()
        all_test_y = []
        all_test_p = []

        with torch.inference_mode():
            for activations_batch, labels_batch in tqdm(
                test_probe_dataloader, desc=f"Testing epoch {epoch + 1}"
            ):
                activations_batch = activations_batch.to(device)

                activations_batch = activations_batch - X_mean

                logits = probe_model(activations_batch).squeeze(-1)
                probs = torch.sigmoid(logits)

                all_test_y.append(labels_batch)
                all_test_p.append(probs)

        all_test_y = torch.cat(all_test_y, dim=0)
        all_test_p = torch.cat(all_test_p, dim=0)
        test_auc = compute_auc(all_test_y, all_test_p)

        # Use a threshold of 0.5 or the calibrated threshold from above
        test_preds = (all_test_p >= probe_threshold).float()

        # Calculate accuracy
        test_acc = (test_preds.cpu() == all_test_y.cpu()).float().mean().item()
        test_benign_acc, test_harmful_acc = per_class_accuracy(all_test_y, test_preds)

        print(
            f"Epoch {epoch + 1} | Train Loss: {total_loss / len(train_probe_dataloader.dataset):.4f} "
            f"| Validation AUC: {val_auc:.4f} | Validation Accuracy: {val_acc:.4f} "
            f"| Val Benign Acc: {val_benign_acc:.4f} | Val Harmful Acc: {val_harmful_acc:.4f} "
            f"| Test AUC: {test_auc:.4f} | Test Accuracy: {test_acc:.4f} "
            f"| Test Benign Acc: {test_benign_acc:.4f} | Test Harmful Acc: {test_harmful_acc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": probe_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "threshold": probe_threshold,
            "X_mean": X_mean,
        }

        torch.save(
            checkpoint,
            f"steering_probe_{layer}_epoch_{epoch + 1}.pt",
        )

    return probe_model, probe_threshold, best_auc, best_val_acc, best_test_acc, X_mean
