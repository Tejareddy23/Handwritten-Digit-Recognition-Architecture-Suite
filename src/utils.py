"""
Utility functions: reproducibility seeds, metrics plotting, and visualization.
"""

import os
import random
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import RESULTS_DIR


def set_seed(seed=42):
    """
    Sets deterministic seeds across Python, NumPy, and PyTorch.
    Ensures experimental reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def plot_training_history(history, save_path=None):
    """
    Plots training and validation loss and accuracy curves over epochs.
    """
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "training_history.png")

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#ff7f0e", linewidth=2)
    axes[0].set_title("Cross-Entropy Loss vs. Epochs", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=11)

    # Accuracy plot
    axes[1].plot(epochs, history["train_acc"], "o-", label="Train Accuracy", color="#2ca02c", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], "s--", label="Val Accuracy", color="#d62728", linewidth=2)
    axes[1].set_title("Accuracy vs. Epochs", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy (%)", fontsize=12)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization] Training curves saved to: {save_path}")


def plot_confusion_matrix(cm, class_names=None, save_path=None):
    """
    Plots a high-resolution confusion matrix heatmap.
    """
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")

    if class_names is None:
        class_names = [str(i) for i in range(10)]

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    plt.title("Confusion Matrix on Test Set", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization] Confusion matrix saved to: {save_path}")


def plot_sample_predictions(images, true_labels, pred_labels, probs=None, save_path=None, num_samples=12):
    """
    Visualizes a grid of test predictions with actual vs predicted labels and confidence.
    """
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "sample_predictions.png")

    num_samples = min(num_samples, len(images))
    rows = (num_samples + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows))
    axes = axes.flatten()

    for i in range(num_samples):
        img = images[i].squeeze()
        # De-normalize MNIST for display (x * std + mean)
        img = img * 0.3081 + 0.1307
        img = np.clip(img, 0, 1)

        true_lbl = true_labels[i]
        pred_lbl = pred_labels[i]
        is_correct = (true_lbl == pred_lbl)
        color = "green" if is_correct else "red"

        title = f"True: {true_lbl} | Pred: {pred_lbl}"
        if probs is not None:
            title += f"\nConf: {probs[i]*100:.1f}%"

        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(title, fontsize=10, color=color, fontweight="bold")
        axes[i].axis("off")

    for j in range(num_samples, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization] Sample predictions saved to: {save_path}")


def plot_model_comparison(comparison_data, save_path=None):
    """
    Plots a side-by-side bar chart comparing MLP vs CNN on Accuracy, Loss, and Parameter efficiency.
    """
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "model_comparison.png")

    models = ["MLP (Feedforward)", "CNN (Convolutional)"]
    accuracies = [comparison_data["mlp"]["test_acc"], comparison_data["cnn"]["test_acc"]]
    losses = [comparison_data["mlp"]["test_loss"], comparison_data["cnn"]["test_loss"]]
    params = [comparison_data["mlp"]["params"], comparison_data["cnn"]["params"]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Accuracy
    axes[0].bar(models, accuracies, color=["#3498db", "#2ecc71"], width=0.5)
    axes[0].set_title("Test Accuracy (%)", fontsize=12, fontweight="bold")
    axes[0].set_ylim(90, 100)
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 0.5, f"{v:.2f}%", ha="center", fontweight="bold")

    # Loss
    axes[1].bar(models, losses, color=["#e67e22", "#e74c3c"], width=0.5)
    axes[1].set_title("Test Loss (Cross-Entropy)", fontsize=12, fontweight="bold")
    for i, v in enumerate(losses):
        axes[1].text(i, v + 0.005, f"{v:.4f}", ha="center", fontweight="bold")

    # Parameter Count
    axes[2].bar(models, [p / 1000 for p in params], color=["#9b59b6", "#1abc9c"], width=0.5)
    axes[2].set_title("Parameter Count (in Thousands)", fontsize=12, fontweight="bold")
    for i, v in enumerate(params):
        axes[2].text(i, (v / 1000) + 5, f"{v:,}", ha="center", fontweight="bold")

    plt.suptitle("Architecture Benchmark: MLP vs. CNN", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization] Architecture comparison chart saved to: {save_path}")


def save_history_json(history, save_path=None):
    """
    Saves metrics history dict to a JSON file.
    """
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(save_path, "w") as f:
        json.dump(history, f, indent=4)

