"""
Deep Learning Evaluation Module.

Evaluates trained deep learning models on unseen test dataset.
Computes:
- Overall Test Accuracy and Cross-Entropy Loss
- Precision, Recall, and F1-Score per class (Scikit-Learn classification report)
- Multi-class Confusion Matrix Heatmap
- Sample Visualizations with prediction confidence
"""

import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.config import DEVICE, MODEL_DIR, RESULTS_DIR
from src.models import build_model
from src.dataset import get_dataloaders
from src.utils import plot_confusion_matrix, plot_sample_predictions


def evaluate(model_name="cnn", checkpoint_path=None, batch_size=64):
    """
    Evaluates the model on the MNIST test set.
    """
    if checkpoint_path is None:
        checkpoint_path = os.path.join(MODEL_DIR, f"{model_name}_best.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. "
            f"Please run training first using: python main.py --mode train --model {model_name}"
        )

    print(f"\n{'='*60}")
    print(f" Deep Learning Model Evaluation")
    print(f" Model Architecture : {model_name.upper()}")
    print(f" Checkpoint Path    : {checkpoint_path}")
    print(f" Evaluation Device  : {DEVICE}")
    print(f"{'='*60}\n")

    # 1. Load Data
    _, _, test_loader = get_dataloaders(batch_size=batch_size)

    # 2. Load Model
    model = build_model(model_name).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    sample_images = []

    print("[1/3] Running inference across 10,000 test images...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # Softmax to get probabilities
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)

            running_loss += loss.item() * images.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(confidences.cpu().numpy())

            if len(sample_images) < 16:
                for img in images.cpu().numpy():
                    if len(sample_images) < 16:
                        sample_images.append(img)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    total_test = len(all_targets)
    test_loss = running_loss / total_test
    test_acc = (np.sum(all_preds == all_targets) / total_test) * 100.0

    print(f"\n[2/3] Evaluation Metrics:")
    print(f"      Test Loss     : {test_loss:.4f}")
    print(f"      Test Accuracy : {test_acc:.2f}% ({np.sum(all_preds == all_targets)}/{total_test})")

    # Detailed Scikit-Learn Classification Report
    class_names = [f"Digit {i}" for i in range(10)]
    report = classification_report(all_targets, all_preds, target_names=class_names, digits=4)
    print("\nDetailed Per-Class Performance:")
    print(report)

    # 3. Generate Visualizations
    print("[3/3] Generating Confusion Matrix & Prediction Visualizations...")
    cm = confusion_matrix(all_targets, all_preds)
    plot_confusion_matrix(cm, class_names=[str(i) for i in range(10)])

    plot_sample_predictions(
        sample_images[:16],
        all_targets[:16],
        all_preds[:16],
        probs=all_probs[:16],
        num_samples=16
    )

    print("\nEvaluation successfully finished! Visualizations stored in results/")
    return {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "confusion_matrix": cm.tolist()
    }

