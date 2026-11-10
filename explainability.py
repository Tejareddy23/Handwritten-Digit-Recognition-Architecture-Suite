"""
Deep Learning Explainability & Robustness Module.

Implements two advanced Deep Learning algorithms:
1. Feature Map Extraction:
   - Visualizes intermediate activation tensors after 2D convolution and pooling.
   - Reveals hierarchical feature learning (edges -> corners -> structural loops).
2. Fast Gradient Sign Method (FGSM Adversarial Attack):
   - Computes gradient of loss with respect to input pixels: dL / dx
   - Perturbs image: x_adv = x + epsilon * sign(dL / dx)
   - Demonstrates neural network decision boundaries and vulnerability.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from src.config import RESULTS_DIR


def extract_feature_maps(model, image_tensor):
    """
    Passes an image tensor through the CNN and extracts feature maps from Conv1 and Conv2.
    """
    model.eval()
    feature_maps = {}

    with torch.no_grad():
        # Conv 1 + BN + ReLU + Pool
        x1 = model.conv1(image_tensor)
        x1_act = F.relu(model.bn1(x1))
        x1_pool = model.pool1(x1_act)
        feature_maps["conv1"] = x1_act.squeeze(0).cpu().numpy()
        feature_maps["pool1"] = x1_pool.squeeze(0).cpu().numpy()

        # Conv 2 + BN + ReLU + Pool
        x2 = model.conv2(x1_pool)
        x2_act = F.relu(model.bn2(x2))
        x2_pool = model.pool2(x2_act)
        feature_maps["conv2"] = x2_act.squeeze(0).cpu().numpy()
        feature_maps["pool2"] = x2_pool.squeeze(0).cpu().numpy()

    return feature_maps


def plot_feature_maps(feature_maps, layer_name="conv1", save_path=None):
    """
    Plots a grid of channel activations for a given layer.
    """
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, f"feature_maps_{layer_name}.png")

    maps = feature_maps[layer_name]
    num_channels = maps.shape[0]

    cols = 8
    rows = (num_channels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for idx in range(num_channels):
        ax = axes[idx]
        ax.imshow(maps[idx], cmap="viridis")
        ax.set_title(f"Ch {idx+1}", fontsize=9)
        ax.axis("off")

    for j in range(num_channels, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Activation Maps for Layer: {layer_name.upper()}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def fgsm_attack(model, image_tensor, target_label, epsilon=0.2):
    """
    Generates an adversarial example using the Fast Gradient Sign Method (FGSM).

    Algorithm:
    1. Set requires_grad=True on input tensor.
    2. Forward pass: compute output logits and cross-entropy loss.
    3. Backward pass: compute dL / dx (gradient of loss w.r.t input image).
    4. Apply perturbation: x_adv = clip(x + epsilon * sign(dL / dx), min_val, max_val).
    """
    model.eval()
    image_copy = image_tensor.clone().detach()
    image_copy.requires_grad = True

    # Forward pass
    output = model(image_copy)
    loss = F.cross_entropy(output, target_label)

    # Zero existing gradients and backward pass
    model.zero_grad()
    loss.backward()

    # Collect gradient sign
    data_grad = image_copy.grad.data
    sign_data_grad = data_grad.sign()

    # Create adversarial perturbation
    perturbed_image = image_copy + epsilon * sign_data_grad

    # De-normalize / Clamp to valid normalized range
    perturbed_image = torch.clamp(perturbed_image, -0.4242, 2.8215)

    # Predict on perturbed image
    with torch.no_grad():
        adv_output = model(perturbed_image)
        adv_probs = torch.softmax(adv_output, dim=1).squeeze()
        adv_pred = int(torch.argmax(adv_probs).item())
        adv_conf = float(adv_probs[adv_pred].item())

    return perturbed_image, adv_pred, adv_conf

