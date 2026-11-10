"""
Deep Learning Training Pipeline.

DL Algorithms implemented:
1. Mini-Batch Stochastic Gradient Descent / Adam Optimization:
   - Computes expected gradient over mini-batches of data.
   - Adam uses exponentially decaying averages of past gradients (momentum) and squared gradients (scaling).
2. Forward Propagation:
   - Passes tensor inputs through layers, activations, and normalizations to output prediction logits.
3. Cross-Entropy Loss Computation:
   - Measures divergence between predicted probability distribution and one-hot true ground truth.
4. Backpropagation (Reverse-mode Automatic Differentiation):
   - Calculates gradients dL/dW for all network parameters using the calculus chain rule.
5. Parameter Optimization Step:
   - Updates weights in direction opposite to gradient: W = W - lr * dW (with weight decay L2 penalty).
6. Learning Rate Scheduling:
   - Gradually reduces learning rate over epochs to enable stable convergence to local minima.
7. Model Checkpointing:
   - Saves network weights yielding the highest validation accuracy.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from src.config import DEVICE, MODEL_DIR, DEFAULT_EPOCHS, LEARNING_RATE, WEIGHT_DECAY
from src.models import build_model, count_parameters
from src.dataset import get_dataloaders
from src.utils import set_seed, plot_training_history, save_history_json


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Executes one training epoch across mini-batches.
    """
    model.train()  # Activate dropout and batch normalization training behavior
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training Batch", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # 1. Zero out parameter gradients from previous step
        optimizer.zero_grad()

        # 2. Forward Propagation: Compute model output
        outputs = model(images)

        # 3. Compute Cross-Entropy Loss
        loss = criterion(outputs, labels)

        # 4. Backpropagation: Compute gradients using Chain Rule
        loss.backward()

        # 5. Parameter Update Step (Optimizer)
        optimizer.step()

        # Accumulate metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on validation data without gradient tracking.
    """
    model.eval()  # Freeze dropout and batch normalization for inference
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable autograd to save memory and computations
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Accumulate metrics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / total
    val_acc = (correct / total) * 100.0
    return val_loss, val_acc


def train(
    model_name="cnn",
    epochs=DEFAULT_EPOCHS,
    lr=LEARNING_RATE,
    batch_size=64,
    optimizer_name="adam",
    weight_decay=WEIGHT_DECAY,
    seed=42
):
    """
    Main training function orchestrating dataset loading, model initialization,
    optimization loop, validation, and checkpointing.
    """
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f" Deep Learning Training Pipeline")
    print(f" Model Architecture : {model_name.upper()}")
    print(f" Target Device      : {DEVICE}")
    print(f" Total Epochs       : {epochs}")
    print(f" Batch Size         : {batch_size}")
    print(f" Optimizer          : {optimizer_name.upper()} (lr={lr}, weight_decay={weight_decay})")
    print(f"{'='*60}\n")

    # 1. Load Data
    print("[1/5] Loading and preparing datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)
    print(f"      Train Samples: {len(train_loader.dataset)} | Val Samples: {len(val_loader.dataset)}")

    # 2. Build Model
    print("\n[2/5] Initializing Deep Learning Model...")
    model = build_model(model_name).to(DEVICE)
    total_params, trainable_params = count_parameters(model)
    print(f"      Total Parameters     : {total_params:,}")
    print(f"      Trainable Parameters : {trainable_params:,}")

    # 3. Loss & Optimizer Algorithms
    criterion = nn.CrossEntropyLoss()

    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. Choose 'adam' or 'sgd'.")

    # Learning rate decay: multiply lr by 0.5 every 2 epochs
    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

    # 4. Training Loop
    print("\n[3/5] Starting Deep Learning Training Loop...")
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }

    best_val_acc = 0.0
    best_model_path = os.path.join(MODEL_DIR, f"{model_name}_best.pth")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        # Step LR Scheduler
        scheduler.step()

        epoch_duration = time.time() - epoch_start

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_duration:.1f}s) | "
            f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} - Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        # Checkpoint Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss
            }, best_model_path)
            print(f"      --> Best model saved! (Validation Accuracy: {val_acc:.2f}%)")

    total_time = time.time() - start_time
    print(f"\n[4/5] Training Completed in {total_time:.2f} seconds!")
    print(f"      Best Validation Accuracy: {best_val_acc:.2f}%")

    # 5. Save History & Visualizations
    print("\n[5/5] Generating training curves and metrics artifact...")
    plot_training_history(history)
    save_history_json(history)

    return model, history

