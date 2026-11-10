"""
Deep Learning Training Pipeline for Text Identification (BiLSTM).

DL Algorithms & Concepts:
1. Backpropagation Through Time (BPTT): Unrolls recurrent cells over sequence steps to compute dL/dW.
2. Gradient Clipping (norm clipping): Prevents exploding gradients in recurrent connections.
3. Adam Optimizer with L2 Weight Decay: Adaptive learning rates per parameter.
4. Model & Vocabulary Checkpointing.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.config import DEVICE, MODEL_DIR
from src.nlp_models import BiLSTMTextClassifier
from src.nlp_dataset import get_nlp_dataloaders, CATEGORIES


def train_text_model(epochs=12, lr=0.003, batch_size=16, embed_dim=64, hidden_dim=64):
    """
    Trains the BiLSTM text identification model.
    """
    print(f"\n{'='*65}")
    print(" NLP Deep Learning Pipeline: BiLSTM Text Identifier")
    print(f" Target Device: {DEVICE}")
    print(f" Categories   : {', '.join(CATEGORIES)}")
    print(f" Total Epochs : {epochs} | Batch Size: {batch_size}")
    print(f"{'='*65}\n")

    # 1. Prepare Data & Vocabulary
    print("[1/4] Tokenizing and building vocabulary...")
    train_loader, test_loader, vocab = get_nlp_dataloaders(batch_size=batch_size)
    print(f"      Vocabulary Size: {len(vocab)} unique tokens")
    print(f"      Train Samples  : {len(train_loader.dataset)} | Test Samples: {len(test_loader.dataset)}")

    # Save vocabulary
    vocab_path = os.path.join(MODEL_DIR, "vocab.json")
    vocab.save(vocab_path)
    print(f"      Saved vocabulary to: {vocab_path}")

    # 2. Build Model
    print("\n[2/4] Initializing BiLSTM Deep Learning Model...")
    model = BiLSTMTextClassifier(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_classes=len(CATEGORIES)
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # 3. Training Loop
    print("\n[3/4] Running Backpropagation Through Time (BPTT) Training Loop...")
    best_acc = 0.0
    best_path = os.path.join(MODEL_DIR, "text_bilstm_best.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for texts, labels in train_loader:
            texts = texts.to(DEVICE)
            labels = labels.to(DEVICE)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass through embedding and bidirectional recurrent cells
            outputs = model(texts)
            loss = criterion(outputs, labels)

            # Backpropagation Through Time (BPTT)
            loss.backward()

            # Gradient Clipping: Essential DL algorithm for recurrent stability
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            # Optimizer parameter update
            optimizer.step()

            running_loss += loss.item() * texts.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = (correct / total) * 100.0

        # Evaluation on test split
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0.0

        with torch.no_grad():
            for texts, labels in test_loader:
                texts = texts.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(texts)
                v_loss = criterion(outputs, labels)

                val_loss_sum += v_loss.item() * texts.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = (val_correct / val_total) * 100.0
        val_loss = val_loss_sum / val_total

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.1f}% | "
            f"Test Loss: {val_loss:.4f} - Acc: {val_acc:.1f}%"
        )

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "vocab_size": len(vocab),
                "embed_dim": embed_dim,
                "hidden_dim": hidden_dim,
                "num_classes": len(CATEGORIES),
                "test_acc": val_acc
            }, best_path)

    print(f"\n[4/4] Text Identifier Training Completed! Best Test Accuracy: {best_acc:.2f}%")
    print(f"      Model checkpoint saved to: {best_path}")
    return model, vocab

