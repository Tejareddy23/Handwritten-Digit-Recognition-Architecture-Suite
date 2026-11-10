"""
NLP Text Inference & Prediction Module.

Classifies arbitrary natural language text into identified categories
using the trained Bidirectional LSTM Deep Learning model.
"""

import os
import torch
import numpy as np

from src.config import DEVICE, MODEL_DIR
from src.nlp_models import BiLSTMTextClassifier
from src.nlp_dataset import Vocabulary, CATEGORIES


class TextIdentifier:
    """
    High-level predictor for natural language text identification.
    """

    def __init__(self, checkpoint_path=None, vocab_path=None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(MODEL_DIR, "text_bilstm_best.pth")
        if vocab_path is None:
            vocab_path = os.path.join(MODEL_DIR, "vocab.json")

        if not os.path.exists(checkpoint_path) or not os.path.exists(vocab_path):
            raise FileNotFoundError(
                f"Trained text model or vocabulary not found.\n"
                f"Please train it first using: python main.py --mode text-train"
            )

        self.device = DEVICE
        self.vocab = Vocabulary.load(vocab_path)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = BiLSTMTextClassifier(
            vocab_size=checkpoint["vocab_size"],
            embed_dim=checkpoint.get("embed_dim", 64),
            hidden_dim=checkpoint.get("hidden_dim", 64),
            num_classes=checkpoint.get("num_classes", len(CATEGORIES))
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def identify(self, text, max_len=25):
        """
        Processes text string and returns identified category, confidence, and distribution.
        """
        indices = self.vocab.text_to_indices(text, max_len=max_len)
        tensor = torch.tensor([indices], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        predicted_idx = int(np.argmax(probs))
        predicted_category = CATEGORIES[predicted_idx]
        confidence = float(probs[predicted_idx])

        tokens = self.vocab.tokenize(text)

        return {
            "text": text,
            "tokens": tokens,
            "tokens_in_vocab": [t for t in tokens if t in self.vocab.word2idx],
            "identified_category": predicted_category,
            "confidence": confidence,
            "probabilities": {CATEGORIES[i]: float(probs[i]) for i in range(len(CATEGORIES))}
        }

