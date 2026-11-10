"""
Inference & Prediction Pipeline.

Accepts an image (file path, PIL Image, or numpy array),
applies preprocessing, passes it through the trained Deep Learning model,
and calculates the Softmax confidence distribution.
"""

import os
import torch
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np

from src.config import DEVICE, MODEL_DIR
from src.models import build_model


class DigitPredictor:
    """
    High-level predictor class for single or batch digit inference.
    """

    def __init__(self, model_name="cnn", checkpoint_path=None):
        self.model_name = model_name
        if checkpoint_path is None:
            checkpoint_path = os.path.join(MODEL_DIR, f"{model_name}_best.pth")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Model checkpoint not found: '{checkpoint_path}'. "
                f"Train the model first using: python main.py --mode train --model {model_name}"
            )

        self.device = DEVICE
        self.model = build_model(model_name).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Preprocessing transform: Resize to 28x28, Grayscale, ToTensor, Normalize
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def preprocess_image(self, image_input):
        """
        Accepts file path, numpy array, or PIL Image.
        Converts to 28x28 grayscale, inverts colors if background is bright,
        and applies normalization.
        """
        if isinstance(image_input, str):
            image = Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input.astype(np.uint8))
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise TypeError("Unsupported image input type. Provide path, PIL Image, or numpy array.")

        # Convert to Grayscale
        image = image.convert("L")

        # If image has white background and dark strokes, invert so it matches MNIST (dark background, bright stroke)
        img_np = np.array(image)
        if img_np.mean() > 127:
            image = ImageOps.invert(image)

        tensor = self.transform(image).unsqueeze(0)  # Shape [1, 1, 28, 28]
        return tensor

    def predict(self, image_input):
        """
        Executes forward pass and returns predicted digit, confidence, and class probabilities.
        """
        tensor = self.preprocess_image(image_input).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])

        return {
            "predicted_digit": predicted_class,
            "confidence": confidence,
            "probabilities": {str(i): float(probs[i]) for i in range(10)}
        }

