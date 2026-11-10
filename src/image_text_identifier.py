"""
Image-to-Text & Multi-Character Sequence Identifier (OCR).

Combines Computer Vision contour segmentation with trained Deep Learning
neural networks to identify sequences of handwritten text/digits from images.
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image

from src.predict import DigitPredictor


class ImageTextIdentifier:
    """
    Identifies sequences of characters/text written in an image.
    Segments bounding boxes using OpenCV, scales them to 28x28,
    and classifies each using the deep learning CNN model.
    """

    def __init__(self, model_name="cnn"):
        self.predictor = DigitPredictor(model_name=model_name)

    def identify_from_image(self, image_input, min_contour_area=50):
        """
        Args:
            image_input: File path, PIL Image, or numpy array.
        Returns:
            dict containing recognized_text, list of character details, annotated image.
        """
        if isinstance(image_input, str):
            cv_img = cv2.imread(image_input)
        elif isinstance(image_input, Image.Image):
            cv_img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            cv_img = image_input.copy()
            if len(cv_img.shape) == 2:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
        else:
            raise TypeError("Unsupported image input type.")

        # Convert to grayscale
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # Thresholding: ensure dark background and bright text (MNIST format)
        if gray.mean() > 127:
            # White background with dark text -> invert
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        else:
            _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        # Morphological dilation to bridge slight gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        # Find external contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter and sort contours from left to right (reading order)
        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > min_contour_area:
                x, y, w, h = cv2.boundingRect(c)
                # Filter out extreme aspect ratio noise
                if h > 8 and w > 4:
                    boxes.append((x, y, w, h))

        # Sort left-to-right by X coordinate
        boxes = sorted(boxes, key=lambda b: b[0])

        annotated_img = cv_img.copy()
        recognized_chars = []

        for (x, y, w, h) in boxes:
            # Add padding
            pad = int(max(w, h) * 0.25)
            x_start = max(0, x - pad)
            y_start = max(0, y - pad)
            x_end = min(cv_img.shape[1], x + w + pad)
            y_end = min(cv_img.shape[0], y + h + pad)

            roi = thresh[y_start:y_end, x_start:x_end]

            # Make square by padding to maintain aspect ratio
            rh, rw = roi.shape
            sq_size = max(rh, rw)
            square_roi = np.zeros((sq_size, sq_size), dtype=np.uint8)
            y_offset = (sq_size - rh) // 2
            x_offset = (sq_size - rw) // 2
            square_roi[y_offset:y_offset + rh, x_offset:x_offset + rw] = roi

            # Resize to standard MNIST 28x28 dimension
            char_img_28 = cv2.resize(square_roi, (28, 28), interpolation=cv2.INTER_AREA)

            # Predict using Deep Learning model
            result = self.predictor.predict(char_img_28)
            pred_char = str(result["predicted_digit"])
            conf = result["confidence"]

            recognized_chars.append({
                "char": pred_char,
                "confidence": conf,
                "box": (x, y, w, h)
            })

            # Draw bounding box and identified label on annotated image
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                annotated_img,
                f"{pred_char} ({conf*100:.0f}%)",
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        full_text = "".join([c["char"] for c in recognized_chars])
        avg_conf = np.mean([c["confidence"] for c in recognized_chars]) if recognized_chars else 0.0

        # Convert annotated BGR image back to RGB for display
        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        return {
            "identified_text": full_text if full_text else "(No characters detected)",
            "average_confidence": float(avg_conf),
            "characters": recognized_chars,
            "annotated_image": annotated_rgb
        }

