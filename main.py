"""
Deep Learning Project CLI Runner.

Usage examples:
  python main.py --mode train --model cnn --epochs 3
  python main.py --mode evaluate --model cnn
  python main.py --mode predict --model cnn
  python main.py --mode compare
"""

import argparse
import sys
import os

from src.train import train
from src.evaluate import evaluate
from src.predict import DigitPredictor
from src.dataset import get_dataloaders
from src.models import build_model, count_parameters
from src.utils import plot_model_comparison
from src.config import DEFAULT_EPOCHS, LEARNING_RATE, BATCH_SIZE, MODEL_DIR


def run_sample_prediction(model_name="cnn"):
    """
    Picks a random sample from the test set, saves it as an image, and predicts its class.
    """
    _, _, test_loader = get_dataloaders(batch_size=1)
    for img, label in test_loader:
        sample_tensor = img
        true_label = label.item()
        break

    # Convert back to PIL Image
    img_np = sample_tensor.squeeze().numpy()
    img_np = (img_np * 0.3081 + 0.1307) * 255.0
    img_np = img_np.clip(0, 255).astype("uint8")
    from PIL import Image
    pil_img = Image.fromarray(img_np, mode="L")

    sample_path = "sample_test_image.png"
    pil_img.save(sample_path)
    print(f"\n[Sample Test] Saved a random test image to '{sample_path}' (Ground Truth Label: {true_label})")

    predictor = DigitPredictor(model_name=model_name)
    result = predictor.predict(sample_path)

    print(f"Prediction Result:")
    print(f"  --> Predicted Digit : {result['predicted_digit']}")
    print(f"  --> Confidence      : {result['confidence']*100:.2f}%")
    print(f"  --> Match Correct   : {'YES (✓)' if result['predicted_digit'] == true_label else 'NO (✗)'}")
    print("\nClass Probabilities:")
    for digit, prob in result["probabilities"].items():
        bar = "█" * int(prob * 30)
        print(f"   Digit {digit}: {prob*100:5.1f}% {bar}")


def run_comparison(retrain=False, epochs=3):
    """
    Compares MLP and CNN models on Test Accuracy, Test Loss, and Parameter Efficiency.
    """
    print("\n" + "="*70)
    print(" Architecture Comparison: Multi-Layer Perceptron (MLP) vs. CNN")
    print("="*70)

    mlp_ckpt = os.path.join(MODEL_DIR, "mlp_best.pth")
    cnn_ckpt = os.path.join(MODEL_DIR, "cnn_best.pth")

    # Train or load MLP
    if retrain or not os.path.exists(mlp_ckpt):
        print("\n>>> Phase 1: Training Multi-Layer Perceptron (MLP)...")
        train(model_name="mlp", epochs=epochs)
    else:
        print("\n>>> Phase 1: Found existing MLP checkpoint. Evaluating directly...")
    mlp_metrics = evaluate(model_name="mlp")

    # Train or load CNN
    if retrain or not os.path.exists(cnn_ckpt):
        print("\n>>> Phase 2: Training Convolutional Neural Network (CNN)...")
        train(model_name="cnn", epochs=epochs)
    else:
        print("\n>>> Phase 2: Found existing CNN checkpoint. Evaluating directly...")
    cnn_metrics = evaluate(model_name="cnn")

    mlp_params = count_parameters(build_model("mlp"))[0]
    cnn_params = count_parameters(build_model("cnn"))[0]

    comparison_data = {
        "mlp": {
            "test_acc": mlp_metrics["test_acc"],
            "test_loss": mlp_metrics["test_loss"],
            "params": mlp_params
        },
        "cnn": {
            "test_acc": cnn_metrics["test_acc"],
            "test_loss": cnn_metrics["test_loss"],
            "params": cnn_params
        }
    }

    # Plot comparison bar charts
    plot_model_comparison(comparison_data)

    print("\n" + "="*70)
    print(" ARCHITECTURE COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Metric':<25} | {'MLP (Dense)':<20} | {'CNN (Convolutional)':<20}")
    print("-" * 70)
    print(f"{'Total Parameters':<25} | {mlp_params:<20,d} | {cnn_params:<20,d}")
    print(f"{'Test Accuracy':<25} | {mlp_metrics['test_acc']:<19.2f}% | {cnn_metrics['test_acc']:<19.2f}%")
    print(f"{'Test Loss':<25} | {mlp_metrics['test_loss']:<20.4f} | {cnn_metrics['test_loss']:<20.4f}")
    print("="*70)
    print("Comparison chart saved to: results/model_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Deep Learning Handwritten Digit Recognition Project")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "evaluate", "predict", "compare", "text-train", "text-predict", "ocr-predict"],
        help="Pipeline mode to execute"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["cnn", "mlp"],
        help="Deep Learning architecture to use ('cnn' or 'mlp')"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate for optimizer (default: {LEARNING_RATE})"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"Mini-batch size (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="Optimization algorithm to use: 'adam' or 'sgd'"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Custom image path for prediction (mode='predict' or 'ocr-predict')"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="The neural network uses backpropagation to optimize loss",
        help="Input sentence for text identification (mode='text-predict')"
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force re-training of models even if checkpoints already exist (mode='compare')"
    )

    args = parser.parse_args()

    if args.mode == "train":
        train(
            model_name=args.model,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            optimizer_name=args.optimizer
        )
    elif args.mode == "evaluate":
        evaluate(model_name=args.model, batch_size=args.batch_size)
    elif args.mode == "predict":
        if args.image_path:
            predictor = DigitPredictor(model_name=args.model)
            res = predictor.predict(args.image_path)
            print(f"Prediction for '{args.image_path}':")
            print(f"Predicted Digit: {res['predicted_digit']} ({res['confidence']*100:.2f}% confidence)")
        else:
            run_sample_prediction(model_name=args.model)
    elif args.mode == "compare":
        run_comparison(retrain=args.retrain, epochs=args.epochs)
    elif args.mode == "text-train":
        from src.nlp_train import train_text_model
        train_text_model(epochs=args.epochs)
    elif args.mode == "text-predict":
        from src.nlp_predict import TextIdentifier
        identifier = TextIdentifier()
        res = identifier.identify(args.text)
        print(f"\nNLP BiLSTM Text Identifier Result:")
        print(f"  --> Input Text         : \"{res['text']}\"")
        print(f"  --> Identified Category: {res['identified_category']}")
        print(f"  --> Model Confidence   : {res['confidence']*100:.2f}%")
        print("\nProbability Distribution:")
        for cat, prob in res["probabilities"].items():
            bar = "█" * int(prob * 30)
            print(f"   {cat:<14}: {prob*100:5.1f}% {bar}")
    elif args.mode == "ocr-predict":
        from src.image_text_identifier import ImageTextIdentifier
        if not args.image_path:
            # Fallback to sample image
            run_sample_prediction(model_name="cnn")
            args.image_path = "sample_test_image.png"
        ocr = ImageTextIdentifier(model_name=args.model)
        res = ocr.identify_from_image(args.image_path)
        print(f"\nImage-to-Text Multi-Character Identification:")
        print(f"  --> Recognized Sequence : {res['identified_text']}")
        print(f"  --> Mean Confidence     : {res['average_confidence']*100:.2f}%")


if __name__ == "__main__":
    main()

