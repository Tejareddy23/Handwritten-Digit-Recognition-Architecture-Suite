import os
import torch

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Ensure required directories exist
for directory in [DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Hardware setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
RANDOM_SEED = 42
BATCH_SIZE = 64
DEFAULT_EPOCHS = 5
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# Model architectures available
AVAILABLE_MODELS = ["cnn", "mlp"]
DEFAULT_MODEL = "cnn"

