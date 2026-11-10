# 🧠 Deep Learning Handwritten Digit Recognition & Architecture Suite

A modular, production-ready Deep Learning project built with **PyTorch**, implementing core Deep Learning algorithms from scratch: Forward Propagation, Cross-Entropy Loss, Backpropagation, Optimizer dynamics (Adam / SGD with Momentum), Learning Rate Decay, Regularization (Dropout & Batch Normalization), and an interactive **Streamlit** dashboard.

---

## 📌 Features

- **Deep Learning Architectures**:
  - **Convolutional Neural Network (CNN)**: Feature extraction via 2D spatial convolution kernels, spatial pooling, and dense classification heads.
  - **Multi-Layer Perceptron (MLP)**: Deep fully connected feedforward architecture with batch normalization and dropout.
  - **Bidirectional LSTM (BiLSTM)**: Recurrent neural network with forget, input, and output gating mechanisms for natural language text classification.
  - **Image-to-Text OCR Sequence Identifier**: Combines OpenCV contour character segmentation with CNN inference to transcribe multi-character handwritten sequences.
- **Deep Learning Algorithms Implemented**:
  - **Forward Propagation**: Matrix transformations with $\text{ReLU}$ non-linear activations.
  - **Loss Optimization**: Multi-Class Cross-Entropy Loss with Softmax normalization.
  - **Backpropagation**: Reverse-mode automatic differentiation computing $\nabla_\theta \mathcal{L}$ via the chain rule.
  - **Adaptive Gradient Descent**: Adam optimizer with first and second moment estimations ($m_t, v_t$) alongside classic SGD with momentum.
  - **Regularization**: Dropout (stochastic feature deactivation), Batch Normalization, and $L_2$ Weight Decay.
- **Evaluation & Diagnostics**:
  - Test accuracy, loss, precision, recall, and F1-score via Scikit-Learn.
  - High-resolution Seaborn confusion matrix heatmaps.
  - Training loss and accuracy convergence curves.
- **Interactive Web App**:
  - Streamlit dashboard to test predictions in real-time, view layer activations, and explore confidence distributions.

---

## 📂 Project Structure

```
rithwik project/
│
├── data/                       # MNIST dataset (automatically downloaded)
├── saved_models/               # Checkpoints of best trained model weights (.pth)
├── results/                    # Generated charts (training curves, confusion matrix, sample predictions)
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Hyperparameters (batch size, learning rate, seed, device)
│   ├── dataset.py              # Data transforms, augmentation, PyTorch DataLoader splits
│   ├── models.py               # CNN & MLP PyTorch nn.Module architectures
│   ├── train.py                # Core training loop (forward, loss, backprop, optimizer step)
│   ├── evaluate.py             # Evaluation, classification report & confusion matrix
│   ├── predict.py              # Single/batch inference pipeline with image preprocessing
│   └── utils.py                # Seed setting, plotting curves & heatmap visualization
│
├── main.py                     # CLI entry point (train, evaluate, predict, compare)
├── app.py                      # Interactive Streamlit Web UI
├── requirements.txt            # Python dependencies
└── README.md                   # Complete documentation
```

---

## 🔬 Deep Learning Algorithms Explained

### 1. Forward Propagation
In each feedforward or dense layer, the input $\mathbf{a}^{[l-1]}$ undergoes an affine linear transformation:
$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$
Followed by the non-linear Rectified Linear Unit ($\text{ReLU}$) activation:
$$\mathbf{a}^{[l]} = \max(0, \mathbf{z}^{[l]})$$

In convolutional layers, 2D discrete cross-correlation is applied:
$$S(i, j) = (I * K)(i, j) = \sum_{m} \sum_{n} I(i-m, j-n) K(m, n)$$

### 2. Categorical Cross-Entropy Loss
Given predicted logits $\mathbf{z}$, the network normalizes logits into class probabilities using the **Softmax** function:
$$\hat{y}_k = \frac{e^{z_k}}{\sum_{j=0}^{C-1} e^{z_j}}$$
The objective function minimizes the negative log likelihood of the true label:
$$\mathcal{L}_{CE} = -\sum_{k=0}^{C-1} y_k \log(\hat{y}_k)$$

### 3. Backpropagation (Chain Rule)
During `loss.backward()`, PyTorch computes the gradient of the scalar loss with respect to every weight parameter:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[l]}} \cdot (\mathbf{a}^{[l-1]})^T$$

### 4. Parameter Optimization
- **Adam (Adaptive Moment Estimation)**:
  $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad (\text{momentum})$$
  $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad (\text{RMS scale})$$
  $$\mathbf{W}_{t} = \mathbf{W}_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train a Model
Train the Convolutional Neural Network (CNN):
```bash
python main.py --mode train --model cnn --epochs 5 --batch_size 128
```
Or train the Multi-Layer Perceptron (MLP):
```bash
python main.py --mode train --model mlp --epochs 5 --batch_size 128
```

### 3. Evaluate the Model
Evaluate on the 10,000 unseen test images and generate metrics and confusion matrix:
```bash
python main.py --mode evaluate --model cnn
```

### 4. Run Single Image Inference
Predict a digit on a random test sample or provide your own image:
```bash
python main.py --mode predict --model cnn
```
With a custom image:
```bash
python main.py --mode predict --model cnn --image_path path/to/digit.png
```

### 5. Compare Architectures (MLP vs. CNN)
Compare accuracy, parameter count, and loss between MLP and CNN:
```bash
python main.py --mode compare
```

### 6. Run NLP Deep Learning Text Identifier
Identify categories/domains from arbitrary natural language text:
```bash
python main.py --mode text-predict --text "The deep learning neural network achieved state of the art accuracy"
```
Or train the text BiLSTM model:
```bash
python main.py --mode text-train --epochs 12
```

### 7. Run Multi-Character Image-to-Text OCR
Transcribe multi-digit text sequences directly from an image:
```bash
python main.py --mode ocr-predict --image_path path/to/multidigits.png
```

### 8. Launch the Interactive Web Dashboard
Run the Streamlit application via Python's module runner:
```bash
python -m streamlit run app.py
```

---

## 📈 Expected Results

| Model Architecture | Parameters | Test Accuracy | Test Loss |
| :--- | :--- | :--- | :--- |
| **Multi-Layer Perceptron (MLP)** | ~235,146 | **~98.1%** | ~0.065 |
| **Convolutional Neural Network (CNN)** | ~106,122 | **~99.1%** | ~0.029 |

