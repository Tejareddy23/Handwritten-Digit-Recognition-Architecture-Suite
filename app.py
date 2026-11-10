"""
Interactive Streamlit Dashboard for Deep Learning Computer Vision & NLP Suite.
Run with:
    python -m streamlit run app.py
"""

import os
import streamlit as st
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from src.config import MODEL_DIR, RESULTS_DIR, DEVICE
from src.models import build_model, count_parameters
from src.predict import DigitPredictor
from src.dataset import get_dataloaders
from src.explainability import extract_feature_maps, fgsm_attack
from src.nlp_predict import TextIdentifier
from src.image_text_identifier import ImageTextIdentifier


st.set_page_config(
    page_title="Deep Learning AI Suite (Vision & NLP)",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Deep Learning Algorithms & Architectures Suite")
st.markdown(
    """
    An interactive platform exploring foundational **Deep Learning Algorithms and Architectures**
    implemented in PyTorch across **Computer Vision (CNN, MLP)** and **Natural Language Processing (BiLSTM)**.
    """
)

# Sidebar
st.sidebar.header("⚙️ Vision Model Configuration")
model_choice = st.sidebar.selectbox("Select Vision Architecture", ["CNN", "MLP"], index=0)
model_name = model_choice.lower()

model_obj = build_model(model_name)
total_params, trainable_params = count_parameters(model_obj)

st.sidebar.markdown(f"**Architecture:** `{model_choice}`")
st.sidebar.markdown(f"**Parameters:** `{total_params:,}`")
st.sidebar.markdown(f"**Hardware Device:** `{'GPU' if torch.cuda.is_available() else 'CPU'}`")

checkpoint_path = os.path.join(MODEL_DIR, f"{model_name}_best.pth")
has_checkpoint = os.path.exists(checkpoint_path)

if has_checkpoint:
    st.sidebar.success(f"✓ Trained checkpoint ready ({model_choice})!")
else:
    st.sidebar.warning(f"⚠️ No checkpoint found for {model_choice}. Run training first.")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🚀 Single Digit Inference",
    "📝 NLP Text Identifier (BiLSTM)",
    "🔍 Image Text / Multi-Digit OCR",
    "🔬 Feature Maps & FGSM Robustness",
    "📊 Benchmarks & Training Curves",
    "📚 Deep Learning Math & Algorithms"
])

# Global state for current image tensor
if "current_tensor" not in st.session_state:
    st.session_state["current_tensor"] = None
    st.session_state["current_label"] = None

# ==================== TAB 1: VISION INFERENCE ====================
with tab1:
    st.subheader("Single Digit Classification")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.write("### 1. Select or Upload an Image")
        source_mode = st.radio("Input Source", ["Test Dataset Sample", "Upload Custom Image"])

        image_to_predict = None

        if source_mode == "Test Dataset Sample":
            if st.button("🎲 Pick New Random Test Sample"):
                st.session_state["sample_seed"] = np.random.randint(0, 1000)

            sample_seed = st.session_state.get("sample_seed", 42)
            _, _, test_loader = get_dataloaders(batch_size=1)
            
            for i, (img_t, lbl_t) in enumerate(test_loader):
                if i == (sample_seed % len(test_loader)):
                    st.session_state["current_tensor"] = img_t
                    st.session_state["current_label"] = lbl_t
                    img_np = img_t.squeeze().numpy()
                    img_np = (img_np * 0.3081 + 0.1307) * 255.0
                    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                    image_to_predict = Image.fromarray(img_np, mode="L")
                    true_label = lbl_t.item()
                    break

            if image_to_predict is not None:
                st.image(image_to_predict, caption=f"Ground Truth Label: {true_label}", width=180)
        else:
            uploaded_file = st.file_uploader("Upload a digit image (PNG / JPG)", type=["png", "jpg", "jpeg"], key="single_digit")
            if uploaded_file is not None:
                image_to_predict = Image.open(uploaded_file)
                st.image(image_to_predict, caption="Uploaded Image", width=180)

    with col2:
        st.write("### 2. Deep Learning Output")
        if image_to_predict is not None:
            if not has_checkpoint:
                st.error(f"Please train the {model_choice} model first before running inference!")
            else:
                try:
                    predictor = DigitPredictor(model_name=model_name)
                    result = predictor.predict(image_to_predict)

                    pred = result["predicted_digit"]
                    conf = result["confidence"] * 100

                    st.metric(label="Predicted Digit", value=str(pred), delta=f"{conf:.2f}% confidence")

                    st.write("**Softmax Output Probabilities:**")
                    prob_dict = result["probabilities"]
                    st.bar_chart(prob_dict)
                except Exception as e:
                    st.error(f"Inference error: {e}")
        else:
            st.info("Pick or upload an image to see neural network predictions.")

# ==================== TAB 2: NLP TEXT IDENTIFIER ====================
with tab2:
    st.subheader("NLP Deep Learning Text Identifier (Embedding + BiLSTM)")
    st.write(
        "Type any sentence or paragraph below. The model unrolls recurrent memory cells across the word sequence "
        "and identifies the domain category (**Technology**, **Finance**, **Sports**, **Support**)."
    )

    sample_prompts = [
        "The deep learning neural network model achieved state of the art accuracy",
        "Stock market corporate earnings report showed high revenue and dividend growth",
        "The football striker scored a championship winning goal in overtime",
        "I need a refund for my order package that was damaged during delivery"
    ]

    selected_prompt = st.selectbox("Select a quick template or write your own below:", ["(Custom Text)"] + sample_prompts)

    default_text = "" if selected_prompt == "(Custom Text)" else selected_prompt
    user_text = st.text_area("Input Text to Identify:", value=default_text, height=100, placeholder="e.g. Deploying containerized python microservices to the cloud...")

    if st.button("🔍 Identify Text Category"):
        if not user_text.strip():
            st.warning("Please enter some text to classify.")
        else:
            try:
                text_id = TextIdentifier()
                res = text_id.identify(user_text)

                t_col1, t_col2 = st.columns([1, 1])

                with t_col1:
                    st.markdown("### Identification Result")
                    st.metric(
                        label="Identified Category",
                        value=res["identified_category"],
                        delta=f"{res['confidence']*100:.1f}% confidence"
                    )

                    st.write("**Tokenization Breakdown:**")
                    st.write(f"Tokens identified: `{res['tokens']}`")
                    st.write(f"Vocabulary matches: `{res['tokens_in_vocab']}`")

                with t_col2:
                    st.markdown("### Softmax Probability Distribution")
                    st.bar_chart(res["probabilities"])
            except Exception as err:
                st.error(f"Text Identification error: {err}")

# ==================== TAB 3: IMAGE-TO-TEXT / MULTI-DIGIT OCR ====================
with tab3:
    st.subheader("Image-to-Text & Sequence Identifier (OCR)")
    st.write(
        "Upload an image containing handwritten sequences of text or numbers (e.g. '2 0 2 6' or '4 8 1'). "
        "OpenCV performs spatial contour segmentation, extracts each character box, and the deep neural network transcribes the sequence."
    )

    ocr_file = st.file_uploader("Upload an image containing digits/characters", type=["png", "jpg", "jpeg"], key="ocr_uploader")

    if ocr_file is not None:
        pil_ocr = Image.open(ocr_file)
        ocr_tool = ImageTextIdentifier(model_name="cnn")
        res_ocr = ocr_tool.identify_from_image(pil_ocr)

        ocr_col1, ocr_col2 = st.columns(2)
        with ocr_col1:
            st.image(pil_ocr, caption="Original Input Image", use_container_width=True)
        with ocr_col2:
            st.image(res_ocr["annotated_image"], caption=f"Identified Text: {res_ocr['identified_text']}", use_container_width=True)

        st.success(f"**Identified Sequence:** `{res_ocr['identified_text']}` (Mean Confidence: `{res_ocr['average_confidence']*100:.1f}%`)")

# ==================== TAB 4: EXPLAINABILITY & ROBUSTNESS ====================
with tab4:
    st.subheader("Deep Learning Interpretability & Robustness")
    
    if model_choice != "CNN":
        st.info("Feature Map extraction requires the Convolutional Neural Network (CNN). Switch to 'CNN' in the sidebar.")
    elif st.session_state["current_tensor"] is None or not has_checkpoint:
        st.info("Pick a sample in the 'Single Digit Inference' tab first to visualize layer activations.")
    else:
        st.markdown("#### 1. Convolutional Feature Maps (What Neurons See)")
        st.write(
            "Visualizing the activations across channels illustrates how early convolutional layers detect "
            "simple edges and strokes, while deeper layers capture compound geometric shapes."
        )

        cnn_model = build_model("cnn").to(DEVICE)
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        cnn_model.load_state_dict(checkpoint["model_state_dict"])

        fmaps = extract_feature_maps(cnn_model, st.session_state["current_tensor"].to(DEVICE))

        layer_choice = st.radio("Choose Feature Layer", ["Conv Layer 1 (16 feature maps)", "Conv Layer 2 (32 feature maps)"], horizontal=True)
        key = "conv1" if "1" in layer_choice else "conv2"

        maps = fmaps[key]
        cols_count = 8
        rows_count = (maps.shape[0] + cols_count - 1) // cols_count
        fig, axes = plt.subplots(rows_count, cols_count, figsize=(12, 1.8 * rows_count))
        axes = axes.flatten()

        for idx in range(maps.shape[0]):
            axes[idx].imshow(maps[idx], cmap="viridis")
            axes[idx].set_title(f"Channel {idx+1}", fontsize=8)
            axes[idx].axis("off")
        for j in range(maps.shape[0], len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        st.markdown("#### 2. Fast Gradient Sign Method (FGSM Adversarial Attack)")
        st.write(
            "FGSM demonstrates how calculating the gradient of the loss with respect to the **input pixels** "
            "allows crafting imperceptible adversarial perturbations that manipulate network predictions."
        )

        epsilon = st.slider("Perturbation Intensity (Epsilon $\\epsilon$)", min_value=0.0, max_value=0.5, value=0.15, step=0.05)

        if st.session_state["current_label"] is not None:
            perturbed_t, adv_pred, adv_conf = fgsm_attack(
                cnn_model,
                st.session_state["current_tensor"].to(DEVICE),
                st.session_state["current_label"].to(DEVICE),
                epsilon=epsilon
            )

            a_col1, a_col2, a_col3 = st.columns(3)

            with a_col1:
                orig_np = st.session_state["current_tensor"].squeeze().numpy() * 0.3081 + 0.1307
                st.image(np.clip(orig_np, 0, 1), caption=f"Original (True: {st.session_state['current_label'].item()})", width=180)

            with a_col2:
                noise_np = (perturbed_t.detach().cpu().squeeze().numpy() - st.session_state["current_tensor"].squeeze().numpy())
                st.image(np.clip(noise_np * 5 + 0.5, 0, 1), caption="Perturbation Noise (scaled)", width=180)

            with a_col3:
                adv_np = perturbed_t.detach().cpu().squeeze().numpy() * 0.3081 + 0.1307
                st.image(
                    np.clip(adv_np, 0, 1),
                    caption=f"Adversarial Image\nPredicted: {adv_pred} ({adv_conf*100:.1f}%)",
                    width=180
                )

# ==================== TAB 5: BENCHMARKS & METRICS ====================
with tab5:
    st.subheader("Model Performance & Diagnostics")
    
    comp_img = os.path.join(RESULTS_DIR, "model_comparison.png")
    if os.path.exists(comp_img):
        st.markdown("#### Architecture Comparison: MLP vs. CNN")
        st.image(comp_img, use_container_width=True)

    m_col1, m_col2 = st.columns(2)

    with m_col1:
        st.markdown("#### Training & Validation Learning Curves")
        history_img = os.path.join(RESULTS_DIR, "training_history.png")
        if os.path.exists(history_img):
            st.image(history_img, use_container_width=True)
        else:
            st.info("Train the model to view Loss vs. Epochs and Accuracy curves.")

    with m_col2:
        st.markdown("#### Multi-Class Confusion Matrix")
        cm_img = os.path.join(RESULTS_DIR, "confusion_matrix.png")
        if os.path.exists(cm_img):
            st.image(cm_img, use_container_width=True)
        else:
            st.info("Run evaluation to view the confusion matrix heatmap.")

    st.markdown("#### Sample Predictions Grid")
    samples_img = os.path.join(RESULTS_DIR, "sample_predictions.png")
    if os.path.exists(samples_img):
        st.image(samples_img, use_container_width=True)

# ==================== TAB 6: ALGORITHMS EXPLAINED ====================
with tab6:
    st.subheader("Deep Learning Fundamentals Implemented in this Project")
    st.markdown(
        """
        ### 1. Forward Propagation & Non-Linearity
        Each layer computes a linear transformation followed by a non-linear activation:
        $$\\mathbf{z}^{[l]} = \\mathbf{W}^{[l]} \\mathbf{a}^{[l-1]} + \\mathbf{b}^{[l]}$$
        $$\\mathbf{a}^{[l]} = \\text{ReLU}(\\mathbf{z}^{[l]}) = \\max(0, \\mathbf{z}^{[l]})$$
        
        ### 2. Loss Function (Categorical Cross-Entropy)
        Measures the divergence between predicted probability distribution $\\hat{\\mathbf{y}}$ and one-hot target $\\mathbf{y}$:
        $$\\mathcal{L} = -\\sum_{k=0}^{C-1} y_k \\log(\\hat{y}_k)$$
        where $\\hat{y}_k = \\frac{e^{z_k}}{\\sum_{j} e^{z_j}}$ is the **Softmax function**.

        ### 3. Backpropagation (Reverse-Mode Automatic Differentiation)
        Calculates gradients of the scalar loss with respect to all layer parameters using the Chain Rule:
        $$\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{W}^{[l]}} = \\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{z}^{[l]}} \\cdot (\\mathbf{a}^{[l-1]})^T$$

        ### 4. Recurrent Gating Mechanism (LSTM)
        Recurrent memory cells prevent vanishing gradients over long sequences:
        $$f_t = \\sigma(W_f \\cdot [h_{t-1}, x_t] + b_f) \\quad \\text{(Forget Gate)}$$
        $$i_t = \\sigma(W_i \\cdot [h_{t-1}, x_t] + b_i) \\quad \\text{(Input Gate)}$$
        $$\\tilde{C}_t = \\tanh(W_c \\cdot [h_{t-1}, x_t] + b_c) \\quad \\text{(Candidate Memory)}$$
        $$C_t = f_t * C_{t-1} + i_t * \\tilde{C}_t \\quad \\text{(Cell State)}$$
        $$o_t = \\sigma(W_o \\cdot [h_{t-1}, x_t] + b_o) \\quad \\text{(Output Gate)}$$
        $$h_t = o_t * \\tanh(C_t) \\quad \\text{(Hidden State)}$$

        ### 5. Adversarial Robustness (Fast Gradient Sign Method - FGSM)
        Backpropagation can also compute gradients with respect to input pixels:
        $$\\mathbf{x}_{adv} = \\mathbf{x} + \\epsilon \\cdot \\text{sign}\\left(\\nabla_{\\mathbf{x}} \\mathcal{L}(\\mathbf{x}, y)\\right)$$
        """
    )
