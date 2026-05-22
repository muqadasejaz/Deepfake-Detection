import streamlit as st
import torch
import timm
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import gdown

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🔍",
    layout="centered",
)

# ── Minimal custom CSS ────────────────────────────────────────
st.markdown("""
<style>
    .result-box {
        padding: 1.5rem 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    .real  { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .fake  { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    .confidence { font-size: 0.95rem; font-weight: 400; margin-top: 0.4rem; }
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────
WEIGHTS_PATH = "xceptionnet_weights.pth"
IMAGE_SIZE   = 299
CLASSES      = ["Real", "Fake"]
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ── Download weights from Google Drive ───────────────────────
def download_weights():
    """Download model weights from Google Drive if not already present."""
    if os.path.exists(WEIGHTS_PATH):
        return  # already downloaded in this session

    file_id = st.secrets["gdrive"]["deepfake_image"]
    url = f"https://drive.google.com/uc?id={file_id}"

    with st.spinner("Downloading model weights… (first run only, may take a minute)"):
        gdown.download(url, WEIGHTS_PATH, quiet=False)

    if not os.path.exists(WEIGHTS_PATH):
        raise RuntimeError(
            "Download failed. Check that the Google Drive file is set to "
            "'Anyone with the link can view' and the file ID in secrets is correct."
        )

# ── Model loading ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    download_weights()

    model = timm.create_model("xception", pretrained=False, num_classes=2)
    state = torch.load(WEIGHTS_PATH, map_location=DEVICE)

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

# ── Preprocessing ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# ── Prediction ────────────────────────────────────────────────
def predict(model, image: Image.Image):
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    label_idx  = int(np.argmax(probs))
    label      = CLASSES[label_idx]
    confidence = float(probs[label_idx]) * 100
    return label, confidence, probs

# ── UI ────────────────────────────────────────────────────────
st.title("🔍 DeepFake Image Detector")
st.write("Upload an image and find out whether it's real or AI-generated.")

try:
    model = load_model()
except Exception as e:
    st.error(f" Failed to load model: {e}")
    st.stop()

uploaded = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, use_column_width=True)

    if st.button("🔍 Analyze Image"):
        with st.spinner("Analyzing…"):
            try:
                label, confidence, probs = predict(model, image)

                css_class = "real" if label == "Real" else "fake"
                icon      = "✅" if label == "Real" else "⚠️"

                st.markdown(f"""
                <div class="result-box {css_class}">
                    {icon} {label}
                    <div class="confidence">
                        Confidence: {confidence:.1f}%
                        &nbsp;|&nbsp;
                        Real: {probs[0]*100:.1f}%&nbsp;&nbsp;Fake: {probs[1]*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f" Prediction failed: {e}")
else:
    st.info("Upload an image above to get started.")
