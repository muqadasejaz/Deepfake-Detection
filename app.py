# import streamlit as st
# import torch
# import timm
# import torchvision.transforms as transforms
# from PIL import Image
# import numpy as np
# import os
# import gdown

# # ── Page config ──────────────────────────────────────────────
# st.set_page_config(
#     page_title="DeepFake Detector",
#     page_icon="🔍",
#     layout="centered",
# )

# # ── Minimal custom CSS ────────────────────────────────────────
# st.markdown("""
# <style>
#     .result-box {
#         padding: 1.5rem 2rem;
#         border-radius: 12px;
#         text-align: center;
#         font-size: 1.4rem;
#         font-weight: 600;
#         margin-top: 1.5rem;
#     }
#     .real  { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
#     .fake  { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
#     .confidence { font-size: 0.95rem; font-weight: 400; margin-top: 0.4rem; }
# </style>
# """, unsafe_allow_html=True)

# # ── Config ────────────────────────────────────────────────────
# WEIGHTS_PATH = "xceptionnet_weights.pth"
# IMAGE_SIZE   = 299
# CLASSES      = ["Real", "Fake"]
# DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MEAN = [0.485, 0.456, 0.406]
# STD  = [0.229, 0.224, 0.225]

# # ── Download weights from Google Drive ───────────────────────
# def download_weights():
#     """Download model weights from Google Drive if not already present."""
#     if os.path.exists(WEIGHTS_PATH):
#         return  # already downloaded in this session

#     file_id = st.secrets["gdrive"]["deepfake_image"]
#     url = f"https://drive.google.com/uc?id={file_id}"

#     with st.spinner("Downloading model weights… (first run only, may take a minute)"):
#         gdown.download(url, WEIGHTS_PATH, quiet=False)

#     if not os.path.exists(WEIGHTS_PATH):
#         raise RuntimeError(
#             "Download failed. Check that the Google Drive file is set to "
#             "'Anyone with the link can view' and the file ID in secrets is correct."
#         )

# # ── Model loading ─────────────────────────────────────────────
# @st.cache_resource(show_spinner="Loading model…")
# def load_model():
#     download_weights()

#     model = timm.create_model("xception", pretrained=False, num_classes=2)
#     state = torch.load(WEIGHTS_PATH, map_location=DEVICE)

#     if isinstance(state, dict) and "model_state_dict" in state:
#         state = state["model_state_dict"]
#     elif isinstance(state, dict) and "state_dict" in state:
#         state = state["state_dict"]

#     model.load_state_dict(state)
#     model.to(DEVICE)
#     model.eval()
#     return model

# # ── Preprocessing ─────────────────────────────────────────────
# transform = transforms.Compose([
#     transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=MEAN, std=STD),
# ])

# # ── Prediction ────────────────────────────────────────────────
# def predict(model, image: Image.Image):
#     tensor = transform(image).unsqueeze(0).to(DEVICE)
#     with torch.no_grad():
#         logits = model(tensor)
#         probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
#     label_idx  = int(np.argmax(probs))
#     label      = CLASSES[label_idx]
#     confidence = float(probs[label_idx]) * 100
#     return label, confidence, probs

# # ── UI ────────────────────────────────────────────────────────
# st.title("🔍 DeepFake Image Detector")
# st.write("Upload an image and find out whether it's real or AI-generated.")

# try:
#     model = load_model()
# except Exception as e:
#     st.error(f" Failed to load model: {e}")
#     st.stop()

# uploaded = st.file_uploader(
#     "Choose an image",
#     type=["jpg", "jpeg", "png", "webp"],
#     label_visibility="collapsed",
# )

# if uploaded:
#     image = Image.open(uploaded).convert("RGB")
#     st.image(image, use_column_width=True)

#     if st.button("🔍 Analyze Image"):
#         with st.spinner("Analyzing…"):
#             try:
#                 label, confidence, probs = predict(model, image)

#                 css_class = "real" if label == "Real" else "fake"
#                 icon      = "✅" if label == "Real" else "⚠️"

#                 st.markdown(f"""
#                 <div class="result-box {css_class}">
#                     {icon} {label}
#                     <div class="confidence">
#                         Confidence: {confidence:.1f}%
#                         &nbsp;|&nbsp;
#                         Real: {probs[0]*100:.1f}%&nbsp;&nbsp;Fake: {probs[1]*100:.1f}%
#                     </div>
#                 </div>
#                 """, unsafe_allow_html=True)

#             except Exception as e:
#                 st.error(f" Prediction failed: {e}")
# else:
#     st.info("Upload an image above to get started.")



"""
DeepFake Detector — Streamlit App
Model: ViT-B/16
Supports: Images (JPG/PNG/WEBP) + Videos (MP4/MOV/AVI)
"""

import os, io, time, tempfile, math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import timm
import torchvision.transforms as T
import gdown
import streamlit as st
from PIL import Image

# ── Optional video deps ───────────────────────────────────────────────────────
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg:       #07090f;
    --surface:  #0f1118;
    --border:   #1c2035;
    --accent:   #4fffb0;
    --danger:   #ff4d6d;
    --text:     #dde3f0;
    --muted:    #525d7a;
    --mono:     'IBM Plex Mono', monospace;
    --sans:     'Syne', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--sans) !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header, [data-testid="stSidebar"] { display: none !important; }
.block-container { padding-top: 2.5rem !important; max-width: 960px !important; }

/* ── Header ── */
.df-header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
}
.df-title {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1;
}
.df-title span { color: var(--accent); }
.df-badge {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--muted);
    border: 1px solid var(--border);
    padding: 4px 12px;
    border-radius: 4px;
    letter-spacing: 0.08em;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 10px !important;
    transition: border-color .2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}
[data-testid="stFileUploader"] label { display: none !important; }

/* ── Analyze button ── */
.stButton > button {
    width: 100% !important;
    background: var(--accent) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--mono) !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.06em !important;
    padding: 0.65rem 1.5rem !important;
    margin-top: 0.5rem !important;
    transition: opacity .2s !important;
}
.stButton > button:hover { opacity: 0.8 !important; }

/* ── Verdict card ── */
.verdict-wrap {
    border-radius: 10px;
    padding: 1.8rem 2rem 1.4rem;
    margin-top: 1.2rem;
    position: relative;
    overflow: hidden;
}
.verdict-wrap::after {
    content: '';
    position: absolute;
    inset: 0;
    opacity: 0.04;
    pointer-events: none;
}
.verdict-real { background: #001c10; border: 1.5px solid var(--accent); }
.verdict-fake { background: #1a0010; border: 1.5px solid var(--danger); }
.verdict-real::after { background: var(--accent); }
.verdict-fake::after { background: var(--danger); }

.vdict-tag {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.4rem;
}
.vdict-label {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1;
}
.vdict-real .vdict-label { color: var(--accent); }
.vdict-fake .vdict-label { color: var(--danger); }
.vdict-conf {
    font-family: var(--mono);
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 0.5rem;
}

/* ── Prob bars ── */
.bar-section { margin-top: 1.2rem; }
.bar-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}
.bar-name {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--muted);
    width: 38px;
    flex-shrink: 0;
}
.bar-track {
    flex: 1;
    height: 5px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
}
.bar-fill { height: 100%; border-radius: 3px; transition: width .6s ease; }
.fill-real { background: var(--accent); }
.fill-fake { background: var(--danger); }
.bar-pct {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--muted);
    width: 40px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.8rem 0 !important; }

/* ── Meta line under image ── */
.img-meta {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--muted);
    margin-top: 6px;
}

/* ── Video metrics ── */
div[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 0.9rem 1.1rem !important;
}
div[data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    font-size: 1.3rem !important;
}
div[data-testid="stMetricLabel"] {
    font-family: var(--mono) !important;
    font-size: 0.7rem !important;
    color: var(--muted) !important;
}

/* ── Frame grid labels ── */
.frame-label {
    text-align: center;
    font-family: var(--mono);
    font-size: 0.65rem;
    margin-top: 3px;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Info box ── */
.stAlert { background: var(--surface) !important; border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
WEIGHTS_PATH     = "deepfake_detection_image.h5"
IMG_SIZE         = 224
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES          = ["Real", "Fake"]
MEAN             = [0.485, 0.456, 0.406]
STD              = [0.229, 0.224, 0.225]
VIDEO_SAMPLE_FPS = 1
MAX_VIDEO_FRAMES = 120


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

class ViTDeepFakeDetector(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            'vit_base_patch16_224', pretrained=False,
            num_classes=0, global_pool='token'
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Dropout(dropout),
            nn.Linear(feat_dim, 512), nn.GELU(),
            nn.LayerNorm(512), nn.Dropout(dropout * 0.67),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHT LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _detect_format(path: str) -> str:
    try:
        with open(path, 'rb') as f:
            magic = f.read(8)
        if magic[:4] == b'\x89HDF':
            return 'h5'
        if magic[:2] == b'PK':
            return 'pt'
        if magic[:2] == b'\x80\x02':
            return 'pt'
    except Exception:
        pass
    return 'unknown'


def _load_h5(path: str, device):
    import h5py
    with h5py.File(path, 'r') as hf:
        raw = bytes(hf['model_weights'][:].tobytes())
    buf = io.BytesIO(raw)
    try:
        return torch.load(buf, map_location=device, weights_only=True)
    except Exception:
        buf.seek(0)
        return torch.load(buf, map_location=device, weights_only=False)


def _safe_load(path: str, device):
    fmt = _detect_format(path)
    if fmt == 'h5':
        return _load_h5(path, device)
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


def download_weights():
    fid = st.secrets.get("gdrive", {}).get("deepfake_image")
    if not fid:
        st.error("Missing secret: add `deepfake_image` under `[gdrive]` in your secrets.")
        st.stop()
    if not os.path.exists(WEIGHTS_PATH):
        with st.spinner("Downloading model weights… (first run only)"):
            gdown.download(f"https://drive.google.com/uc?id={fid}", WEIGHTS_PATH, quiet=False)
        if not os.path.exists(WEIGHTS_PATH):
            st.error("Download failed. Make sure the Drive file is set to 'Anyone with the link can view'.")
            st.stop()


@st.cache_resource(show_spinner="Loading ViT-B/16 weights…")
def load_model():
    download_weights()
    raw = _safe_load(WEIGHTS_PATH, DEVICE)
    if isinstance(raw, dict):
        state = raw.get('model_state_dict', raw.get('state_dict', raw))
    else:
        state = raw

    model = ViTDeepFakeDetector()
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    return model


TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def predict(model, pil_img: Image.Image) -> dict:
    tensor = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    return {
        "label":      CLASSES[idx],
        "confidence": float(probs[idx]) * 100,
        "p_real":     float(probs[0]) * 100,
        "p_fake":     float(probs[1]) * 100,
    }


def extract_frames(video_path: str):
    if not HAS_CV2:
        return [], []
    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    step   = max(1, int(fps / VIDEO_SAMPLE_FPS))
    frames, timestamps = [], []
    idx = 0
    while cap.isOpened() and len(frames) < MAX_VIDEO_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            timestamps.append(idx / fps)
        idx += 1
    cap.release()
    return frames, timestamps


def analyse_video(model, video_path: str) -> dict:
    frames, timestamps = extract_frames(video_path)
    if not frames:
        return None

    results = []
    bar = st.progress(0, text="Analysing frames…")
    for i, (frame, ts) in enumerate(zip(frames, timestamps)):
        r = predict(model, frame)
        r['timestamp'] = ts
        r['frame']     = frame
        results.append(r)
        bar.progress((i + 1) / len(frames), text=f"Frame {i+1}/{len(frames)}  [{ts:.1f}s]")
    bar.empty()

    fake_frames = [r for r in results if r['label'] == 'Fake']
    fake_pct    = len(fake_frames) / len(results) * 100
    avg_p_fake  = float(np.mean([r['p_fake'] for r in results]))
    verdict     = 'Fake' if fake_pct > 40 else 'Real'
    return {
        'verdict':       verdict,
        'avg_p_fake':    avg_p_fake,
        'avg_p_real':    100 - avg_p_fake,
        'fake_pct':      fake_pct,
        'total_frames':  len(results),
        'fake_frames':   len(fake_frames),
        'frame_results': results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def render_verdict(label: str, confidence: float, p_real: float, p_fake: float):
    css  = "verdict-real" if label == "Real" else "verdict-fake"
    vcss = "vdict-real"   if label == "Real" else "vdict-fake"
    icon = "✓" if label == "Real" else "✗"
    st.markdown(f"""
    <div class="verdict-wrap {css}">
        <div class="{vcss}">
            <div class="vdict-tag">VERDICT</div>
            <div class="vdict-label">{icon}&ensp;{label.upper()}</div>
            <div class="vdict-conf">Confidence: {confidence:.1f}%</div>
        </div>
        <div class="bar-section">
            <div class="bar-row">
                <div class="bar-name">Real</div>
                <div class="bar-track"><div class="bar-fill fill-real" style="width:{p_real:.1f}%"></div></div>
                <div class="bar-pct">{p_real:.1f}%</div>
            </div>
            <div class="bar-row">
                <div class="bar-name">Fake</div>
                <div class="bar-track"><div class="bar-fill fill-fake" style="width:{p_fake:.1f}%"></div></div>
                <div class="bar-pct">{p_fake:.1f}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_video_verdict(result: dict):
    v    = result['verdict']
    css  = "verdict-real" if v == "Real" else "verdict-fake"
    vcss = "vdict-real"   if v == "Real" else "vdict-fake"
    icon = "✓" if v == "Real" else "✗"
    st.markdown(f"""
    <div class="verdict-wrap {css}">
        <div class="{vcss}">
            <div class="vdict-tag">VERDICT</div>
            <div class="vdict-label">{icon}&ensp;{v.upper()}</div>
            <div class="vdict-conf">
                {result['fake_frames']} / {result['total_frames']} frames flagged fake
                &nbsp;·&nbsp; Avg P(Fake): {result['avg_p_fake']:.1f}%
            </div>
        </div>
        <div class="bar-section">
            <div class="bar-row">
                <div class="bar-name">Real</div>
                <div class="bar-track"><div class="bar-fill fill-real" style="width:{result['avg_p_real']:.1f}%"></div></div>
                <div class="bar-pct">{result['avg_p_real']:.1f}%</div>
            </div>
            <div class="bar-row">
                <div class="bar-name">Fake</div>
                <div class="bar-track"><div class="bar-fill fill-fake" style="width:{result['avg_p_fake']:.1f}%"></div></div>
                <div class="bar-pct">{result['avg_p_fake']:.1f}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_frame_grid(frame_results: list, max_show: int = 24):
    st.markdown("#### Frame-by-Frame Analysis")
    step   = max(1, math.ceil(len(frame_results) / max_show))
    subset = frame_results[::step][:max_show]
    cols   = st.columns(min(6, len(subset)))
    for i, r in enumerate(subset):
        color = "#4fffb0" if r['label'] == "Real" else "#ff4d6d"
        with cols[i % len(cols)]:
            st.image(r['frame'], use_column_width=True)
            st.markdown(
                f"<div class='frame-label' style='color:{color}'>"
                f"{r['label']} {r['confidence']:.0f}% · {r['timestamp']:.1f}s"
                f"</div>",
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════
try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="df-header">
    <div class="df-title">Deep<span>Fake</span> Detector</div>
    <div class="df-badge">ViT-B/16 · RESEARCH ONLY</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_img, tab_vid = st.tabs(["🖼️  Image Analysis", "🎬  Video Analysis"])


# ── IMAGE TAB ─────────────────────────────────────────────────────────────────
with tab_img:
    st.markdown(
        "<p style='color:var(--muted);font-size:0.85rem;margin-bottom:1.2rem'>"
        "Upload one or more images to check if they are authentic or AI-generated."
        "</p>",
        unsafe_allow_html=True,
    )

    uploaded_imgs = st.file_uploader(
        "Images",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=True,
        key="img_uploader",
        label_visibility="collapsed",
    )

    if uploaded_imgs:
        if st.button("ANALYZE IMAGES", key="img_btn"):
            for uploaded in uploaded_imgs:
                st.markdown("---")
                pil_img = Image.open(uploaded).convert("RGB")
                col_img, col_res = st.columns([1, 1], gap="large")

                with col_img:
                    st.image(pil_img, use_column_width=True, caption=uploaded.name)
                    w, h = pil_img.size
                    st.markdown(
                        f"<div class='img-meta'>{w}×{h}px · {uploaded.size / 1024:.0f} KB</div>",
                        unsafe_allow_html=True,
                    )

                with col_res:
                    with st.spinner("Running inference…"):
                        t0  = time.time()
                        res = predict(model, pil_img)
                        ms  = (time.time() - t0) * 1000
                    render_verdict(res['label'], res['confidence'], res['p_real'], res['p_fake'])
                    st.markdown(
                        f"<div style='font-family:var(--mono);font-size:0.7rem;"
                        f"color:var(--muted);margin-top:8px'>Inference: {ms:.0f} ms</div>",
                        unsafe_allow_html=True,
                    )
    else:
        st.info("Drop one or more images above to get started.")


# ── VIDEO TAB ─────────────────────────────────────────────────────────────────
with tab_vid:
    if not HAS_CV2:
        st.error(
            "OpenCV is required for video analysis.\n\n"
            "Add `opencv-python-headless` to your `requirements.txt` and redeploy."
        )
        st.stop()

    st.markdown(
        f"<p style='color:var(--muted);font-size:0.85rem;margin-bottom:1.2rem'>"
        f"Upload a video for frame-by-frame analysis. "
        f"Samples at 1 fps · max {MAX_VIDEO_FRAMES} frames per video."
        f"</p>",
        unsafe_allow_html=True,
    )

    uploaded_vid = st.file_uploader(
        "Video",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        key="vid_uploader",
        label_visibility="collapsed",
    )

    if uploaded_vid:
        st.video(uploaded_vid)
        size_mb = uploaded_vid.size / 1e6
        st.markdown(
            f"<p style='font-family:var(--mono);font-size:0.72rem;color:var(--muted)'>"
            f"{uploaded_vid.name} · {size_mb:.1f} MB</p>",
            unsafe_allow_html=True,
        )

        if st.button("ANALYZE VIDEO", key="vid_btn"):
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(uploaded_vid.name).suffix
            ) as tmp:
                tmp.write(uploaded_vid.read())
                tmp_path = tmp.name

            try:
                result = analyse_video(model, tmp_path)

                if result is None:
                    st.error("Could not extract frames. Please check the video file.")
                else:
                    render_video_verdict(result)
                    st.markdown("---")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Frames", result['total_frames'])
                    c2.metric("Fake Frames",  result['fake_frames'])
                    c3.metric("Fake %",        f"{result['fake_pct']:.1f}%")
                    c4.metric("Avg P(Fake)",   f"{result['avg_p_fake']:.1f}%")
                    st.markdown("---")
                    render_frame_grid(result['frame_results'])
            finally:
                os.unlink(tmp_path)
    else:
        st.info("Drop a video above to get started.")


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:4rem;padding-top:1rem;border-top:1px solid var(--border);
     text-align:center;font-size:0.7rem;color:var(--muted);font-family:var(--mono)">
    DeepFake Detector · ViT-B/16 · For research purposes only
</div>
""", unsafe_allow_html=True)
