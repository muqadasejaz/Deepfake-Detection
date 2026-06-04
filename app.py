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
Supports: Images (JPG/PNG/WEBP) + Videos (MP4/MOV/AVI)
Model: ViT-B/16 (primary) | XceptionNet (fallback key: 'deepfake_image')
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
# CSS — Dark forensic aesthetic, monospace accents, sharp geometry
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0a0c10;
    --surface:   #111318;
    --border:    #1e2330;
    --accent:    #00e5ff;
    --accent2:   #ff3b6b;
    --real:      #00c97a;
    --fake:      #ff3b6b;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --mono:      'Space Mono', monospace;
    --sans:      'DM Sans', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--sans);
    background:  var(--bg) !important;
    color:       var(--text) !important;
}

/* ── Header ── */
.app-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.app-logo {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, var(--accent), #006080);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px; flex-shrink: 0;
}
.app-title  { font-size: 1.7rem; font-weight: 600; letter-spacing: -0.02em; }
.app-sub    { font-size: 0.85rem; color: var(--muted); font-family: var(--mono); margin-top:2px; }

/* ── Tabs ── */
.tab-row {
    display: flex; gap: 2px; margin-bottom: 2rem;
    border-bottom: 1px solid var(--border); padding-bottom: 0;
}
.tab-btn {
    padding: 10px 24px; background: transparent;
    border: none; color: var(--muted); cursor: pointer;
    font-family: var(--mono); font-size: 0.82rem;
    border-bottom: 2px solid transparent; margin-bottom: -1px;
    transition: all .2s;
}
.tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); }
.tab-btn:hover  { color: var(--text); }

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}

/* ── Result cards ── */
.verdict {
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin-top: 1.5rem;
    position: relative; overflow: hidden;
}
.verdict::before {
    content: '';
    position: absolute; inset: 0;
    opacity: 0.06;
}
.verdict-real { background: #001a0e; border: 1.5px solid var(--real); }
.verdict-fake { background: #1a0008; border: 1.5px solid var(--fake); }
.verdict-real::before { background: var(--real); }
.verdict-fake::before { background: var(--fake); }

.verdict-label {
    font-size: 2rem; font-weight: 700;
    font-family: var(--mono); letter-spacing: 0.05em;
}
.verdict-real .verdict-label { color: var(--real); }
.verdict-fake .verdict-label { color: var(--fake); }
.verdict-conf  { font-size: 0.9rem; color: var(--muted); margin-top: 6px; font-family: var(--mono); }

/* ── Confidence bar ── */
.conf-bar-wrap { margin-top: 1.5rem; }
.conf-bar-label {
    display: flex; justify-content: space-between;
    font-size: 0.78rem; font-family: var(--mono); color: var(--muted);
    margin-bottom: 4px;
}
.conf-bar-bg {
    height: 6px; background: var(--border); border-radius: 3px; overflow: hidden;
}
.conf-bar-fill { height: 100%; border-radius: 3px; }
.bar-real { background: var(--real); }
.bar-fake { background: var(--fake); }

/* ── Video frame grid ── */
.frame-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 8px; margin-top: 1rem;
}
.frame-cell {
    border-radius: 8px; overflow: hidden; position: relative;
    border: 1.5px solid var(--border);
}
.frame-label {
    position: absolute; bottom: 4px; left: 50%; transform: translateX(-50%);
    font-size: 10px; font-family: var(--mono); font-weight: 700;
    padding: 2px 8px; border-radius: 4px; white-space: nowrap;
}
.label-real { background: var(--real); color: #000; }
.label-fake { background: var(--fake); color: #fff; }

/* ── Info chip ── */
.chip {
    display: inline-block; padding: 3px 12px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 20px; font-size: 0.75rem;
    font-family: var(--mono); color: var(--muted); margin: 2px;
}

/* ── Streamlit overrides ── */
.stButton > button {
    background: var(--accent) !important; color: #000 !important;
    border: none !important; border-radius: 8px !important;
    font-family: var(--mono) !important; font-weight: 700 !important;
    font-size: 0.85rem !important; padding: 0.6rem 2rem !important;
    transition: opacity .2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

.stSpinner > div { border-top-color: var(--accent) !important; }

div[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}
div[data-testid="stMetricValue"] { font-family: var(--mono) !important; }

[data-testid="stSidebar"] { background: var(--surface) !important; }
hr { border-color: var(--border) !important; }

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
WEIGHTS_PATH = "deepfake_vit.pt"
XCEPTION_PATH = "xceptionnet_weights.pth"
IMG_SIZE     = 224          # ViT-B/16 native
XCEPTION_SIZE = 299
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES      = ["Real", "Fake"]
MEAN         = [0.485, 0.456, 0.406]
STD          = [0.229, 0.224, 0.225]
VIDEO_SAMPLE_FPS = 1        # analyse 1 frame per second
MAX_VIDEO_FRAMES = 120      # cap at 120 frames (~2 min at 1fps)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
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


class XceptionDeepFakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = timm.create_model(
            'xception', pretrained=False, num_classes=num_classes
        )
    def forward(self, x):
        return self.model(x)


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHT DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

def _detect_file_format(path: str) -> str:
    """
    Detect whether a weights file is HDF5 (.h5) or PyTorch pickle (.pt/.pth).
    Reads the first 8 bytes — HDF5 magic is b'\\x89HDF\\r\\n\\x1a\\n'.
    PyTorch zip-based files start with b'PK'.
    Returns 'h5', 'pt', or 'unknown'.
    """
    try:
        with open(path, 'rb') as f:
            magic = f.read(8)
        if magic[:4] == b'\x89HDF':
            return 'h5'
        if magic[:2] == b'PK':       # zip/torch.save format
            return 'pt'
        if magic[:2] == b'\x80\x02': # legacy pickle
            return 'pt'
    except Exception:
        pass
    return 'unknown'


def _load_from_h5(path: str, device):
    """
    Extract the PyTorch state dict stored inside an HDF5 file
    written by the Kaggle training notebook (dataset key: 'model_weights').
    """
    import h5py, io
    with h5py.File(path, 'r') as hf:
        raw = bytes(hf['model_weights'][:].tobytes())
    buf = io.BytesIO(raw)
    try:
        state = torch.load(buf, map_location=device, weights_only=True)
    except Exception:
        buf.seek(0)
        state = torch.load(buf, map_location=device, weights_only=False)
    return state


def _safe_load(path: str, device):
    """
    Auto-detect file format and load state dict correctly.
    Handles:
      - HDF5  (.h5)  — saved by Kaggle notebook via h5py
      - PyTorch zip  (.pt / .pth) — saved by torch.save()
    Returns a state dict or checkpoint dict.
    """
    fmt = _detect_file_format(path)
    if fmt == 'h5':
        return _load_from_h5(path, device)   # returns state dict directly
    else:
        try:
            return torch.load(path, map_location=device, weights_only=True)
        except Exception:
            return torch.load(path, map_location=device, weights_only=False)


def download_weights():
    """Download from Google Drive using secrets. Single key supported."""
    secrets  = st.secrets.get("gdrive", {})
    local_path = XCEPTION_PATH   # default download target

    # Use deepfake_vit if present, else fall back to deepfake_image
    fid = secrets.get("deepfake_vit") or secrets.get("deepfake_image")

    if not fid:
        st.error("No Google Drive file ID found in secrets.\n"
                 "Add `deepfake_image` or `deepfake_vit` under `[gdrive]`.")
        st.stop()

    if not os.path.exists(local_path):
        with st.spinner("⬇️  Downloading model weights… (first run only)"):
            gdown.download(
                f"https://drive.google.com/uc?id={fid}",
                local_path, quiet=False
            )
        if not os.path.exists(local_path):
            st.error("Download failed. Make sure the Drive file is set to "
                     "'Anyone with the link can view'.")
            st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING (cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="🧠 Loading model weights…")
def load_model(arch: str):
    download_weights()

    # ── Determine which local file to use ─────────────────────────────────────
    weights_file = XCEPTION_PATH   # download_weights() always saves here

    # ── Detect format BEFORE choosing model class ─────────────────────────────
    fmt = _detect_file_format(weights_file)

    # ── Load raw state dict ───────────────────────────────────────────────────
    raw = _safe_load(weights_file, DEVICE)

    # raw may be: state_dict directly, or a checkpoint dict containing one
    if isinstance(raw, dict):
        state = raw.get('model_state_dict',
                raw.get('state_dict', raw))
    else:
        state = raw

    # ── Pick architecture ─────────────────────────────────────────────────────
    # Auto-detect from state dict key names if possible
    key_sample = next(iter(state.keys()), "")
    if 'backbone' in key_sample or 'head' in key_sample:
        detected_arch = 'vit'
    elif 'model.conv1' in key_sample or 'model.bn1' in key_sample:
        detected_arch = 'xception'
    else:
        detected_arch = arch   # trust user setting

    if detected_arch == 'vit':
        model = ViTDeepFakeDetector()
    else:
        model = XceptionDeepFakeDetector()

    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        pass   # silently ignore extra keys (e.g. from different head versions)

    model.to(DEVICE)
    model.eval()
    return model, detected_arch


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════

def get_transform(arch):
    size = IMG_SIZE if arch == "vit" else XCEPTION_SIZE
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def predict_image(model, pil_img: Image.Image, transform) -> dict:
    tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx        = int(np.argmax(probs))
    return {
        "label":      CLASSES[idx],
        "confidence": float(probs[idx]) * 100,
        "p_real":     float(probs[0]) * 100,
        "p_fake":     float(probs[1]) * 100,
    }


def extract_frames(video_path: str,
                   sample_fps: int = VIDEO_SAMPLE_FPS,
                   max_frames: int = MAX_VIDEO_FRAMES):
    """Extract frames at sample_fps rate. Returns list of PIL Images + timestamps."""
    if not HAS_CV2:
        return [], []

    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step   = max(1, int(fps / sample_fps))

    frames, timestamps = [], []
    frame_idx = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil   = Image.fromarray(rgb)
            ts    = frame_idx / fps
            frames.append(pil)
            timestamps.append(ts)
        frame_idx += 1

    cap.release()
    return frames, timestamps


def analyse_video(model, video_path: str, transform) -> dict:
    frames, timestamps = extract_frames(video_path)
    if not frames:
        return None

    results = []
    bar = st.progress(0, text="Analysing frames…")
    for i, (frame, ts) in enumerate(zip(frames, timestamps)):
        r = predict_image(model, frame, transform)
        r['timestamp'] = ts
        r['frame']     = frame
        results.append(r)
        bar.progress((i + 1) / len(frames),
                     text=f"Frame {i+1}/{len(frames)}  [{ts:.1f}s]")
    bar.empty()

    fake_results = [r for r in results if r['label'] == 'Fake']
    fake_pct     = len(fake_results) / len(results) * 100
    avg_p_fake   = np.mean([r['p_fake'] for r in results])
    verdict      = 'Fake' if fake_pct > 40 else 'Real'
    conf         = avg_p_fake if verdict == 'Fake' else (100 - avg_p_fake)

    return {
        'verdict':    verdict,
        'confidence': conf,
        'avg_p_fake': avg_p_fake,
        'avg_p_real': 100 - avg_p_fake,
        'fake_pct':   fake_pct,
        'total_frames': len(results),
        'fake_frames':  len(fake_results),
        'frame_results': results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def render_verdict(label: str, confidence: float, p_real: float, p_fake: float):
    css   = "verdict-real" if label == "Real" else "verdict-fake"
    icon  = "✅" if label == "Real" else "⚠️"
    color = "var(--real)" if label == "Real" else "var(--fake)"

    st.markdown(f"""
    <div class="verdict {css}">
        <div class="verdict-label">{icon}&nbsp;&nbsp;{label.upper()}</div>
        <div class="verdict-conf">Confidence: {confidence:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<div class="conf-bar-wrap">""", unsafe_allow_html=True)
    for name, val, bar_cls in [("Real", p_real, "bar-real"),
                                ("Fake", p_fake, "bar-fake")]:
        st.markdown(f"""
        <div class="conf-bar-label"><span>{name}</span><span>{val:.1f}%</span></div>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill {bar_cls}" style="width:{val}%"></div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_video_verdict(result: dict):
    v     = result['verdict']
    css   = "verdict-real" if v == "Real" else "verdict-fake"
    icon  = "✅" if v == "Real" else "⚠️"

    st.markdown(f"""
    <div class="verdict {css}">
        <div class="verdict-label">{icon}&nbsp;&nbsp;{v.upper()}</div>
        <div class="verdict-conf">
            {result['fake_frames']} / {result['total_frames']} frames flagged as Fake
            &nbsp;·&nbsp; Avg P(Fake): {result['avg_p_fake']:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Bars
    st.markdown("""<div class="conf-bar-wrap">""", unsafe_allow_html=True)
    for name, val, bar_cls in [("Avg Real", result['avg_p_real'], "bar-real"),
                                ("Avg Fake", result['avg_p_fake'], "bar-fake")]:
        st.markdown(f"""
        <div class="conf-bar-label"><span>{name}</span><span>{val:.1f}%</span></div>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill {bar_cls}" style="width:{val}%"></div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_frame_grid(frame_results: list, max_show: int = 24):
    """Show a grid of sampled frames with per-frame verdict labels."""
    st.markdown("#### Frame Analysis")
    step   = max(1, math.ceil(len(frame_results) / max_show))
    subset = frame_results[::step][:max_show]

    cols = st.columns(min(6, len(subset)))
    for i, r in enumerate(subset):
        col = cols[i % len(cols)]
        with col:
            # Extract expressions to variables — backslashes not allowed
            # inside f-string {} in Python < 3.12
            frame_color = "#00c97a" if r['label'] == "Real" else "#ff3b6b"
            frame_label = r['label']
            frame_conf  = f"{r['confidence']:.0f}%"
            frame_ts    = f"{r['timestamp']:.1f}s"
            st.image(r['frame'], use_column_width=True)
            st.markdown(
                f"<div style='text-align:center;font-size:10px;"
                f"font-family:var(--mono);color:{frame_color}'>"
                f"{frame_label} {frame_conf}&nbsp;|&nbsp;{frame_ts}"
                f"</div>",
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="app-header">
    <div class="app-logo">🛡️</div>
    <div>
        <div class="app-title">DeepFake Detector</div>
        <div class="app-sub">ViT-B/16 · Computer Vision · Image & Video Analysis</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Model selector (sidebar) ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    arch = st.selectbox(
        "Model Architecture",
        ["vit", "xception"],
        format_func=lambda x: "ViT-B/16 (recommended)" if x == "vit" else "XceptionNet (legacy)",
    )
    st.markdown("---")
    st.markdown(f"""
    <div class="chip">Device: {DEVICE}</div>
    <div class="chip">Input: {IMG_SIZE if arch=='vit' else XCEPTION_SIZE}px</div>
    <div class="chip">Threshold: 0.5</div>
    """, unsafe_allow_html=True)

    if not HAS_CV2:
        st.warning("OpenCV not installed — video analysis unavailable.\n\n"
                   "Add `opencv-python-headless` to requirements.txt")

# ── Load model ────────────────────────────────────────────────────────────────
try:
    model, detected_arch = load_model(arch)
    transform = get_transform(detected_arch)   # use auto-detected arch for correct input size
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ── Mode tabs ─────────────────────────────────────────────────────────────────
tab_img, tab_vid = st.tabs(["🖼️  Image Analysis", "🎬  Video Analysis"])


# ════════════════════════════════════════════════
# TAB 1 — IMAGE
# ════════════════════════════════════════════════
with tab_img:
    st.markdown("Upload one or more images to check if they are real or AI-generated/deepfake.")

    uploaded_imgs = st.file_uploader(
        "Drop images here",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=True,
        key="img_uploader",
        label_visibility="collapsed",
    )

    if uploaded_imgs:
        analyze_btn = st.button("🔍 Analyze Images", key="img_btn")

        if analyze_btn:
            for uploaded in uploaded_imgs:
                st.markdown("---")
                pil_img = Image.open(uploaded).convert("RGB")

                col_img, col_res = st.columns([1, 1], gap="large")

                with col_img:
                    st.image(pil_img, use_column_width=True,
                             caption=uploaded.name)
                    w, h = pil_img.size
                    st.markdown(
                        f'<div style="font-size:0.75rem;font-family:var(--mono);'
                        f'color:var(--muted);margin-top:4px">'
                        f'{w}×{h}px · {uploaded.size/1024:.0f} KB</div>',
                        unsafe_allow_html=True
                    )

                with col_res:
                    with st.spinner("Analysing…"):
                        t0  = time.time()
                        res = predict_image(model, pil_img, transform)
                        ms  = (time.time() - t0) * 1000

                    render_verdict(
                        res['label'], res['confidence'],
                        res['p_real'], res['p_fake']
                    )
                    st.caption(f"Inference: {ms:.0f} ms · Model: {arch.upper()}")
    else:
        st.info("Upload an image above to get started.")


# ════════════════════════════════════════════════
# TAB 2 — VIDEO
# ════════════════════════════════════════════════
with tab_vid:
    if not HAS_CV2:
        st.error("OpenCV is required for video analysis.\n\n"
                 "Add `opencv-python-headless` to your `requirements.txt` and redeploy.")
        st.stop()

    st.markdown(
        f"Upload a video to scan it frame-by-frame. "
        f"Analysing **1 frame/sec** · max **{MAX_VIDEO_FRAMES} frames** per video."
    )

    uploaded_vid = st.file_uploader(
        "Drop video here",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        key="vid_uploader",
        label_visibility="collapsed",
    )

    if uploaded_vid:
        # Preview
        st.video(uploaded_vid)
        size_mb = uploaded_vid.size / 1e6
        st.markdown(
            f'<div class="chip">{uploaded_vid.name}</div>'
            f'<div class="chip">{size_mb:.1f} MB</div>',
            unsafe_allow_html=True
        )

        analyze_vid_btn = st.button("🎬 Analyze Video", key="vid_btn")

        if analyze_vid_btn:
            # Write to temp file (cv2 needs a real path)
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(uploaded_vid.name).suffix
            ) as tmp:
                tmp.write(uploaded_vid.read())
                tmp_path = tmp.name

            try:
                with st.spinner("Extracting and analysing frames…"):
                    result = analyse_video(model, tmp_path, transform)

                if result is None:
                    st.error("Could not extract frames. Check the video file.")
                else:
                    render_video_verdict(result)

                    st.markdown("---")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Frames",  result['total_frames'])
                    c2.metric("Fake Frames",   result['fake_frames'])
                    c3.metric("Fake %",        f"{result['fake_pct']:.1f}%")
                    c4.metric("Avg P(Fake)",   f"{result['avg_p_fake']:.1f}%")

                    st.markdown("---")
                    render_frame_grid(result['frame_results'])

            finally:
                os.unlink(tmp_path)
    else:
        st.info("Upload a video above to get started.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:4rem;padding-top:1rem;border-top:1px solid var(--border);
     text-align:center;font-size:0.75rem;color:var(--muted);font-family:var(--mono)">
    ViT-B/16 DeepFake Detector &nbsp;·&nbsp; Computer Vision Research &nbsp;·&nbsp;
    For research purposes only
</div>
""", unsafe_allow_html=True)
