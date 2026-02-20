"""
🛰️ RSI-CB128 — Satellite Image Classifier
Application Streamlit — CNN PyTorch
Auteur : TSANGNING GRACE | M2 Deep Learning 2025/2026
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import json
import pickle
import os
import io
import time
import plotly.graph_objects as go
import plotly.express as px
from torchvision import transforms
from pathlib import Path

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RSI-CB128 Classifier | CNN PyTorch",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Dark Space Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* === IMPORT FONTS === */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;600;800&display=swap');

/* === ROOT VARIABLES === */
:root {
    --bg-deep:     #040d1a;
    --bg-panel:    #071428;
    --bg-card:     #0b1e3d;
    --accent-cyan: #00f0ff;
    --accent-lime: #a0ff00;
    --accent-violet: #7b5ea7;
    --text-primary: #e8f4ff;
    --text-muted:   #6b8db5;
    --border:       rgba(0, 240, 255, 0.15);
    --glow:         0 0 20px rgba(0, 240, 255, 0.3);
}

/* === GLOBAL === */
.stApp {
    background: var(--bg-deep) !important;
    font-family: 'Outfit', sans-serif;
    color: var(--text-primary);
}

/* === HEADER HERO === */
.hero-container {
    background: linear-gradient(135deg, #071428 0%, #0d2149 50%, #071428 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 40px 50px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
    box-shadow: var(--glow), inset 0 1px 0 rgba(0, 240, 255, 0.2);
}
.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -20%;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(0, 240, 255, 0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-container::after {
    content: '';
    position: absolute;
    bottom: -30%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(123, 94, 167, 0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    color: var(--accent-cyan);
    letter-spacing: -1px;
    margin: 0 0 8px 0;
    text-shadow: 0 0 40px rgba(0, 240, 255, 0.5);
}
.hero-subtitle {
    font-size: 1.1rem;
    color: var(--text-muted);
    font-weight: 300;
    margin: 0 0 24px 0;
}
.hero-badges {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
}
.badge {
    background: rgba(0, 240, 255, 0.08);
    border: 1px solid rgba(0, 240, 255, 0.25);
    color: var(--accent-cyan);
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.5px;
}
.badge-lime {
    background: rgba(160, 255, 0, 0.07);
    border-color: rgba(160, 255, 0, 0.2);
    color: var(--accent-lime);
}
.badge-violet {
    background: rgba(123, 94, 167, 0.12);
    border-color: rgba(123, 94, 167, 0.3);
    color: #b89fd4;
}

/* === METRIC CARDS === */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 28px;
}
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--glow);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-violet));
}
.metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--text-muted);
    margin-bottom: 8px;
    font-family: 'Space Mono', monospace;
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: var(--accent-cyan);
    line-height: 1;
    margin-bottom: 4px;
    font-family: 'Space Mono', monospace;
}
.metric-sub {
    font-size: 0.8rem;
    color: var(--text-muted);
}

/* === SECTION TITLES === */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.15rem;
    color: var(--accent-cyan);
    margin: 0 0 20px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
}

/* === PREDICTION CARD === */
.pred-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px;
    margin-top: 20px;
    box-shadow: var(--glow);
}
.pred-class-name {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    color: var(--accent-lime);
    font-weight: 700;
    margin-bottom: 4px;
    text-shadow: 0 0 20px rgba(160, 255, 0, 0.4);
}
.pred-confidence {
    font-size: 2.8rem;
    font-weight: 800;
    color: var(--accent-cyan);
    font-family: 'Space Mono', monospace;
    line-height: 1;
    text-shadow: 0 0 30px rgba(0, 240, 255, 0.5);
}
.confidence-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    font-family: 'Space Mono', monospace;
}

/* === UPLOAD ZONE === */
.upload-zone {
    background: var(--bg-card);
    border: 2px dashed rgba(0, 240, 255, 0.3);
    border-radius: 16px;
    padding: 40px;
    text-align: center;
    transition: border-color 0.3s;
}
.upload-zone:hover {
    border-color: var(--accent-cyan);
}

/* === SIDEBAR === */
section[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .stMarkdown {
    color: var(--text-primary);
}

/* === TABS === */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-panel) !important;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
    padding: 8px 18px !important;
    border: none !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: rgba(0, 240, 255, 0.12) !important;
    color: var(--accent-cyan) !important;
    box-shadow: none !important;
}

/* === FILE UPLOADER === */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed rgba(0, 240, 255, 0.3) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"] label {
    color: var(--text-muted) !important;
}

/* === BUTTONS === */
.stButton > button {
    background: linear-gradient(135deg, rgba(0, 240, 255, 0.15), rgba(123, 94, 167, 0.15)) !important;
    border: 1px solid rgba(0, 240, 255, 0.4) !important;
    color: var(--accent-cyan) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
    letter-spacing: 1px !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    transition: all 0.2s !important;
    text-transform: uppercase;
}
.stButton > button:hover {
    background: rgba(0, 240, 255, 0.2) !important;
    box-shadow: var(--glow) !important;
    transform: translateY(-1px);
}

/* === PROGRESS BARS === */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-violet)) !important;
    border-radius: 10px !important;
}

/* === INFO / WARNING BOXES === */
.stInfo {
    background: rgba(0, 240, 255, 0.05) !important;
    border: 1px solid rgba(0, 240, 255, 0.2) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}

/* === DIVIDERS === */
hr {
    border-color: var(--border) !important;
}

/* === PLOTLY CHARTS BG === */
.js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}

/* === ARCHITECTURE BLOCKS === */
.arch-block {
    background: rgba(0, 240, 255, 0.04);
    border: 1px solid rgba(0, 240, 255, 0.12);
    border-left: 3px solid var(--accent-cyan);
    border-radius: 8px;
    padding: 12px 18px;
    margin: 8px 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    color: var(--text-primary);
}
.arch-block .block-label {
    color: var(--accent-cyan);
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 4px;
}

/* === STATUS DOT === */
.status-online { color: #00ff88; font-size: 0.8rem; }

/* === HIDE STREAMLIT BRANDING === */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* === SCROLLBAR === */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: rgba(0, 240, 255, 0.2); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────
IMG_SIZE = 128

class RSI_CNN(nn.Module):
    def __init__(self, num_classes):
        super(RSI_CNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=0),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2)
        )
        self._flat_size = self._get_flat_size()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flat_size, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def _get_flat_size(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            x = self.block1(dummy)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            return int(np.prod(x.shape[1:]))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.classifier(x)

# ─────────────────────────────────────────────
# DATA LOADING FUNCTIONS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_and_metadata():
    """Load model, class names, and metadata from saved files."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Try to load class names
    class_names = None
    for path in ['class_names.pkl', 'models/class_names.pkl']:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                class_names = pickle.load(f)
            break

    # Try to load metadata
    metadata = {}
    for path in ['models/model_info.json', 'model_metadata.json']:
        if os.path.exists(path):
            with open(path, 'r') as f:
                metadata = json.load(f)
            break

    if class_names is None and 'classes' in metadata:
        class_names = metadata['classes']

    if class_names is None:
        return None, None, None, None, None

    num_classes = len(class_names)
    model = RSI_CNN(num_classes=num_classes)

    # Try to load weights
    model_loaded = False
    for path in ['models/best_model.pth', 'best_model.pth']:
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model_loaded = True
            break

    if not model_loaded:
        return None, class_names, metadata, device, None

    model.to(device)
    model.eval()
    return model, class_names, metadata, device, True

@st.cache_data(show_spinner=False)
def load_training_history():
    for path in ['training_history.pkl']:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
    return None

# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)

def predict(model, image_tensor, device, class_names):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0].cpu().numpy()
        top5_indices = np.argsort(probabilities)[::-1][:5]
        top5_probs = probabilities[top5_indices]
        top5_classes = [class_names[i] for i in top5_indices]
    return top5_classes, top5_probs, probabilities

# ─────────────────────────────────────────────
# LOAD EVERYTHING
# ─────────────────────────────────────────────
model, class_names, metadata, device, model_loaded = load_model_and_metadata()
history = load_training_history()

# Fallback metadata for display even without model
if not metadata:
    metadata = {}

num_classes = len(class_names) if class_names else 45
test_acc = metadata.get('test_acc', metadata.get('test_accuracy', 0))
if test_acc and test_acc < 1:
    test_acc = test_acc * 100
best_val_acc = metadata.get('best_val_acc', metadata.get('best_val_accuracy', 0))
if best_val_acc and best_val_acc < 1:
    best_val_acc = best_val_acc * 100
total_params = metadata.get('total_params', metadata.get('total_parameters', 0))
epochs_trained = metadata.get('epochs_trained', metadata.get('num_epochs', 20))

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
        <div style='font-size:2.5rem'>🛰️</div>
        <div style='font-family: Space Mono; color: #00f0ff; font-size: 0.95rem; font-weight:700; margin-top:8px;'>RSI-CB128</div>
        <div style='font-family: Space Mono; color: #6b8db5; font-size: 0.72rem; margin-top:4px;'>CNN PYTORCH CLASSIFIER</div>
    </div>
    <hr style='border-color: rgba(0,240,255,0.1); margin: 16px 0;'/>
    """, unsafe_allow_html=True)

    if model_loaded:
        st.markdown('<span class="status-online">● MODÈLE CHARGÉ</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#ff4444; font-size:0.8rem;">● MODÈLE NON TROUVÉ</span>', unsafe_allow_html=True)
        st.info("Placez `best_model.pth` et `class_names.pkl` dans le dossier de l'app.")

    st.markdown("---")
    st.markdown("""
    <div style='font-family: Space Mono; font-size:0.72rem; color:#6b8db5; text-transform:uppercase; letter-spacing:2px; margin-bottom:12px;'>Navigation</div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "",
        options=["🔍 Classification", "📊 Performance", "🧠 Architecture", "ℹ️ À propos"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; color:#6b8db5; line-height:1.7;'>
        <div style='color:#00f0ff; font-family:Space Mono; font-size:0.72rem; letter-spacing:1px; margin-bottom:8px;'>INFOS MODÈLE</div>
        <div>🏗️ Architecture : RSI_CNN</div>
        <div>🧩 Framework : PyTorch</div>
        <div>🖼️ Input : 128×128 RGB</div>
        <div>🏷️ Classes : {}</div>
        <div>⚙️ Filtres : 32→64→128→256</div>
    </div>
    """.format(num_classes), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#6b8db5; text-align:center;'>
        TSANGNING GRACE<br/>M2 Deep Learning 2025/2026
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🛰️ RSI-CB128 Classifier</div>
    <div class="hero-subtitle">Classification d'Images Satellite par Deep Learning — CNN PyTorch</div>
    <div class="hero-badges">
        <span class="badge">PyTorch</span>
        <span class="badge">CNN 4 Blocs</span>
        <span class="badge lime">RSI-CB128</span>
        <span class="badge badge-violet">M2 Deep Learning</span>
        <span class="badge">128×128 px</span>
        <span class="badge">Adam Optimizer</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# METRICS ROW
# ─────────────────────────────────────────────
if metadata:
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Test Accuracy</div>
            <div class="metric-value">{test_acc:.1f}<span style="font-size:1.2rem">%</span></div>
            <div class="metric-sub">Sur le jeu de test</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Best Val Acc</div>
            <div class="metric-value">{best_val_acc:.1f}<span style="font-size:1.2rem">%</span></div>
            <div class="metric-sub">Epoch {metadata.get('best_epoch', '—')}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Classes</div>
            <div class="metric-value">{num_classes}</div>
            <div class="metric-sub">Catégories satellite</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Paramètres</div>
            <div class="metric-value">{total_params/1e6:.1f}<span style="font-size:1.2rem">M</span></div>
            <div class="metric-sub">Poids entraînables</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: CLASSIFICATION
# ─────────────────────────────────────────────
if page == "🔍 Classification":
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<div class="section-title">📡 Image Satellite</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Glissez une image satellite (JPG, PNG, TIF)",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            help="Formats supportés : JPG, PNG, TIF — RSI-CB128 (128×128 recommandé)"
        )

        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption=f"Image chargée : {uploaded_file.name}", use_container_width=True)

            st.markdown(f"""
            <div style='background: var(--bg-card); border: 1px solid var(--border); border-radius:10px;
                        padding:14px 18px; margin-top:12px; font-size:0.82rem; color:#6b8db5;
                        font-family: Space Mono;'>
                📐 Taille originale : {image.size[0]}×{image.size[1]} px<br/>
                🎨 Mode : {image.mode}<br/>
                📦 Fichier : {uploaded_file.name}<br/>
                💾 Taille : {uploaded_file.size / 1024:.1f} KB
            </div>
            """, unsafe_allow_html=True)

    with col_result:
        st.markdown('<div class="section-title">🎯 Résultat de Classification</div>', unsafe_allow_html=True)

        if not uploaded_file:
            st.markdown("""
            <div style='background: var(--bg-card); border: 1px dashed rgba(0,240,255,0.2);
                        border-radius: 14px; padding: 60px 30px; text-align: center; color: #6b8db5;'>
                <div style='font-size: 3rem; margin-bottom:16px;'>🛰️</div>
                <div style='font-family: Space Mono; font-size: 0.85rem;'>
                    Chargez une image satellite<br/>pour lancer la classification
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif not model_loaded:
            st.error("❌ Modèle non chargé. Vérifiez que `best_model.pth` est présent.")

        else:
            classify_btn = st.button("🚀 CLASSIFIER L'IMAGE", use_container_width=True)

            if classify_btn or (uploaded_file and 'last_file' not in st.session_state):
                st.session_state.last_file = uploaded_file.name

                with st.spinner("Analyse en cours..."):
                    progress = st.progress(0)
                    for i in range(100):
                        time.sleep(0.008)
                        progress.progress(i + 1)

                    tensor = preprocess_image(image)
                    top5_classes, top5_probs, all_probs = predict(model, tensor, device, class_names)
                    progress.empty()

                top_class = top5_classes[0]
                top_conf = top5_probs[0] * 100

                # Main result
                conf_color = "#00ff88" if top_conf >= 70 else "#ffcc00" if top_conf >= 40 else "#ff6666"
                st.markdown(f"""
                <div class="pred-card">
                    <div class="confidence-label">CLASSE PRÉDITE</div>
                    <div class="pred-class-name">{top_class.replace('_', ' ').title()}</div>
                    <div style="margin: 12px 0;">
                        <div class="confidence-label">CONFIANCE</div>
                        <div class="pred-confidence" style="color:{conf_color}">{top_conf:.1f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Top 5 chart
                st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)
                fig_top5 = go.Figure(go.Bar(
                    x=top5_probs * 100,
                    y=[c.replace('_', ' ').title() for c in top5_classes],
                    orientation='h',
                    marker=dict(
                        color=top5_probs * 100,
                        colorscale=[[0, '#0b1e3d'], [0.5, '#7b5ea7'], [1.0, '#00f0ff']],
                        showscale=False,
                        line=dict(color='rgba(0,240,255,0.3)', width=1)
                    ),
                    text=[f"{p*100:.1f}%" for p in top5_probs],
                    textposition='outside',
                    textfont=dict(color='#e8f4ff', size=11, family='Space Mono')
                ))
                fig_top5.update_layout(
                    title=dict(text="Top 5 Classes", font=dict(color='#6b8db5', size=12, family='Space Mono')),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#6b8db5'),
                    xaxis=dict(range=[0, min(top5_probs[0]*130, 100)],
                               showgrid=True, gridcolor='rgba(255,255,255,0.05)',
                               color='#6b8db5'),
                    yaxis=dict(color='#e8f4ff', tickfont=dict(size=11)),
                    height=240,
                    margin=dict(l=10, r=60, t=40, b=20)
                )
                st.plotly_chart(fig_top5, use_container_width=True, config={'displayModeBar': False})

# ─────────────────────────────────────────────
# PAGE: PERFORMANCE
# ─────────────────────────────────────────────
elif page == "📊 Performance":
    st.markdown('<div class="section-title">📈 Courbes d\'Entraînement</div>', unsafe_allow_html=True)

    if history:
        tab1, tab2 = st.tabs(["Accuracy", "Loss"])

        with tab1:
            # Clés compatibles : 'accuracy' (PyTorch notebook) ou 'acc' (fallback)
            acc_key     = 'accuracy'     if 'accuracy'     in history else 'acc'
            val_acc_key = 'val_accuracy' if 'val_accuracy' in history else 'val_acc'

            # Convertir en % si les valeurs sont en [0,1]
            acc_vals     = np.array(history[acc_key])
            if acc_vals.max() <= 1.0:
                acc_vals = acc_vals * 100

            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(
                y=acc_vals, name='Train',
                line=dict(color='#00f0ff', width=2.5),
                fill='tozeroy', fillcolor='rgba(0,240,255,0.05)'
            ))
            if val_acc_key in history:
                val_acc_vals = np.array(history[val_acc_key])
                if val_acc_vals.max() <= 1.0:
                    val_acc_vals = val_acc_vals * 100
                fig_acc.add_trace(go.Scatter(
                    y=val_acc_vals, name='Validation',
                    line=dict(color='#a0ff00', width=2.5, dash='dot'),
                    fill='tozeroy', fillcolor='rgba(160,255,0,0.04)'
                ))
            fig_acc.update_layout(
                title='Accuracy par Epoch',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(7,20,40,0.5)',
                font=dict(color='#6b8db5', family='Outfit'),
                xaxis=dict(title='Epoch', color='#6b8db5', gridcolor='rgba(255,255,255,0.04)'),
                yaxis=dict(title='Accuracy (%)', color='#6b8db5', gridcolor='rgba(255,255,255,0.04)'),
                legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#e8f4ff')),
                height=380
            )
            st.plotly_chart(fig_acc, use_container_width=True, config={'displayModeBar': False})

        with tab2:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                y=history['loss'], name='Train Loss',
                line=dict(color='#7b5ea7', width=2.5),
                fill='tozeroy', fillcolor='rgba(123,94,167,0.06)'
            ))
            if 'val_loss' in history:
                fig_loss.add_trace(go.Scatter(
                    y=history['val_loss'], name='Val Loss',
                    line=dict(color='#ff6b6b', width=2.5, dash='dot'),
                    fill='tozeroy', fillcolor='rgba(255,107,107,0.04)'
                ))
            fig_loss.update_layout(
                title='Loss par Epoch',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(7,20,40,0.5)',
                font=dict(color='#6b8db5', family='Outfit'),
                xaxis=dict(title='Epoch', color='#6b8db5', gridcolor='rgba(255,255,255,0.04)'),
                yaxis=dict(title='Loss', color='#6b8db5', gridcolor='rgba(255,255,255,0.04)'),
                legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#e8f4ff')),
                height=380
            )
            st.plotly_chart(fig_loss, use_container_width=True, config={'displayModeBar': False})

    else:
        st.info("Aucun historique d'entraînement trouvé. Placez `training_history.pkl` dans le dossier de l'app.")

    # Dataset stats
    if metadata and class_names:
        st.markdown('<div class="section-title" style="margin-top:28px;">📦 Statistiques Dataset</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")

        with col1:
            train_size = metadata.get('train_size', 0)
            val_size = metadata.get('val_size', 0)
            test_size = metadata.get('test_size', 0)
            total = train_size + val_size + test_size

            if total > 0:
                fig_split = go.Figure(go.Pie(
                    labels=['Train', 'Validation', 'Test'],
                    values=[train_size, val_size, test_size],
                    hole=0.6,
                    marker=dict(colors=['#00f0ff', '#7b5ea7', '#a0ff00'],
                                line=dict(color='#040d1a', width=3)),
                    textinfo='label+percent',
                    textfont=dict(color='#e8f4ff', size=12, family='Space Mono')
                ))
                fig_split.add_annotation(
                    text=f"<b>{total:,}</b><br>images",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(color='#00f0ff', size=14, family='Space Mono')
                )
                fig_split.update_layout(
                    title='Répartition Dataset',
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#6b8db5'), height=320,
                    showlegend=False,
                    margin=dict(t=40, b=20)
                )
                st.plotly_chart(fig_split, use_container_width=True, config={'displayModeBar': False})

        with col2:
            # Hyperparameters table
            st.markdown("""
            <div style='background: var(--bg-card); border: 1px solid var(--border);
                        border-radius:12px; padding:20px;'>
                <div style='font-family: Space Mono; font-size:0.72rem; color:#6b8db5;
                            text-transform:uppercase; letter-spacing:2px; margin-bottom:14px;'>
                    Hyperparamètres
                </div>
            """, unsafe_allow_html=True)

            params = [
                ("Batch Size", metadata.get('batch_size', 32)),
                ("Learning Rate", metadata.get('learning_rate', 0.001)),
                ("Epochs entraînés", metadata.get('epochs_trained', metadata.get('num_epochs', 20))),
                ("Meilleure Epoch", metadata.get('best_epoch', '—')),
                ("Optimiseur", "Adam"),
                ("Scheduler", "ReduceLROnPlateau"),
                ("Early Stopping", "patience=5"),
                ("Dropout", "0.5 / 0.3"),
            ]
            for label, value in params:
                st.markdown(f"""
                <div style='display:flex; justify-content:space-between; align-items:center;
                            padding:8px 0; border-bottom:1px solid rgba(0,240,255,0.07);
                            font-size:0.83rem;'>
                    <span style='color:#6b8db5;'>{label}</span>
                    <span style='color:#00f0ff; font-family:Space Mono;'>{value}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: ARCHITECTURE
# ─────────────────────────────────────────────
elif page == "🧠 Architecture":
    st.markdown('<div class="section-title">🧠 Architecture RSI_CNN</div>', unsafe_allow_html=True)

    col_arch, col_detail = st.columns([1, 1], gap="large")

    with col_arch:
        # Visual architecture diagram
        layers = [
            ("INPUT", "128×128×3", "#4a90d9", "Image satellite RGB"),
            ("CONV BLOC 1", "Conv2D(32) + BN + ReLU + MaxPool → 63×63×32", "#00f0ff", "32 filtres 3×3"),
            ("CONV BLOC 2", "Conv2D(64) + BN + ReLU + MaxPool → 30×30×64", "#00d4e0", "64 filtres 3×3"),
            ("CONV BLOC 3", "Conv2D(128) + BN + ReLU + MaxPool → 14×14×128", "#7b5ea7", "128 filtres 3×3"),
            ("CONV BLOC 4", "Conv2D(256) + BN + ReLU + MaxPool → 6×6×256", "#9b7ec7", "256 filtres 3×3"),
            ("FLATTEN", "→ 9216 neurones", "#888", "Aplatissement"),
            ("DENSE 512", "Linear(512) + ReLU + Dropout(0.5)", "#a0ff00", "Couche FC dense"),
            ("DENSE 256", "Linear(256) + ReLU + Dropout(0.3)", "#80cc00", "Couche FC dense"),
            ("OUTPUT", f"Linear({num_classes}) + Softmax", "#ff9f43", f"{num_classes} classes"),
        ]

        for layer_name, desc, color, note in layers:
            st.markdown(f"""
            <div class="arch-block" style="border-left-color:{color};">
                <div class="block-label" style="color:{color};">{layer_name}</div>
                <div style="color:#e8f4ff; font-size:0.82rem;">{desc}</div>
                <div style="color:#6b8db5; font-size:0.75rem; margin-top:2px;">{note}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_detail:
        st.markdown("""
        <div style='background: var(--bg-card); border: 1px solid var(--border);
                    border-radius:12px; padding:24px; margin-bottom:16px;'>
            <div style='font-family: Space Mono; font-size:0.72rem; color:#6b8db5;
                        text-transform:uppercase; letter-spacing:2px; margin-bottom:16px;'>
                Résumé Architecture
            </div>
        """, unsafe_allow_html=True)

        infos = [
            ("Type", "CNN Séquentiel"),
            ("Blocs conv.", "4 (32→64→128→256)"),
            ("Kernel size", "3×3, padding=0"),
            ("Batch Norm.", "Après chaque Conv"),
            ("Activation", "ReLU"),
            ("Pooling", "MaxPool 2×2"),
            ("FC 1", "512 neurones + Dropout(0.5)"),
            ("FC 2", "256 neurones + Dropout(0.3)"),
            ("Sortie", f"{num_classes} neurones (Softmax)"),
            ("Params totaux", f"{total_params:,}" if total_params else "—"),
        ]
        for k, v in infos:
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between;
                        padding:7px 0; border-bottom:1px solid rgba(0,240,255,0.06);
                        font-size:0.82rem;'>
                <span style='color:#6b8db5;'>{k}</span>
                <span style='color:#e8f4ff; font-family:Space Mono;'>{v}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Parameter distribution chart
        layers_data = {
            'Conv Bloc 1 (32)': 32*3*9 + 32,
            'Conv Bloc 2 (64)': 64*32*9 + 64,
            'Conv Bloc 3 (128)': 128*64*9 + 128,
            'Conv Bloc 4 (256)': 256*128*9 + 256,
            'FC 512': 9216*512 + 512,
            'FC 256': 512*256 + 256,
            f'Output ({num_classes})': 256*num_classes + num_classes,
        }
        fig_params = go.Figure(go.Bar(
            x=list(layers_data.keys()),
            y=list(layers_data.values()),
            marker=dict(
                color=list(layers_data.values()),
                colorscale=[[0, '#0b1e3d'], [0.5, '#7b5ea7'], [1.0, '#00f0ff']],
                showscale=False
            ),
            text=[f"{v/1e6:.2f}M" if v > 500000 else f"{v/1000:.1f}K" for v in layers_data.values()],
            textposition='outside',
            textfont=dict(color='#e8f4ff', size=9, family='Space Mono')
        ))
        fig_params.update_layout(
            title="Paramètres par Couche",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(7,20,40,0.5)',
            font=dict(color='#6b8db5', size=10),
            xaxis=dict(color='#6b8db5', tickangle=-25, gridcolor='rgba(255,255,255,0.03)'),
            yaxis=dict(color='#6b8db5', gridcolor='rgba(255,255,255,0.03)', type='log'),
            height=300,
            margin=dict(t=40, b=60)
        )
        st.plotly_chart(fig_params, use_container_width=True, config={'displayModeBar': False})

    # Classes list
    if class_names:
        st.markdown('<div class="section-title" style="margin-top:16px;">🏷️ Classes du Dataset</div>', unsafe_allow_html=True)
        cols = st.columns(5)
        for i, cls in enumerate(sorted(class_names)):
            with cols[i % 5]:
                st.markdown(f"""
                <div style='background: var(--bg-card); border: 1px solid var(--border);
                            border-radius: 8px; padding: 8px 12px; margin-bottom: 8px;
                            font-size: 0.75rem; color: #e8f4ff; font-family: Space Mono;
                            text-align:center;'>
                    <span style='color:#00f0ff;'>{i+1:02d}</span> {cls.replace('_', ' ')[:18]}
                </div>
                """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: À PROPOS
# ─────────────────────────────────────────────
elif page == "ℹ️ À propos":
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div style='background: var(--bg-card); border: 1px solid var(--border);
                    border-radius:14px; padding:28px;'>
            <div class="section-title">📖 Projet</div>
            <div style='font-size:0.88rem; color:#c5d8f0; line-height:1.9;'>
                Ce projet implémente un <strong style='color:#00f0ff'>CNN PyTorch</strong>
                pour la classification d'images satellite du dataset
                <strong style='color:#a0ff00'>RSI-CB128</strong>.
                <br/><br/>
                Le modèle est entraîné sur des images de <strong style='color:#00f0ff'>128×128 pixels</strong>
                en RGB, avec augmentation de données (flip, rotation, ColorJitter).
                <br/><br/>
                L'architecture s'inspire d'un modèle miroir TensorFlow avec 4 blocs
                convolutifs et des couches denses entièrement connectées.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='background: var(--bg-card); border: 1px solid var(--border);
                    border-radius:14px; padding:28px; margin-top:16px;'>
            <div class="section-title">🧰 Stack Technique</div>
        """, unsafe_allow_html=True)

        stack = [
            ("🔥", "PyTorch", "Framework Deep Learning"),
            ("📐", "torchvision", "Transforms & augmentation"),
            ("🎛️", "Streamlit", "Interface Web"),
            ("📊", "Plotly", "Visualisations interactives"),
            ("🔢", "NumPy / Pillow", "Traitement données & images"),
            ("📉", "Scikit-learn", "Métriques d'évaluation"),
        ]
        for icon, tech, desc in stack:
            st.markdown(f"""
            <div style='display:flex; align-items:center; gap:14px; padding:10px 0;
                        border-bottom:1px solid rgba(0,240,255,0.06);'>
                <span style='font-size:1.3rem;'>{icon}</span>
                <div>
                    <div style='color:#00f0ff; font-family:Space Mono; font-size:0.85rem;'>{tech}</div>
                    <div style='color:#6b8db5; font-size:0.77rem;'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background: var(--bg-card); border: 1px solid var(--border);
                    border-radius:14px; padding:28px;'>
            <div class="section-title">🏋️ Entraînement</div>
        """, unsafe_allow_html=True)

        train_infos = [
            ("Dataset", "RSI-CB128 (Remote Sensing Image Classification Benchmark)"),
            ("Répartition", "70% train / 20% val / 10% test"),
            ("Epochs max", "20 (EarlyStopping patience=5)"),
            ("Batch size", "32"),
            ("Optimiseur", "Adam (LR=0.001)"),
            ("Scheduler", "ReduceLROnPlateau (factor=0.5, patience=3)"),
            ("Augmentation", "HFlip, VFlip, Rotation±20°, ColorJitter"),
            ("Normalisation", "mean=0.5, std=0.5 → [-1, 1]"),
            ("Loss function", "CrossEntropyLoss"),
            ("Early Stop", "Restore best weights"),
        ]
        for k, v in train_infos:
            st.markdown(f"""
            <div style='display:flex; gap:12px; padding:8px 0;
                        border-bottom:1px solid rgba(0,240,255,0.06); font-size:0.82rem;'>
                <span style='color:#6b8db5; min-width:120px;'>{k}</span>
                <span style='color:#e8f4ff;'>{v}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style='background: var(--bg-card); border: 1px solid var(--border);
                    border-radius:14px; padding:28px; margin-top:16px;'>
            <div class="section-title">👤 Auteur</div>
            <div style='font-size:0.88rem; line-height:2; color:#c5d8f0;'>
                <div><strong style='color:#00f0ff;'>Nom :</strong> TSANGNING GRACE</div>
                <div><strong style='color:#00f0ff;'>Niveau :</strong> M2 Deep Learning</div>
                <div><strong style='color:#00f0ff;'>Année :</strong> 2025/2026</div>
                <div><strong style='color:#00f0ff;'>Module :</strong> Contrôle Continu Deep Learning</div>
                <div><strong style='color:#00f0ff;'>Framework :</strong> PyTorch</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Deployment guide
    st.markdown("""
    <div style='background: rgba(0,240,255,0.03); border: 1px solid rgba(0,240,255,0.15);
                border-radius:12px; padding:20px 24px; margin-top:20px;'>
        <div style='font-family: Space Mono; font-size:0.72rem; color:#6b8db5;
                    text-transform:uppercase; letter-spacing:2px; margin-bottom:12px;'>
            ⚡ Déploiement Streamlit Cloud
        </div>
        <div style='font-size:0.82rem; color:#c5d8f0; line-height:1.9;'>
            Placez ces fichiers dans votre repo GitHub :<br/>
            <code style='background:rgba(0,240,255,0.08); padding:1px 6px; border-radius:4px;
                          color:#a0ff00; font-family:Space Mono;'>app.py</code>&nbsp;
            <code style='background:rgba(0,240,255,0.08); padding:1px 6px; border-radius:4px;
                          color:#a0ff00; font-family:Space Mono;'>requirements.txt</code>&nbsp;
            <code style='background:rgba(0,240,255,0.08); padding:1px 6px; border-radius:4px;
                          color:#a0ff00; font-family:Space Mono;'>models/best_model.pth</code>&nbsp;
            <code style='background:rgba(0,240,255,0.08); padding:1px 6px; border-radius:4px;
                          color:#a0ff00; font-family:Space Mono;'>class_names.pkl</code>&nbsp;
            <code style='background:rgba(0,240,255,0.08); padding:1px 6px; border-radius:4px;
                          color:#a0ff00; font-family:Space Mono;'>training_history.pkl</code>&nbsp;
            <code style='background:rgba(0,240,255,0.08); padding:1px 6px; border-radius:4px;
                          color:#a0ff00; font-family:Space Mono;'>models/model_info.json</code>
        </div>
    </div>
    """, unsafe_allow_html=True)