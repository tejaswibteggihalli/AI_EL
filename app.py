import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import numpy as np
import os

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WasteSort AI",
    page_icon="♻️",
    layout="centered",
)

# ── Styling ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  h1, h2, h3 { font-family: 'Space Mono', monospace; }

  .main { background-color: #f5f5f0; }

  .hero {
    text-align: center;
    padding: 2rem 0 1rem;
  }
  .hero h1 {
    font-size: 2.4rem;
    color: #1a1a1a;
    letter-spacing: -1px;
  }
  .hero p {
    color: #555;
    font-size: 1.05rem;
    margin-top: -0.5rem;
  }

  .result-card {
    background: white;
    border-radius: 16px;
    padding: 1.8rem 2rem;
    border-left: 6px solid var(--accent);
    box-shadow: 0 4px 24px rgba(0,0,0,0.07);
    margin: 1.5rem 0;
  }
  .result-label {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #1a1a1a;
    text-transform: uppercase;
    letter-spacing: 2px;
  }
  .result-confidence {
    font-size: 1rem;
    color: #777;
    margin-top: 0.2rem;
  }

  .tip-box {
    background: #f0faf4;
    border: 1px solid #b2dfcc;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    font-size: 0.95rem;
    color: #2d6a4f;
  }
  .bar-wrap {
    margin-top: 1rem;
  }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

CLASS_META = {
    "cardboard": {
        "emoji": "📦",
        "color": "#c47c2b",
        "tip": "Flatten boxes before recycling. Remove any tape or staples if possible.",
        "bin": "Recycling Bin"
    },
    "glass": {
        "emoji": "🍾",
        "color": "#4a90d9",
        "tip": "Rinse bottles and jars. Remove lids — they're often a different material.",
        "bin": "Glass Recycling"
    },
    "metal": {
        "emoji": "🥫",
        "color": "#888888",
        "tip": "Rinse cans clean. Aluminium cans are infinitely recyclable — always recycle them!",
        "bin": "Recycling Bin"
    },
    "paper": {
        "emoji": "📄",
        "color": "#6aab6a",
        "tip": "Keep paper dry. Greasy or wet paper (like pizza boxes) goes to compost, not recycling.",
        "bin": "Paper Recycling"
    },
    "plastic": {
        "emoji": "🧴",
        "color": "#e06c75",
        "tip": "Check the resin code (♳–♷). Types 1 and 2 are most widely accepted. Rinse before recycling.",
        "bin": "Plastic Recycling"
    },
    "trash": {
        "emoji": "🗑️",
        "color": "#333333",
        "tip": "This item cannot be recycled. Dispose in general waste. Consider if it can be reused first.",
        "bin": "General Waste"
    },
}

MODEL_PATH  = "best_model.pth"   # change path if needed
DEVICE      = torch.device("cpu")   # use "cuda" if GPU available locally

# ── Model loader ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, len(CLASSES)),
    )
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

# ── Inference ────────────────────────────────────────────────────────────
val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def predict(image: Image.Image, model):
    tensor = val_tfms(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    top_idx  = int(np.argmax(probs))
    return CLASSES[top_idx], float(probs[top_idx]), probs

# ── UI ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>♻️ WasteSort AI</h1>
  <p>Upload a photo of waste — we'll tell you how to dispose of it correctly.</p>
</div>
""", unsafe_allow_html=True)

# Check model file exists
if not os.path.exists(MODEL_PATH):
    st.error(
        f"**Model file not found:** `{MODEL_PATH}`\n\n"
        "Place your `best_model.pth` in the same folder as `app.py` and rerun."
    )
    st.stop()

model = load_model()

# Upload widget
uploaded = st.file_uploader(
    "Drop an image here",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if uploaded:
    image = Image.open(uploaded)
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(image, caption="Your image", use_container_width=True)

    with col2:
        with st.spinner("Analysing…"):
            label, confidence, probs = predict(image, model)

        meta   = CLASS_META[label]
        accent = meta["color"]

        st.markdown(f"""
        <style>.result-card {{ --accent: {accent}; }}</style>
        <div class="result-card">
          <div style="font-size:2.5rem">{meta['emoji']}</div>
          <div class="result-label">{label}</div>
          <div class="result-confidence">
            {confidence*100:.1f}% confidence &nbsp;·&nbsp; 🗂 {meta['bin']}
          </div>
        </div>
        <div class="tip-box">💡 {meta['tip']}</div>
        """, unsafe_allow_html=True)

    # Confidence bar chart
    st.markdown("#### Confidence breakdown")
    bar_data = {c.capitalize(): float(p) for c, p in zip(CLASSES, probs)}
    st.bar_chart(bar_data, height=200)

else:
    st.info("👆 Upload a photo of waste above to get started.", icon="ℹ️")

st.markdown("---")
st.caption("Model: EfficientNet-B0 · Trained on Garbage Dataset Classification · 96.6% val accuracy")
