"""PlantGuard AI — Elite Streamlit Dashboard."""
import os, sys, pickle, numpy as np, torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from src.utils.helpers import load_config, set_seed
from src.data.dataset import DatasetLoader
from src.models.custom_cnn import CustomCNN
from src.models.efficientnet import EfficientNetClassifier
from src.inference.predictor import Predictor
from src.explainability.gradcam import GradCAM

st.set_page_config(page_title="PlantGuard AI", page_icon="🌿", layout="wide")

# ── CSS ──
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
*{font-family:'Inter',sans-serif}
.hero{background:linear-gradient(135deg,#0d9488,#065f46);padding:2.5rem;border-radius:20px;text-align:center;margin-bottom:2rem}
.hero h1{color:white;font-size:2.8rem;font-weight:800;margin:0}
.hero p{color:#d1fae5;font-size:1.1rem;margin-top:0.5rem}
.card{background:linear-gradient(135deg,#1a1d29,#252836);border:1px solid #333;border-radius:16px;padding:1.5rem;text-align:center}
.card .val{font-size:2rem;font-weight:800;color:#10b981}.card .lbl{color:#888;font-size:0.8rem;text-transform:uppercase;letter-spacing:1px}
.pred-box{background:linear-gradient(135deg,#064e3b,#065f46);border-radius:16px;padding:2rem;margin:1rem 0}
.stTabs [data-baseweb="tab"]{background:#1a1d29;border-radius:10px;padding:10px 20px;border:1px solid #333;color:#ccc}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#0d9488,#065f46)!important;color:white!important;border:none!important}
</style>""", unsafe_allow_html=True)

# ── Resources ──
@st.cache_resource
def init():
    set_seed()
    config = load_config()
    device = torch.device("cpu")
    return config, device

config, device = init()
class_names = config["dataset"]["class_names"]
emojis = ["✈️","🚗","🐦","🐱","🦌","🐕","🐸","🐴","🚢","🚛"]

@st.cache_resource
def load_model(model_type):
    try:
        return Predictor(config, model_type)
    except Exception:
        return None

def quick_train(model_type):
    from src.training.trainer import Trainer
    loader = DatasetLoader(config)
    trainer = Trainer(config, device)
    if model_type == "custom":
        model = CustomCNN(config)
        tl, vl, _ = loader.get_loaders(transfer=False, subset_size=300)
        path = config["paths"]["custom_model"]
    else:
        model = EfficientNetClassifier(config)
        tl, vl, _ = loader.get_loaders(transfer=True, subset_size=300)
        path = config["paths"]["efficientnet_model"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    trainer.train(model, tl, vl, path, model_type, epochs=2)

# ── Sidebar ──
with st.sidebar:
    st.markdown("## 🌿 PlantGuard AI")
    st.markdown("---")
    page = st.radio("Navigate", ["🏠 Home","🔍 Predict","📊 Performance","🔬 Grad-CAM","📈 History"], label_visibility="collapsed")
    st.markdown("---")
    model_choice = st.selectbox("Model", ["Custom CNN", "EfficientNet-B0"])
    model_type = "custom" if "Custom" in model_choice else "efficientnet"
    st.markdown("---")
    st.caption("v2.0 • PyTorch • Production")

# Model existence check
path = config["paths"]["custom_model"] if model_type == "custom" else config["paths"]["efficientnet_model"]
model_exists = os.path.exists(path)

if not model_exists:
    st.warning(f"⚠️ {model_type.title()} model not found.")
    if st.button(f"🚀 Initialize & Train {model_type.title()} (Quick Demo)"):
        with st.spinner(f"🔄 Training {model_type} model (~2 min)..."):
            quick_train(model_type)
        st.success("✅ Training complete!")
        st.rerun()
    st.info("Note: For production, pre-train models locally and upload to the `models/saved/` directory.")

predictor = load_model(model_type) if model_exists else None

def call_api_predict(image_np, model_type):
    api_url = os.getenv("API_URL", "http://localhost:8000")
    try:
        # Convert numpy array to bytes
        img_pil = Image.fromarray(image_np)
        buf = io.BytesIO()
        img_pil.save(buf, format="JPEG")
        files = {"file": ("image.jpg", buf.getvalue(), "image/jpeg")}
        params = {"model": model_type}
        response = requests.post(f"{api_url}/predict", files=files, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

# Check API health
api_url = os.getenv("API_URL", "http://localhost:8000")
api_healthy = False
try:
    health_res = requests.get(f"{api_url}/health", timeout=2)
    if health_res.status_code == 200:
        api_healthy = True
except:
    pass

# ── HOME ──
if "Home" in page:
    st.markdown('<div class="hero"><h1>🌿 PlantGuard AI</h1><p>Production-grade Deep Learning Image Classification System</p></div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    for col, val, lbl in [(c1,"10","Classes"),(c2,"60K","Images"),(c3,"2","Models"),(c4,"Grad-CAM","Explainability")]:
        col.markdown(f'<div class="card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)
    st.markdown("### 🏗️ System Architecture")
    st.code("""
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │  Streamlit   │────▶│   FastAPI     │────▶│  PyTorch     │
    │  Frontend    │     │   Backend     │     │  Models      │
    └──────────────┘     └──────────────┘     └──────────────┘
          │                    │                     │
          ▼                    ▼                     ▼
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │  Grad-CAM    │     │  /predict    │     │  CustomCNN   │
    │  Heatmaps    │     │  /health     │     │  EfficientNet│
    └──────────────┘     └──────────────┘     └──────────────┘
    """, language=None)
    st.markdown("### 🎯 Supported Classes")
    cols = st.columns(5)
    for i, (n, e) in enumerate(zip(class_names, emojis)):
        cols[i % 5].markdown(f"**{e} {n.title()}**")

# ── PREDICT ──
elif "Predict" in page:
    st.markdown("## 🔍 Real-Time Classification")
    t1, t2 = st.tabs(["📤 Upload Image", "🎲 Random Sample"])
    with t1:
        up = st.file_uploader("Upload (JPG/PNG)", type=["jpg","jpeg","png"])
        if up and predictor:
            img = Image.open(up).convert("RGB")
            img_np = np.array(img)
            c1, c2 = st.columns(2)
            c1.image(img, caption="Uploaded", use_container_width=True)
            
            # Predict via API
            if api_healthy:
                with st.spinner("Connecting to FastAPI backend..."):
                    res = call_api_predict(img_np, model_type)
            elif predictor:
                st.warning("FastAPI backend is offline. Falling back to local inference.")
                res = predictor.predict(img_np)
            else:
                st.error("No model or API available for prediction.")
                res = None
                with c2:
                    idx = class_names.index(res["prediction"]) if res["prediction"] in class_names else 0
                    st.markdown(f'<div class="pred-box"><h2 style="color:white">{emojis[idx]} {res["prediction"].upper()}</h2><h3 style="color:#6ee7b7">{res["confidence"]:.1%} confidence</h3></div>', unsafe_allow_html=True)
                    for r in res["top_k"]:
                        st.progress(r["confidence"], text=f"{r['class'].title()} — {r['confidence']:.1%}")
    with t2:
        if st.button("🎲 Random Test Image", use_container_width=True) and predictor:
            from torchvision import datasets
            ds = datasets.CIFAR10(config["paths"]["data_dir"], train=False, download=True)
            idx = np.random.randint(len(ds))
            img, lbl = ds[idx]
            img_np = np.array(img)
            c1, c2 = st.columns(2)
            c1.image(img_np, caption=f"True: {class_names[lbl]}", use_container_width=True)
            res = predictor.predict(img_np)
            ok = res["prediction"] == class_names[lbl]
            with c2:
                st.markdown(f"### {'✅ Correct!' if ok else '❌ Wrong'}")
                st.markdown(f"**Predicted:** {res['prediction'].upper()} ({res['confidence']:.1%})")
                st.markdown(f"**True:** {class_names[lbl].upper()}")
                for r in res["top_k"]:
                    st.progress(r["confidence"], text=f"{r['class'].title()} — {r['confidence']:.1%}")

# ── PERFORMANCE ──
elif "Performance" in page:
    st.markdown("## 📊 Model Performance")
    comp_path = config["paths"]["comparison"]
    if os.path.exists(comp_path):
        with open(comp_path, "rb") as f:
            results = pickle.load(f)
        for name, res in results.items():
            st.markdown(f"### {name}")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Accuracy", f"{res['accuracy']:.4f}")
            c2.metric("Precision", f"{res['precision']:.4f}")
            c3.metric("Recall", f"{res['recall']:.4f}")
            c4.metric("F1-Score", f"{res['f1_score']:.4f}")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(res["confusion_matrix"], annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues", ax=ax)
            ax.set_title(f"{name} — Confusion Matrix"); ax.set_ylabel("True"); ax.set_xlabel("Predicted")
            st.pyplot(fig); plt.close()
    else:
        st.info("Run `python pipelines/training_pipeline.py` to generate performance data.")

# ── GRAD-CAM ──
elif "Grad-CAM" in page:
    st.markdown("## 🔬 Model Explainability — Grad-CAM")
    if predictor:
        if st.button("🔬 Generate Grad-CAM on Random Image", use_container_width=True):
            from torchvision import datasets
            ds = datasets.CIFAR10(config["paths"]["data_dir"], train=False, download=True)
            idx = np.random.randint(len(ds))
            img_pil, lbl = ds[idx]
            img_np = np.array(img_pil)
            model = predictor.get_model()
            gcam = GradCAM(model)
            transfer = model_type == "efficientnet"
            loader = DatasetLoader(config)
            tensor = loader.preprocess_image(img_np, transfer=transfer).to(device)
            heatmap = gcam.generate(tensor)
            overlay = gcam.overlay(heatmap, img_np)
            res = predictor.predict(img_np)
            c1,c2,c3 = st.columns(3)
            c1.image(img_np, caption=f"True: {class_names[lbl]}", use_container_width=True)
            fig2, ax2 = plt.subplots(figsize=(4,4))
            import cv2
            hm_r = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
            ax2.imshow(hm_r, cmap="jet"); ax2.axis("off"); ax2.set_title("Heatmap")
            c2.pyplot(fig2); plt.close()
            c3.image(overlay, caption=f"Pred: {res['prediction']} ({res['confidence']:.1%})", use_container_width=True, clamp=True)
            
            st.markdown("""
            ### 💡 How to read this?
            **Grad-CAM (Gradient-weighted Class Activation Mapping)** uses the gradients of the target class flowing into the final convolutional layer to produce a localization map highlighting the important regions in the image for predicting the concept.
            - **Red regions**: High influence on the model's decision.
            - **Blue regions**: Low/No influence.
            - **Insight**: In plant pathology, the model should focus on leaf lesions, spots, or discoloration rather than the background.
            """)

# ── HISTORY ──
elif "History" in page:
    st.markdown("## 📈 Training History")
    comp_path = config["paths"]["comparison"]
    if os.path.exists(comp_path):
        with open(comp_path, "rb") as f:
            results = pickle.load(f)
        for name, res in results.items():
            if "history" in res:
                h = res["history"]
                fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
                a1.plot(h["loss"], label="Train", color="#ef4444"); a1.plot(h["val_loss"], label="Val", color="#10b981")
                a1.set_title(f"{name} — Loss"); a1.legend(); a1.grid(alpha=0.2)
                a2.plot(h["accuracy"], label="Train", color="#ef4444"); a2.plot(h["val_accuracy"], label="Val", color="#10b981")
                a2.set_title(f"{name} — Accuracy"); a2.legend(); a2.grid(alpha=0.2)
                st.pyplot(fig); plt.close()
    else:
        st.info("No training history. Run the training pipeline first.")

st.markdown("---")
st.markdown('<p style="text-align:center;color:#555;font-size:0.8rem">🌿 PlantGuard AI v2.0 — PyTorch + Streamlit + FastAPI</p>', unsafe_allow_html=True)
