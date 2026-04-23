"""
DeepVision CNN — Streamlit Dashboard (PyTorch Version)
=====================================================
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import pickle
import matplotlib.pyplot as plt
from src.utils.helpers import load_config
from src.data.dataset import DatasetLoader
from src.models.custom_cnn import CustomCNN
from src.models.transfer_learning import TransferLearningModel
from src.evaluation.evaluator import ModelEvaluator
from src.visualization.gradcam import GradCAM

st.set_page_config(page_title="DeepVision PyTorch", layout="wide")

@st.cache_resource
def get_resources():
    config = load_config()
    loader = DatasetLoader(config)
    class_names = loader.get_class_names()
    evaluator = ModelEvaluator(class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return config, loader, class_names, evaluator, device

config, loader, class_names, evaluator, device = get_resources()

def load_model(model_type):
    if model_type == "custom":
        model = CustomCNN(config)
        path = config["paths"]["custom_model"]
    else:
        model = TransferLearningModel(config)
        path = config["paths"]["transfer_model"]
    
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

st.sidebar.title("DeepVision 🧠")
model_choice = st.sidebar.selectbox("Model", ["Custom CNN", "Transfer Learning"])
model_type = "custom" if "Custom" in model_choice else "transfer"
model = load_model(model_type)

st.title("Elite CNN Image Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Preprocess
    img_np = np.array(image)
    input_tensor = loader.preprocess_image(img_np, for_transfer=(model_type == "transfer")).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        conf, pred = torch.max(probs, 0)
        
    st.write(f"### Prediction: **{class_names[pred.item()]}** ({conf.item():.2%})")
    
    # Grad-CAM
    if st.button("Generate Grad-CAM"):
        gcam = GradCAM(model)
        heatmap = gcam.compute_heatmap(input_tensor, pred.item())
        overlay = gcam.overlay_heatmap(heatmap, img_np)
        st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)
