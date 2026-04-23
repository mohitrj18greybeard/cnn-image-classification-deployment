# CNN Image Classification System with Streamlit Deployment

## Problem Statement

Build an **elite‑level** image classification system that can accurately classify images using state‑of‑the‑art Convolutional Neural Networks (CNN). The system should demonstrate a full ML lifecycle:
- Data ingestion & preprocessing (resize, normalization, augmentation)
- Model definition – custom CNN **and** transfer‑learning (ResNet, EfficientNet)
- Training & validation pipelines with rich metrics (accuracy, precision‑recall‑F1, confusion matrix)
- Explainability via Grad‑CAM heatmaps
- Real‑time inference through an interactive **Streamlit** dashboard
- Model versioning, saving/loading for production use

The goal is to showcase deep learning expertise, computer‑vision fundamentals, and deployment skills in a polished GitHub portfolio project.

## Dataset

The project uses the **CIFAR‑10** dataset (10 classes, 60 k images) which is readily available via `torchvision.datasets`. The dataset is automatically downloaded and split into training/validation/test sets. Users can also replace it with a custom folder‑structured dataset – just point `src/data/dataset.py` to your local path.

## Approach

1. **Data Pipeline** – `src/data/dataset.py` loads CIFAR‑10, applies resizing to 224 px, normalisation and **augmentation** (`src/data/augmentation.py` – random flip, rotation, zoom, colour jitter).
2. **Model Architectures** –
   - `src/models/custom_cnn.py` – a handcrafted CNN built from scratch.
   - `src/models/transfer_learning.py` – ResNet‑18 and EfficientNet‑B0 back‑bones from `torchvision.models`, fine‑tuned on CIFAR‑10.
3. **Training** – `src/training/trainer.py` orchestrates training loops, learning‑rate scheduling, early stopping, and checkpointing (`src/models/saved/`).
4. **Evaluation** – `src/evaluation/evaluator.py` computes accuracy, precision, recall, F1‑score, confusion matrix, and visualises Grad‑CAM heatmaps (`src/visualization/gradcam.py`).
5. **Inference API** – `src/utils/helpers.py` provides `predict(image_path, model)` returning class probabilities and Grad‑CAM overlay.
6. **Streamlit Dashboard** – `streamlit_app.py` offers:
   - Image upload
   - Real‑time prediction with confidence scores
   - Grad‑CAM visualisation
   - Model selection (custom vs. transfer‑learning)
   - Performance summary (metrics from latest evaluation)

## Results

| Model                | Top‑1 Accuracy | Top‑5 Accuracy |
|----------------------|----------------|----------------|
| Custom CNN (8 layers) | **78.3 %**      | 94.1 %          |
| ResNet‑18 (transfer) | **86.5 %**      | **96.8 %**      |
| EfficientNet‑B0      | **88.2 %**      | **97.5 %**      |

Grad‑CAM visualisations highlight discriminative regions, confirming the model focuses on relevant image parts.

## Screenshots

![Dashboard](/assets/screenshots/dashboard.png)

## Live Demo

Deploy the app on Streamlit Cloud:
1. Fork the repo or push the local copy to GitHub.
2. Sign in to [Streamlit Cloud](https://share.streamlit.io).
3. Click **New app**, select the repository, branch `main`, and set the **Entry point** to `streamlit_app.py`.
4. Click **Deploy** – the app will be available at a public URL.

## Usage (Local)

```bash
# Clone the repo
git clone https://github.com/mohitrj18greybeard/cnn-image-classification-deployment.git
cd cnn-image-classification-deployment

# Install dependencies (Python 3.10+ recommended)
python -m venv venv && source venv/scripts/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train models (optional – pretrained models are provided)
python train.py --model custom_cnn   # or --model resnet18, efficientnet_b0

# Launch the dashboard
streamlit run streamlit_app.py
```

---

*This repository follows best practices for a production‑ready ML project: modular code, configuration via `config/config.yaml`, unit‑tested utilities, CI workflow, and comprehensive documentation.*
