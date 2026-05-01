#  **🌿[PlantGuard AI →](https://cnn-image-classification-deployment-yldn7fmageuksfr7cdia3c.streamlit.app/)**


<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi" />
  <img src="https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/MLflow-2.0+-0194E2?style=for-the-badge&logo=mlflow" />
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker" />
  <img src="https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions" />
</p>

<p align="center">
  <h1 align="center">**🌿[PlantGuard AI →](https://cnn-image-classification-deployment-yldn7fmageuksfr7cdia3c.streamlit.app/)**</h1>
  <p align="center">
    <strong>Production-Grade Deep Learning Image Classification System</strong><br>
    An end-to-end machine learning pipeline for detecting fraudulent financial transactions using advanced imbalanced data handling techniques.
  </p>


<p align="center">
An end-to-end, FAANG-level computer vision system featuring <b>Custom CNN vs EfficientNet</b> model comparison, <b>Grad-CAM explainability</b>, <b>FastAPI microservice</b>, <b>MLflow experiment tracking</b>, and <b>interactive Streamlit dashboard</b> — fully containerized with CI/CD.
</p>


---

## 🎯 Problem Statement

Image classification is a foundational computer vision task with applications across agriculture, healthcare, manufacturing, and environmental monitoring. This project demonstrates a **production-grade** approach to building, evaluating, deploying, and explaining deep learning image classifiers — going far beyond typical academic demos.

The system is designed as a **domain-agnostic classification platform** currently configured for general image recognition (CIFAR-10), with architecture ready for plant disease detection, medical imaging, or industrial defect detection.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PlantGuard AI                            │
├──────────────┬──────────────┬──────────────┬───────────────────┤
│  Streamlit   │   FastAPI    │   PyTorch    │   Grad-CAM        │
│  Dashboard   │   Backend    │   Models     │   Explainability  │
│  (app.py)    │  (api/)      │  (src/)      │  (src/explain/)   │
├──────────────┴──────────────┴──────────────┴───────────────────┤
│                    Training Pipeline                            │
│              (pipelines/training_pipeline.py)                   │
├────────────────────────────────────────────────────────────────┤
│  Docker  │  GitHub Actions CI/CD  │  Config-driven (YAML)      │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
cnn-image-classification-deployment/
├── .github/workflows/ci.yml    # CI/CD pipeline
├── api/
│   └── main.py                 # FastAPI — /predict, /health
├── app.py                      # Streamlit dashboard
├── config/
│   └── config.yaml             # Centralized configuration
├── pipelines/
│   ├── training_pipeline.py    # End-to-end training orchestration
│   └── inference_pipeline.py   # CLI inference
├── src/
│   ├── data/dataset.py         # DataLoader with augmentation
│   ├── models/
│   │   ├── custom_cnn.py       # Residual CNN from scratch
│   │   └── efficientnet.py     # EfficientNet-B0 transfer learning
│   ├── training/trainer.py     # Training engine + evaluation
│   ├── inference/predictor.py  # Production inference engine
│   ├── explainability/
│   │   └── gradcam.py          # Grad-CAM heatmap generation
│   └── utils/helpers.py        # Config, logging, seeding
├── tests/test_core.py          # Unit tests
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Multi-service orchestration
└── requirements.txt
```

---

## 🔬 Models & Approach

### Custom CNN (Residual Architecture)
- 4-block residual network with skip connections
- BatchNorm + Dropout regularization
- Global Average Pooling → Dense head
- ~2.8M parameters

### EfficientNet-B0 (Transfer Learning)
- ImageNet-pretrained backbone
- Frozen feature extractor → fine-tuned top layers
- Cosine annealing LR schedule
- ~5.3M parameters (0.4M trainable)

### Training Features
- **Data Augmentation**: Random flip, rotation, color jitter, random erasing
- **Cosine Annealing** learning rate schedule
- **Early Stopping** with patience tracking
- **Gradient Clipping** for training stability
- **MLflow Integration**: Tracking parameters, metrics (accuracy, loss), and model versioning
- **Model Comparison** pipeline with serialized results

---

## 📊 Evaluation Metrics

| Metric | Custom CNN | EfficientNet-B0 |
|--------|-----------|-----------------|
| Accuracy | ~78% | ~88% |
| Precision | ~78% | ~88% |
| Recall | ~78% | ~88% |
| F1-Score | ~78% | ~88% |

*Results from full training (30 epochs). Quick-train demo uses fewer epochs.*

Additional analysis includes:
- ✅ Confusion Matrix visualization
- ✅ Per-class accuracy breakdown
- ✅ Confidence distribution analysis
- ✅ Training history (loss/accuracy curves)

---

## 🔥 Grad-CAM Explainability

The system includes **Gradient-weighted Class Activation Mapping** to visualize which image regions drive the model's decisions:

- Hook-based gradient capture on last Conv2D layer
- ReLU-activated, normalized heatmap generation
- Jet-colormap overlay on original images
- Interactive exploration via Streamlit

---

## 🚀 Quick Start

### Local Development
```bash
git clone https://github.com/mohitrj18greybeard/cnn-image-classification-deployment.git
cd cnn-image-classification-deployment
pip install -r requirements.txt

# Train models
python pipelines/training_pipeline.py --model both --quick

# Launch dashboard
streamlit run app.py

# Start API server (separate terminal)
uvicorn api.main:app --reload
```

### Docker
```bash
docker-compose up --build
# API: http://localhost:8000
# UI:  http://localhost:8501
```

---

## 🔌 API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Predict
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg" \
  -F "model=efficientnet"
```

### Batch Predict
```bash
curl -X POST http://localhost:8000/predict_batch \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "model=custom"
```

Response:
```json
{
  "prediction": "cat",
  "confidence": 0.92,
  "top_k": [
    {"class": "cat", "confidence": 0.92},
    {"class": "dog", "confidence": 0.05}
  ]
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | PyTorch, torchvision |
| Models | Custom ResNet CNN, EfficientNet-B0 |
| Backend API | FastAPI, Uvicorn |
| MLOps | MLflow |
| Frontend | Streamlit |
| Explainability | Grad-CAM |
| Data Science | NumPy, Pandas, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Config | YAML-driven pipeline |

---

## 🔮 Future Improvements

- [ ] ONNX model export for optimized inference
- [ ] MLflow experiment tracking integration
- [ ] Kubernetes deployment manifests
- [ ] Real-time webcam classification
- [ ] PlantVillage dataset integration for plant disease detection
- [ ] Model quantization (INT8) for edge deployment
- [ ] A/B testing framework for model comparison
- [ ] Prometheus + Grafana monitoring

---


## 👤 Author

**Mohit**

- GitHub: [@mohitrj18greybeard](https://github.com/mohitrj18greybeard)



---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>⭐ If you found this project useful, please give it a star!</strong><br/>
  <em>Built with ❤️ </em>
</
