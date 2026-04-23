"""Tests for API and core components."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_config_loads():
    from src.utils.helpers import load_config
    cfg = load_config()
    assert "dataset" in cfg
    assert "training" in cfg
    assert cfg["dataset"]["num_classes"] == 10

def test_custom_cnn_forward():
    import torch
    from src.utils.helpers import load_config
    from src.models.custom_cnn import CustomCNN
    cfg = load_config()
    model = CustomCNN(cfg)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)

def test_efficientnet_forward():
    import torch
    from src.utils.helpers import load_config
    from src.models.efficientnet import EfficientNetClassifier
    cfg = load_config()
    model = EfficientNetClassifier(cfg)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 10)

def test_api_health():
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"
