"""Inference engine — loads model, preprocesses, and predicts."""
import torch, logging, os, numpy as np
from src.data.dataset import DatasetLoader

log = logging.getLogger(__name__)

class Predictor:
    def __init__(self, config, model_type="custom"):
        self.config = config
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loader = DatasetLoader(config)
        self.class_names = self.loader.get_class_names()
        self.model = self._load_model()

    def _load_model(self):
        from src.models.custom_cnn import CustomCNN
        from src.models.efficientnet import EfficientNetClassifier
        if self.model_type == "efficientnet":
            model = EfficientNetClassifier(self.config)
            path = self.config["paths"]["efficientnet_model"]
        else:
            model = CustomCNN(self.config)
            path = self.config["paths"]["custom_model"]
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            log.info(f"Loaded {self.model_type} from {path}")
        else:
            log.warning(f"No saved model at {path}, using untrained weights")
        model.to(self.device); model.eval()
        return model

    def predict(self, image_np):
        transfer = self.model_type == "efficientnet"
        tensor = self.loader.preprocess_image(image_np, transfer=transfer).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, 1)[0].cpu().numpy()
        top_k = int(self.config["inference"]["top_k"])
        top_idx = np.argsort(probs)[::-1][:top_k]
        results = [{"class": self.class_names[i], "confidence": float(probs[i])} for i in top_idx]
        return {"prediction": results[0]["class"], "confidence": results[0]["confidence"],
                "top_k": results, "all_probabilities": probs.tolist()}

    def get_model(self):
        return self.model
