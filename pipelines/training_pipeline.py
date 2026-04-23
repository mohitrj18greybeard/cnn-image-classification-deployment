"""Training pipeline — orchestrates data loading, model training, evaluation, and comparison."""
import os, sys, pickle, argparse, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import load_config, setup_logging, set_seed, get_device
from src.data.dataset import DatasetLoader
from src.models.custom_cnn import CustomCNN
from src.models.efficientnet import EfficientNetClassifier
from src.training.trainer import Trainer

log = logging.getLogger(__name__)

def run(args):
    setup_logging(); set_seed()
    config = load_config()
    device = get_device()
    loader = DatasetLoader(config)
    trainer = Trainer(config, device)
    class_names = loader.get_class_names()
    subset = 500 if args.quick else None
    results = {}

    if args.model in ("custom", "both"):
        log.info("═" * 50 + " CUSTOM CNN " + "═" * 50)
        model = CustomCNN(config)
        train_l, val_l, test_l = loader.get_loaders(transfer=False, subset_size=subset)
        hist = trainer.train(model, train_l, val_l, config["paths"]["custom_model"],
                             "CustomCNN", epochs=3 if args.quick else None)
        model.load_state_dict(__import__("torch").load(config["paths"]["custom_model"], map_location=device, weights_only=True))
        ev = trainer.evaluate(model, test_l, class_names)
        results["Custom CNN"] = {**ev, "history": hist["history"], "training_time": hist["training_time"]}
        log.info(f"Custom CNN — Acc: {ev['accuracy']:.4f}  F1: {ev['f1_score']:.4f}")

    if args.model in ("efficientnet", "both"):
        log.info("═" * 50 + " EFFICIENTNET " + "═" * 50)
        model = EfficientNetClassifier(config)
        train_l, val_l, test_l = loader.get_loaders(transfer=True, subset_size=subset)
        hist = trainer.train(model, train_l, val_l, config["paths"]["efficientnet_model"],
                             "EfficientNet", epochs=3 if args.quick else None)
        model.load_state_dict(__import__("torch").load(config["paths"]["efficientnet_model"], map_location=device, weights_only=True))
        ev = trainer.evaluate(model, test_l, class_names)
        results["EfficientNet"] = {**ev, "history": hist["history"], "training_time": hist["training_time"]}
        log.info(f"EfficientNet — Acc: {ev['accuracy']:.4f}  F1: {ev['f1_score']:.4f}")

    os.makedirs("models/saved", exist_ok=True)
    with open(config["paths"]["comparison"], "wb") as f:
        pickle.dump(results, f)
    log.info("✅ Training pipeline complete. Run: streamlit run app.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="both", choices=["custom", "efficientnet", "both"])
    parser.add_argument("--quick", action="store_true")
    run(parser.parse_args())
