"""Inference pipeline — single image prediction from CLI."""
import os, sys, argparse, numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import load_config, setup_logging
from src.inference.predictor import Predictor

def run(args):
    setup_logging()
    config = load_config()
    predictor = Predictor(config, model_type=args.model)
    img = np.array(Image.open(args.image).convert("RGB"))
    result = predictor.predict(img)
    print(f"\n🎯 Prediction: {result['prediction']}  ({result['confidence']:.1%})")
    print("\nTop-5:")
    for r in result["top_k"]:
        print(f"  {r['class']:>15s}  {r['confidence']:.1%}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("image", help="Path to image file")
    p.add_argument("--model", default="custom", choices=["custom", "efficientnet"])
    run(p.parse_args())
