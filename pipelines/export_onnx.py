"""Export PyTorch models to ONNX for optimized inference."""
import torch
import os
import argparse
from src.utils.helpers import load_config
from src.models.custom_cnn import CustomCNN
from src.models.efficientnet import EfficientNetClassifier

def export_onnx(model_type="custom"):
    config = load_config()
    device = torch.device("cpu")
    
    if model_type == "custom":
        model = CustomCNN(config)
        path = config["paths"]["custom_model"]
        out_path = "models/saved/custom_cnn.onnx"
        input_size = (1, 3, 32, 32)
    else:
        model = EfficientNetClassifier(config)
        path = config["paths"]["efficientnet_model"]
        out_path = "models/saved/efficientnet.onnx"
        input_size = (1, 3, 224, 224)
        
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        print(f"Loaded {model_type} from {path}")
    else:
        print(f"Warning: No saved model at {path}, using untrained weights for export.")
        
    model.eval()
    dummy_input = torch.randn(input_size)
    
    torch.onnx.export(model, dummy_input, out_path, 
                      export_params=True, 
                      opset_version=12, 
                      do_constant_folding=True,
                      input_names=['input'], 
                      output_names=['output'],
                      dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})
    
    print(f"✅ Model successfully exported to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="custom", choices=["custom", "efficientnet"])
    args = parser.parse_args()
    export_onnx(args.model)
