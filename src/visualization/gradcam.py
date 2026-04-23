"""
Grad-CAM Visualization Module (PyTorch Version)
==============================================
Gradient-weighted Class Activation Mapping for PyTorch.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.cm as cm
import logging

logger = logging.getLogger(__name__)


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer or self._find_target_layer()
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _find_target_layer(self):
        # Look for last Conv2d layer
        target = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                target = module
        if target is None:
            raise ValueError("No Conv2d layer found.")
        return target

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def compute_heatmap(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        gradients = self.gradients
        activations = self.activations
        
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        for i in range(activations.size(1)):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)
        
        return heatmap.detach().cpu().numpy()

    def overlay_heatmap(self, heatmap, original_image, alpha=0.4):
        h, w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
        
        if original_image.max() > 1.0:
            original_image = original_image / 255.0
            
        overlaid = (1 - alpha) * original_image + alpha * heatmap_colored
        return np.clip(overlaid, 0, 1)
