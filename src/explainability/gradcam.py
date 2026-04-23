"""Grad-CAM explainability for PyTorch CNN models."""
import torch, torch.nn.functional as F, numpy as np, cv2
import matplotlib.cm as cm

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self._target = self._find_last_conv()
        self._target.register_forward_hook(self._fwd_hook)
        self._target.register_full_backward_hook(self._bwd_hook)

    def _find_last_conv(self):
        target = None
        for m in self.model.modules():
            if isinstance(m, torch.nn.Conv2d):
                target = m
        if target is None:
            raise ValueError("No Conv2d layer found")
        return target

    def _fwd_hook(self, mod, inp, out):
        self.activations = out.detach()

    def _bwd_hook(self, mod, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        heatmap = (weights * self.activations).sum(dim=1, keepdim=True)
        heatmap = F.relu(heatmap).squeeze()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        return heatmap.cpu().numpy()

    def overlay(self, heatmap, original_image, alpha=0.4):
        h, w = original_image.shape[:2]
        hm = cv2.resize(heatmap, (w, h))
        colored = cm.jet(hm)[:, :, :3]
        img = original_image / 255.0 if original_image.max() > 1 else original_image.astype(np.float64)
        return np.clip((1 - alpha) * img + alpha * colored, 0, 1)
