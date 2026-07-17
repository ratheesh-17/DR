import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.activations = None
        self.gradients   = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = F.relu((weights * self.activations).sum(dim=1).squeeze())
        if cam.max() > 0:
            cam = cam / cam.max()
        probs = output.softmax(dim=1).cpu().detach().numpy()[0]
        return cam.cpu().numpy(), class_idx, probs


class GradCAMPP:
    def __init__(self, model, target_layer):
        self.model       = model
        self.activations = None
        self.gradients   = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        output[0, class_idx].backward()
        grads     = self.gradients
        acts      = self.activations
        grads_sq  = grads ** 2
        grads_cub = grads ** 3
        denom     = 2 * grads_sq + acts * grads_cub
        denom     = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha     = grads_sq / denom
        relu_grads = F.relu(output[0, class_idx].exp() * grads)
        weights    = (alpha * relu_grads).sum(dim=[2, 3], keepdim=True)
        cam        = F.relu((weights * acts).sum(dim=1).squeeze())
        if cam.max() > 0:
            cam = cam / cam.max()
        probs = output.softmax(dim=1).cpu().detach().numpy()[0]
        return cam.cpu().numpy(), class_idx, probs


def overlay_heatmap(img_np: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
    heatmap     = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return (alpha * heatmap_rgb + (1 - alpha) * img_np).astype(np.uint8)
