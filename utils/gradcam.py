"""
Grad-CAM visualiser — highlights which facial regions most influenced
the CNN / ViT emotion prediction.

For ViT models, we use a variant called Attention Rollout which
aggregates attention maps across transformer layers.

Reference:
  Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep Networks"
  Abnar & Zuidema (2020) "Quantifying Attention Flow in Transformers"
"""

from __future__ import annotations
import numpy as np
from PIL import Image
import cv2
from typing import Optional


class GradCAMVisualiser:
    """
    Generates a Grad-CAM heatmap overlay for a given image and model.

    Works with both CNN and ViT architectures:
      - CNN: hooks the last convolutional layer.
      - ViT: uses Attention Rollout over all attention heads.
    """

    def __init__(self, model, target_layer_name: Optional[str] = None):
        """
        Args:
            model:             The PyTorch model (CNN or ViT).
            target_layer_name: For CNNs — name of the target conv layer.
                               If None, auto-detects the last conv layer.
        """
        self.model            = model
        self.target_layer_name = target_layer_name
        self._gradients: Optional[np.ndarray] = None
        self._activations: Optional[np.ndarray] = None

    def generate(self, image: Image.Image,
                 target_class: Optional[int] = None) -> Image.Image:
        """
        Generate a Grad-CAM heatmap overlaid on the original image.

        Args:
            image:        Input PIL image.
            target_class: Class index to explain. If None, uses the predicted class.

        Returns:
            PIL Image — original photo with colour heatmap overlay.
        """
        model_type = self._detect_model_type()

        if model_type == "vit":
            heatmap = self._attention_rollout(image)
        else:
            heatmap = self._gradcam_cnn(image, target_class)

        return self._overlay_heatmap(image, heatmap)

    # ── ViT: Attention Rollout ─────────────────────────────────────────────
    def _attention_rollout(self, image: Image.Image) -> np.ndarray:
        """
        Attention Rollout using PyTorch forward hooks on the attention
        dropout layers.

        We hook nn.Dropout modules that sit inside each ViTSelfAttention
        block. At that point in the forward pass the tensor flowing through
        is the (batch, heads, seq, seq) attention weight matrix — exactly
        what we need — and it is always present regardless of the
        output_attentions config flag.
        """
        import torch
        import torch.nn as nn
        from transformers import AutoImageProcessor

        # ── preprocess ────────────────────────────────────────────────────
        model_name = self.model.config._name_or_path
        try:
            processor = AutoImageProcessor.from_pretrained(model_name)
        except Exception:
            from transformers import ViTImageProcessor
            processor = ViTImageProcessor.from_pretrained(model_name)

        device       = next(self.model.parameters()).device
        pixel_values = processor(images=image, return_tensors="pt") \
                           ["pixel_values"].to(device)

        # ── find the ViTSelfAttention dropout layers ──────────────────────
        # Each ViTSelfAttention has exactly one nn.Dropout used to drop
        # attention weights.  We identify them by walking the module tree
        # and checking their parent class name.
        captured = []
        hooks    = []

        try:
            from transformers.models.vit.modeling_vit import ViTSelfAttention
            has_vit_cls = True
        except ImportError:
            has_vit_cls = False

        for parent_name, parent_mod in self.model.named_modules():
            is_self_attn = (
                (has_vit_cls and isinstance(parent_mod, ViTSelfAttention))
                or "selfattention" in type(parent_mod).__name__.lower()
            )
            if not is_self_attn:
                continue
            # The dropout inside ViTSelfAttention receives the softmax
            # attention matrix as its sole input tensor.
            for child_name, child_mod in parent_mod.named_children():
                if isinstance(child_mod, nn.Dropout):
                    def _hook(mod, inp, out, _cap=captured):
                        # inp[0] is the pre-dropout attention tensor
                        # shape: (batch, heads, seq, seq)
                        t = inp[0]
                        if t is not None and t.dim() == 4:
                            _cap.append(t.detach().cpu())
                    hooks.append(child_mod.register_forward_hook(_hook))
                    break   # only one dropout per ViTSelfAttention

        # ── forward pass ─────────────────────────────────────────────────
        try:
            with torch.no_grad():
                self.model(pixel_values=pixel_values)
        finally:
            for h in hooks:
                h.remove()

        # ── fallback ──────────────────────────────────────────────────────
        if not captured:
            return np.ones((14, 14), dtype=np.float32)

        # ── rollout ───────────────────────────────────────────────────────
        seq_len = captured[0].shape[-1]
        result  = torch.eye(seq_len)
        for attn in captured:
            attn_avg = attn[0].mean(dim=0)                    # (seq, seq)
            attn_aug = attn_avg + torch.eye(seq_len)
            attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True)
            result   = torch.matmul(attn_aug, result)

        # row 0 = CLS; columns 1: = patch tokens
        mask        = result[0, 1:]
        num_patches = int(round(mask.shape[0] ** 0.5))
        heatmap     = mask[: num_patches * num_patches] \
                          .reshape(num_patches, num_patches).numpy()
        return heatmap

    # ── CNN: Grad-CAM ─────────────────────────────────────────────────────
    def _gradcam_cnn(self, image: Image.Image,
                     target_class: Optional[int]) -> np.ndarray:
        """Classic Grad-CAM using gradient of class score w.r.t. feature maps."""
        import torch

        target_layer = self._find_target_layer()
        if target_layer is None:
            # fallback: return uniform heatmap
            return np.ones((7, 7), dtype=np.float32)

        # register hooks
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

        # dummy preprocessing — in practice use the model's own extractor
        img_array = np.array(image.resize((224, 224))) / 255.0
        img_tensor = torch.tensor(img_array.transpose(2, 0, 1),
                                  dtype=torch.float32).unsqueeze(0)

        self.model.zero_grad()
        output = self.model(pixel_values=img_tensor).logits
        if target_class is None:
            target_class = int(output.argmax().item())

        output[0, target_class].backward()

        grads  = self._gradients                        # (1, C, H, W)
        acts   = self._activations                      # (1, C, H, W)

        weights  = grads.mean(axis=(2, 3), keepdims=True)
        cam      = (weights * acts).sum(axis=1)[0]
        cam      = np.maximum(cam, 0)                   # ReLU
        cam      = cam / (cam.max() + 1e-8)             # normalise
        return cam

    # ── hooks ──────────────────────────────────────────────────────────────
    def _save_activations(self, module, input, output):
        self._activations = output.detach().cpu().numpy()

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach().cpu().numpy()

    # ── overlay ────────────────────────────────────────────────────────────
    @staticmethod
    def _overlay_heatmap(image: Image.Image, heatmap: np.ndarray,
                         alpha: float = 0.45) -> Image.Image:
        """Resize heatmap to image size and blend with original."""
        w, h = image.size
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap_uint = (heatmap_norm * 255).astype(np.uint8)
        heatmap_resized = cv2.resize(heatmap_uint, (w, h), interpolation=cv2.INTER_CUBIC)
        colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        img_array = np.array(image.convert("RGB"))
        blended   = (1 - alpha) * img_array + alpha * colored_rgb
        return Image.fromarray(blended.astype(np.uint8))

    # ── helpers ─────────────────────────────────────────────────────────────
    def _detect_model_type(self) -> str:
        """Heuristic: check model architecture via config."""
        try:
            mt = self.model.config.model_type
            return "vit" if "vit" in mt.lower() else "cnn"
        except AttributeError:
            return "cnn"

    def _find_target_layer(self):
        """Auto-detect the last Conv2d layer in a CNN."""
        target = None
        for module in self.model.modules():
            import torch.nn as nn
            if isinstance(module, nn.Conv2d):
                target = module
        return target