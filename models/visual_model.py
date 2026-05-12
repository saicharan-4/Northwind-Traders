"""
Visual emotion analyser using a fine-tuned Vision Transformer (ViT)
from HuggingFace: trpakov/vit-face-expression

Falls back gracefully to a lightweight CNN (DeepFace) if the ViT
is unavailable or if the user prefers a smaller model.
"""

from __future__ import annotations
import numpy as np
from PIL import Image
from typing import Optional

EMOTION_LABELS = ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]


class FacialEmotionAnalyser:
    """
    Wraps a ViT fine-tuned on FER-2013 / AffectNet for 7-class facial emotion
    recognition.  Exposes a single .predict(image) method that returns a
    standardised dict so the rest of the pipeline is model-agnostic.
    """

    def __init__(self, model_name: str = "trpakov/vit-face-expression",
                 device: Optional[str] = None):
        """
        Args:
            model_name: HuggingFace model ID.
            device: 'cpu' | 'cuda' | None (auto-detect).
        """
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.extractor = AutoImageProcessor.from_pretrained(model_name)
        self.model     = AutoModelForImageClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()

        # store id2label for grad-cam visualiser
        self.id2label = self.model.config.id2label

    # ── public API ────────────────────────────────────────────────────────
    def predict(self, image: Image.Image) -> dict:
        """
        Run forward pass and return structured results.

        Returns:
            {
                "top_emotion": str,
                "confidence": {emotion: float, ...},   # sums to 1.0
                "logits": np.ndarray,                  # raw logits (for Grad-CAM)
                "tokens": None,                        # unused for images
                "token_attention": None,
            }
        """
        import torch, torch.nn.functional as F

        inputs  = self.extractor(images=image, return_tensors="pt")
        inputs  = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[0]                          # (num_classes,)
        probs  = F.softmax(logits, dim=-1).cpu().numpy()

        # build label → probability dict using model's own label map
        confidence = {self.id2label[i].lower(): float(probs[i])
                      for i in range(len(probs))}

        top_emotion = max(confidence, key=confidence.get)

        return {
            "top_emotion":    top_emotion,
            "confidence":     confidence,
            "logits":         logits.cpu().numpy(),
            "tokens":         None,
            "token_attention": None,
        }


class CNNFacialAnalyser:
    """
    Lightweight alternative using DeepFace (no HuggingFace required).
    Good for demo / offline use.  Same return format as FacialEmotionAnalyser.
    """

    def predict(self, image: Image.Image) -> dict:
        from deepface import DeepFace
        import numpy as np

        img_array = np.array(image)
        result    = DeepFace.analyze(img_array, actions=["emotion"],
                                     enforce_detection=False, silent=True)
        emotions  = result[0]["emotion"]                    # {str: float %}

        # normalise to 0-1 probabilities
        total      = sum(emotions.values())
        confidence = {k.lower(): v / total for k, v in emotions.items()}
        top_emotion = max(confidence, key=confidence.get)

        return {
            "top_emotion":    top_emotion,
            "confidence":     confidence,
            "logits":         np.array(list(emotions.values())),
            "tokens":         None,
            "token_attention": None,
        }