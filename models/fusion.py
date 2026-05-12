"""
Multimodal Fusion Layer
=======================
Combines visual and text modality predictions into a unified assessment.

Two fusion strategies are implemented:
  1. WeightedFusion  – simple configurable weighted average (default)
  2. LearnedFusion   – small MLP trained on combined embedding vectors
                       (extended feature — train with train_fusion.py)

Mismatch detection uses cosine distance between the two probability
vectors mapped to a shared emotion space.
"""

from __future__ import annotations
import numpy as np
from typing import Optional


# ── shared emotion space ─────────────────────────────────────────────────────
# Maps (visual_emotion, sentiment) pairs → unified emotional state labels.
# Used to build a shared probability vector for mismatch scoring.
EMOTION_TO_VALENCE = {
    "happy":     "positive",
    "surprised": "positive",
    "neutral":   "neutral",
    "disgusted": "negative",
    "sad":       "negative",
    "fearful":   "negative",
    "angry":     "negative",
    "disgust":   "negative",
}

UNIFIED_LABELS = ["positive", "neutral", "negative"]


def _visual_to_unified(confidence: dict) -> np.ndarray:
    """Collapse 7-class visual confidence into 3-class valence vector."""
    v = np.zeros(3)
    for emo, conf in confidence.items():
        valence = EMOTION_TO_VALENCE.get(emo, "neutral")
        idx = UNIFIED_LABELS.index(valence)
        v[idx] += conf
    return v / (v.sum() + 1e-8)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Returns cosine distance in [0, 1]; 0 = identical, 1 = opposite."""
    dot   = np.dot(a, b)
    norms = np.linalg.norm(a) * np.linalg.norm(b)
    if norms < 1e-8:
        return 0.0
    cosine_sim = dot / norms
    return float((1.0 - cosine_sim) / 2.0)   # map [-1,1] → [0,1]


# ── weighted fusion ───────────────────────────────────────────────────────────
class FusionLayer:
    """
    Default fusion: configurable weighted average of the two modality
    probability vectors (both mapped to the unified 3-class space).

    Parameters
    ----------
    visual_weight : float
        Weight of the visual modality (0–1).
    text_weight : float
        Weight of the text modality; should sum to 1 with visual_weight.
    mismatch_threshold : float
        Cosine distance above which the two signals are flagged as mismatched.
    """

    def __init__(self,
                 visual_weight: float = 0.6,
                 text_weight:   float = 0.4,
                 mismatch_threshold: float = 0.5):
        self.visual_weight      = visual_weight
        self.text_weight        = text_weight
        self.mismatch_threshold = mismatch_threshold

    def fuse(self,
             visual_result: Optional[dict],
             text_result:   Optional[dict]) -> dict:
        """
        Combine the two modality results.

        Returns
        -------
        {
            "mismatch_detected": bool,
            "mismatch_score":    float,           # cosine distance 0→1
            "fused_emotions":    dict[str, float],
            "visual_valence":    np.ndarray | None,
            "text_valence":      np.ndarray | None,
            "timeline":          list[dict],       # for webcam mode
        }
        """
        visual_vec = None
        text_vec   = None

        if visual_result:
            visual_vec = _visual_to_unified(visual_result["confidence"])

        if text_result:
            # text model already outputs positive/neutral/negative
            conf = text_result["confidence"]
            text_vec = np.array([
                conf.get("positive", 0.0),
                conf.get("neutral",  0.0),
                conf.get("negative", 0.0),
            ])
            text_vec /= (text_vec.sum() + 1e-8)

        # ── mismatch detection ────────────────────────────────────────────
        if visual_vec is not None and text_vec is not None:
            mismatch_score = _cosine_distance(visual_vec, text_vec)
        else:
            mismatch_score = 0.0

        mismatch_detected = mismatch_score > self.mismatch_threshold

        # ── weighted fusion ────────────────────────────────────────────────
        if visual_vec is not None and text_vec is not None:
            fused_vec = (self.visual_weight * visual_vec +
                         self.text_weight   * text_vec)
        elif visual_vec is not None:
            fused_vec = visual_vec
        elif text_vec is not None:
            fused_vec = text_vec
        else:
            fused_vec = np.array([1/3, 1/3, 1/3])

        fused_emotions = {label: float(fused_vec[i])
                          for i, label in enumerate(UNIFIED_LABELS)}

        return {
            "mismatch_detected": mismatch_detected,
            "mismatch_score":    round(mismatch_score, 3),
            "fused_emotions":    fused_emotions,
            "visual_valence":    visual_vec,
            "text_valence":      text_vec,
            "timeline":          [],     # populated by webcam mode
        }


# ── learned fusion (extended feature) ────────────────────────────────────────
class LearnedFusionLayer(FusionLayer):
    """
    Extended feature: replaces the weighted average with a small MLP
    trained to combine the two modality embedding vectors.

    The MLP takes [visual_probs | text_probs] (3+3 = 6-dim) as input
    and outputs a 3-dim fused distribution.

    Training: see train_fusion.py
    """

    def __init__(self, model_path: str = "models/learned_fusion.pt",
                 **kwargs):
        super().__init__(**kwargs)
        self.mlp = self._load_or_init_mlp(model_path)

    def _load_or_init_mlp(self, path: str):
        import torch, torch.nn as nn
        mlp = nn.Sequential(
            nn.Linear(6, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 3),  nn.Softmax(dim=-1),
        )
        try:
            mlp.load_state_dict(torch.load(path, map_location="cpu"))
            mlp.eval()
        except FileNotFoundError:
            pass   # returns randomly initialised MLP — train first
        return mlp

    def fuse(self, visual_result, text_result):
        import torch
        base = super().fuse(visual_result, text_result)

        if base["visual_valence"] is not None and base["text_valence"] is not None:
            x       = np.concatenate([base["visual_valence"], base["text_valence"]])
            x_t     = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                out = self.mlp(x_t)[0].numpy()
            base["fused_emotions"] = {label: float(out[i])
                                      for i, label in enumerate(UNIFIED_LABELS)}
        return base
