"""
Generative Language Model — produces a natural language summary of the
combined emotional state.

Uses GPT-2 (or a fine-tuned variant) with a structured prompt that encodes
the visual result, text sentiment, and fusion mismatch signal.

For better results in production, swap the model_name for:
  - "gpt2-medium"
  - "mistralai/Mistral-7B-Instruct-v0.1"  (requires more GPU RAM)
  - Any instruction-tuned chat model via the same interface.
"""

from __future__ import annotations
from typing import Optional


MISMATCH_TEMPLATES = {
    True:  (
        "Despite expressing {text_sent} sentiment verbally, the speaker's facial "
        "cues indicate {visual_emo}. This incongruence is worth noting in the "
        "context of the conversation and may suggest underlying {underlying}."
    ),
    False: (
        "Both verbal and facial cues are consistent. The speaker shows {visual_emo} "
        "expression alongside {text_sent} language, suggesting a coherent emotional "
        "state."
    ),
}

UNDERLYING_MAP = {
    ("sad",      "positive"): "distress or a desire to mask discomfort",
    ("fearful",  "positive"): "anxiety being suppressed behind a positive facade",
    ("angry",    "positive"): "frustration concealed with polite language",
    ("sad",      "neutral"):  "low mood not yet expressed verbally",
    ("fearful",  "neutral"):  "stress or uncertainty beneath a calm exterior",
    ("angry",    "neutral"):  "suppressed tension",
    ("happy",    "negative"): "forced positivity or sarcasm in tone",
    ("neutral",  "negative"): "verbal frustration not mirrored in expression",
}


class GenerativeSummariser:
    """
    Wraps GPT-2 for conditional text generation.
    Falls back to a rule-based template when generation quality is poor
    (e.g. model not yet fine-tuned).
    """

    def __init__(self,
                 model_name: str = "gpt2",
                 use_template_fallback: bool = True,
                 max_new_tokens: int = 80,
                 device: Optional[str] = None):
        """
        Args:
            model_name: HuggingFace model ID.
            use_template_fallback: If True, return a template string when the
                                   generated output looks too generic.
            max_new_tokens: Maximum tokens to generate.
            device: 'cpu' | 'cuda' | None.
        """
        import torch
        from transformers import GPT2Tokenizer, GPT2LMHeadModel

        self.device      = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fallback    = use_template_fallback
        self.max_new_tok = max_new_tokens

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model     = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.to(self.device).eval()
        self.tokenizer.pad_token = self.tokenizer.eos_token

    # ── public API ────────────────────────────────────────────────────────
    def summarise(self,
                  visual_result: Optional[dict],
                  text_result:   Optional[dict],
                  fusion_result: dict) -> str:
        """
        Generate a natural language summary of the multi-modal analysis.

        Returns a single descriptive sentence / short paragraph.
        """
        visual_emo = visual_result["top_emotion"] if visual_result else "unknown"
        text_sent  = text_result["top_sentiment"] if text_result else "unknown"
        mismatch   = fusion_result["mismatch_detected"]
        score      = fusion_result["mismatch_score"]

        # ── try GPT-2 generation first ─────────────────────────────────────
        prompt = self._build_prompt(visual_emo, text_sent, mismatch, score)
        generated = self._generate(prompt)

        if self.fallback and self._is_low_quality(generated):
            return self._template_summary(visual_emo, text_sent, mismatch)

        return generated.strip()

    # ── private helpers ───────────────────────────────────────────────────
    def _build_prompt(self, visual_emo: str, text_sent: str,
                      mismatch: bool, score: float) -> str:
        mismatch_str = "a mismatch" if mismatch else "alignment"
        return (
            f"Emotion analysis report: The person's face shows {visual_emo} emotion "
            f"while their words express {text_sent} sentiment. "
            f"There is {mismatch_str} (score: {score:.2f}). "
            f"Summary:"
        )

    def _generate(self, prompt: str) -> str:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt",
                                truncation=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens   = self.max_new_tok,
                do_sample        = True,
                temperature      = 0.7,
                top_p            = 0.9,
                repetition_penalty = 1.3,
                pad_token_id     = self.tokenizer.eos_token_id,
            )

        # strip the prompt from the output
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    @staticmethod
    def _is_low_quality(text: str) -> bool:
        """Heuristic: generated text is too short or repetitive."""
        words = text.split()
        return len(words) < 8 or len(set(words)) < len(words) * 0.5

    @staticmethod
    def _template_summary(visual_emo: str, text_sent: str, mismatch: bool) -> str:
        """Rule-based fallback template."""
        underlying = UNDERLYING_MAP.get(
            (visual_emo.lower(), text_sent.lower()),
            "an emotional incongruence"
        )
        template = MISMATCH_TEMPLATES[mismatch]
        return template.format(
            visual_emo  = visual_emo,
            text_sent   = text_sent,
            underlying  = underlying,
        )
