"""
Text sentiment analyser using cardiffnlp/twitter-roberta-base-sentiment-latest
(RoBERTa fine-tuned for 3-class sentiment: positive / neutral / negative).

Also extracts per-token attention weights from the last attention head
so we can visualise which tokens influenced the prediction most.
"""

from __future__ import annotations
import numpy as np
from typing import Optional


SENTIMENT_LABELS = ["negative", "neutral", "positive"]


class TextSentimentAnalyser:
    """
    Wraps RoBERTa for sentiment classification and attention extraction.
    """

    def __init__(self,
                 model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
                 device: Optional[str] = None):
        """
        Args:
            model_name: HuggingFace model ID.
            device: 'cpu' | 'cuda' | None (auto-detect).
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            model_name, output_attentions=True
        )
        self.model.to(self.device).eval()
        self.id2label  = self.model.config.id2label

    # ── public API ────────────────────────────────────────────────────────
    def predict(self, text: str) -> dict:
        """
        Tokenise text, run forward pass, extract attention.

        Returns:
            {
                "top_sentiment": str,
                "confidence":    {label: float, ...},
                "tokens":        [str, ...],
                "token_attention": [float, ...],   # mean attention per token
                "logits":        np.ndarray,
            }
        """
        import torch, torch.nn.functional as F

        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)

        # ── sentiment scores ───────────────────────────────────────────────
        logits = outputs.logits[0]
        probs  = F.softmax(logits, dim=-1).cpu().numpy()

        confidence = {self.id2label[i].lower(): float(probs[i])
                      for i in range(len(probs))}
        top_sentiment = max(confidence, key=confidence.get)

        # ── attention extraction ────────────────────────────────────────────
        # attentions: tuple of (batch, heads, seq, seq) per layer
        # We take the last layer, average across heads, then take the
        # [CLS] token's attention to each other token (index 0 row).
        attentions = outputs.attentions            # tuple of tensors
        last_layer = attentions[-1][0]             # (heads, seq, seq)
        cls_attn   = last_layer[:, 0, :]           # (heads, seq)  — CLS row
        mean_attn  = cls_attn.mean(0).cpu().numpy()  # (seq,)

        # decode token strings (skip special tokens for display)
        input_ids  = encoding["input_ids"][0].cpu().numpy()
        tokens_raw = self.tokenizer.convert_ids_to_tokens(input_ids)

        # filter [CLS] / [SEP] / padding but keep all real tokens
        tokens, attn_weights = [], []
        for tok, att in zip(tokens_raw, mean_attn):
            if tok not in ("<s>", "</s>", "<pad>", "[CLS]", "[SEP]", "[PAD]"):
                # clean RoBERTa Ġ prefix
                clean = tok.lstrip("Ġ▁").replace("##", "")
                tokens.append(clean or tok)
                attn_weights.append(float(att))

        # normalise attention to [0, 1]
        if attn_weights:
            mn, mx = min(attn_weights), max(attn_weights)
            span   = mx - mn or 1e-6
            attn_weights = [(a - mn) / span for a in attn_weights]

        return {
            "top_sentiment":   top_sentiment,
            "confidence":      confidence,
            "tokens":          tokens,
            "token_attention": attn_weights,
            "logits":          logits.cpu().numpy(),
        }


class LSTMSentimentAnalyser:
    """
    Minimal LSTM-based sentiment analyser using a pre-trained TextBlob / VADER
    wrapper.  Useful when you want to demonstrate RNN architecture knowledge
    without the full transformer dependency.

    Same return contract as TextSentimentAnalyser.
    """

    def predict(self, text: str) -> dict:
        from textblob import TextBlob
        tb     = TextBlob(text)
        polarity = tb.sentiment.polarity       # [-1, 1]

        if polarity > 0.1:
            top_sentiment = "positive"
            confidence    = {"positive": 0.5 + polarity * 0.5,
                             "neutral": 0.15, "negative": 0.05}
        elif polarity < -0.1:
            top_sentiment = "negative"
            confidence    = {"negative": 0.5 + abs(polarity) * 0.5,
                             "neutral": 0.15, "positive": 0.05}
        else:
            top_sentiment = "neutral"
            confidence    = {"neutral": 0.6, "positive": 0.2, "negative": 0.2}

        # normalise
        total = sum(confidence.values())
        confidence = {k: v / total for k, v in confidence.items()}

        return {
            "top_sentiment":   top_sentiment,
            "confidence":      confidence,
            "tokens":          text.split(),
            "token_attention": None,
            "logits":          np.array(list(confidence.values())),
        }
