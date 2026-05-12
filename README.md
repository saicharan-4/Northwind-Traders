# MoodSyncAI — Multi-Modal Sentiment & Emotion Analyser
**DA3 Final Project | Module: Data Analytics-3**

---

## Overview

MoodSyncAI analyses the emotional state of a person from multiple input modalities — a face image, spoken/typed text, and optionally an audio clip — and detects mismatches between verbal and non-verbal signals.

```
Face image  ──► CNN / ViT ──────────────────────┐
Text input  ──► RoBERTa ──────────────────────► Fusion Layer ──► GPT-2 ──► Summary
Audio clip  ──► Whisper ──► Text pipeline ───────┘
```

---

## Architecture

| Component | Model | Purpose |
|-----------|-------|---------|
| Visual emotion | `trpakov/vit-face-expression` (ViT) | 7-class facial emotion from image |
| Text sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Positive / Neutral / Negative |
| Audio transcription | OpenAI Whisper (`base`) | Speech-to-text (optional) |
| Fusion layer | Weighted average + cosine mismatch | Combine modality predictions |
| Generative summary | GPT-2 with structured prompt | Natural language explanation |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. Use the app
1. Upload a face photo (JPG/PNG)
2. Type or paste what the person said
3. Optionally enable webcam or audio features in the sidebar
4. Click **Analyse emotions**

---

## Extended Features

| Feature | Module | How to enable |
|---------|--------|---------------|
| Webcam / real-time | `utils/webcam.py` | Toggle in sidebar |
| Audio via Whisper | `models/audio_model.py` | Toggle in sidebar |
| Grad-CAM heatmap | `utils/gradcam.py` | Toggle in sidebar |
| Token attention viz | `utils/attention_viz.py` | Toggle in sidebar |
| Learned fusion MLP | `models/fusion.py::LearnedFusionLayer` | See training below |

### Training the learned fusion MLP

```python
from models.fusion import LearnedFusionLayer
# Replace FusionLayer with LearnedFusionLayer in app.py
# Train by passing labelled (visual_vec, text_vec, true_label) examples
```

### Deploy to Hugging Face Spaces

1. Push this repo to a HuggingFace Space (Streamlit SDK)
2. Add a `README.md` with YAML front matter:
   ```yaml
   ---
   title: MoodSyncAI
   emoji: 🧠
   colorFrom: blue
   colorTo: purple
   sdk: streamlit
   app_file: app.py
   pinned: false
   ---
   ```

---

## Project Structure

```
moodsynai/
├── app.py                   # Main Streamlit app
├── requirements.txt
├── models/
│   ├── visual_model.py      # CNN / ViT facial emotion (FacialEmotionAnalyser)
│   ├── text_model.py        # RoBERTa text sentiment (TextSentimentAnalyser)
│   ├── audio_model.py       # Whisper transcription (AudioTranscriber)
│   ├── fusion.py            # Fusion layer + mismatch detection
│   └── generative.py        # GPT-2 natural language summary
└── utils/
    ├── gradcam.py           # Grad-CAM / Attention Rollout visualiser
    ├── attention_viz.py     # Token attention HTML renderer
    └── webcam.py            # Real-time webcam capture + emotion timeline
```

---

## Key Design Decisions

### Why ViT for visual emotion?
Vision Transformers consistently outperform traditional CNNs on small-to-medium face datasets (FER-2013, AffectNet) because their self-attention mechanism captures global facial structure rather than relying on local filter responses.  The `trpakov/vit-face-expression` model achieves ~75% accuracy on FER-2013.

### Why RoBERTa for text sentiment?
RoBERTa is a robustly optimised BERT variant trained without Next Sentence Prediction and with dynamic masking.  The Cardiff NLP Twitter-fine-tuned variant handles informal/conversational text well, which matches real-world speech transcripts.

### Mismatch detection via cosine distance
Both modality outputs are mapped to a shared 3-class valence space (positive / neutral / negative).  Cosine distance between the two vectors captures directional disagreement independent of confidence magnitude, making it robust to confidence calibration differences between models.

### GPT-2 generative summary
A structured prompt encoding the top emotions and mismatch flag is passed to GPT-2 with temperature=0.7.  A rule-based template fallback ensures a coherent output even when the base GPT-2 produces low-quality continuations.

---

## Sample Output

| Input | Result |
|-------|--------|
| Face: sad/fearful (68%) | Visual: **Sad** |
| Text: "No, I think the project is going really well." | Sentiment: **Positive** (81%) |
| Fusion | **MISMATCH** score: 0.73 |
| Summary | "Despite expressing positive sentiment verbally, the speaker's facial cues indicate stress or discomfort…" |

---

## Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| ViT + GPT-2 slow on CPU | Added model caching with `@st.cache_resource`; Whisper uses `tiny` on CPU |
| Token mismatch between modalities | Mapped to shared valence space before fusion |
| Webcam not available in cloud | Toggle disables cleanly; image upload fallback |
| GPT-2 generates repetitive text | Added `repetition_penalty=1.3` and template fallback |

---

*Instructor: Prof. Dr. Gayan de Silva | Exam: 13 May 2025*
