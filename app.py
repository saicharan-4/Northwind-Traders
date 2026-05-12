"""
MoodSyncAI — Multi-Modal Sentiment & Emotion Analyser
Entry point: run with `streamlit run app.py`
"""

import streamlit as st
from PIL import Image
import numpy as np
import tempfile, os

from models.visual_model import FacialEmotionAnalyser
from models.text_model import TextSentimentAnalyser
from models.fusion import FusionLayer
from models.generative import GenerativeSummariser
from models.audio_model import AudioTranscriber
from utils.gradcam import GradCAMVisualiser
from utils.attention_viz import render_token_attention
from utils.webcam import WebcamCapture

st.set_page_config(page_title="MoodSyncAI", page_icon="🧠", layout="wide")

# ── custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
.mismatch-banner {
    background: #FEF3C7; border: 1px solid #F59E0B;
    border-radius: 8px; padding: 12px 16px;
    color: #92400E; font-weight: 600; margin-bottom: 1rem;
}
.match-banner {
    background: #D1FAE5; border: 1px solid #10B981;
    border-radius: 8px; padding: 12px 16px;
    color: #065F46; font-weight: 600; margin-bottom: 1rem;
}
.metric-card {
    background: #F9FAFB; border-radius: 8px;
    padding: 12px; text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    mismatch_threshold = st.slider("Mismatch threshold", 0.3, 0.9, 0.5, 0.05,
        help="Cosine distance above which visual/text signals are flagged as mismatched")
    visual_weight = st.slider("Visual modality weight", 0.0, 1.0, 0.6, 0.1)
    text_weight   = 1.0 - visual_weight
    st.caption(f"Text weight: {text_weight:.1f}")

    st.divider()
    st.subheader("Optional features")
    enable_webcam = st.toggle("🎥 Webcam input", value=False)
    enable_audio  = st.toggle("🎙️ Audio via Whisper", value=False)
    show_gradcam  = st.toggle("🔥 Grad-CAM attention", value=True)
    show_tokens   = st.toggle("📝 Token attention", value=True)
    st.divider()
    st.caption("DA3 Final Project — MoodSyncAI")

# ── header ────────────────────────────────────────────────────────────────────
st.title("🧠 MoodSyncAI")
st.caption("Multi-modal sentiment & emotion analyser — CNN · RoBERTa · Fusion · GPT-2")
st.divider()

# ── input section ─────────────────────────────────────────────────────────────
col_img, col_text = st.columns(2)

with col_img:
    st.subheader("📷 Face image")
    if enable_webcam:
        st.info("Webcam mode: capture will run at analysis time")
        webcam_placeholder = st.empty()
    uploaded_file = st.file_uploader("Upload a face photo", type=["jpg", "jpeg", "png"],
                                     label_visibility="collapsed" if enable_webcam else "visible")

with col_text:
    st.subheader("💬 Spoken / typed text")
    text_input = st.text_area("Enter what the person said",
                              placeholder="e.g. No, I think the project is going really well.",
                              height=140, label_visibility="collapsed")

if enable_audio:
    st.subheader("🎙️ Audio input (Whisper)")
    audio_file = st.file_uploader("Upload a short audio clip (.wav / .mp3)",
                                  type=["wav", "mp3", "m4a"])
    if audio_file:
        st.audio(audio_file)

st.divider()
run_analysis = st.button("🔍 Analyse emotions", type="primary", use_container_width=True)

# ── analysis pipeline ─────────────────────────────────────────────────────────
if run_analysis:
    image_input = None
    final_text  = text_input.strip()

    # ── step 1: resolve image ──────────────────────────────────────────────
    with st.spinner("Loading models..."):
        visual_analyser  = FacialEmotionAnalyser()
        text_analyser    = TextSentimentAnalyser()
        fusion_layer     = FusionLayer(visual_weight=visual_weight,
                                       text_weight=text_weight,
                                       mismatch_threshold=mismatch_threshold)
        summariser       = GenerativeSummariser()

    if enable_webcam and not uploaded_file:
        with st.spinner("Capturing from webcam..."):
            webcam      = WebcamCapture()
            image_input = webcam.capture_frame()
            if image_input:
                webcam_placeholder.image(image_input, caption="Captured frame", use_container_width=True)
    elif uploaded_file:
        image_input = Image.open(uploaded_file).convert("RGB")

    # ── step 2: audio transcription ────────────────────────────────────────
    if enable_audio and audio_file:
        with st.spinner("Transcribing audio with Whisper..."):
            transcriber   = AudioTranscriber()
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name
            transcription = transcriber.transcribe(tmp_path)
            os.unlink(tmp_path)
            st.info(f"🎙️ Whisper transcript: *\"{transcription}\"*")
            if not final_text:
                final_text = transcription

    # ── step 3: validation ─────────────────────────────────────────────────
    if not image_input and not final_text:
        st.warning("Please upload an image and/or enter some text.")
        st.stop()

    # ── step 4: visual inference ────────────────────────────────────────────
    visual_result = None
    if image_input:
        with st.spinner("Running CNN on face image..."):
            visual_result = visual_analyser.predict(image_input)

    # ── step 5: text inference ─────────────────────────────────────────────
    text_result = None
    if final_text:
        with st.spinner("Running RoBERTa on text..."):
            text_result = text_analyser.predict(final_text)

    # ── step 6: fusion ─────────────────────────────────────────────────────
    with st.spinner("Computing fusion..."):
        fusion_result = fusion_layer.fuse(visual_result, text_result)

    # ── step 7: generative summary ─────────────────────────────────────────
    with st.spinner("Generating natural language summary..."):
        summary = summariser.summarise(visual_result, text_result, fusion_result)

    # ══════════════════════════════════════════════════════════════════════
    #  RESULTS
    # ══════════════════════════════════════════════════════════════════════
    st.divider()
    st.header("📊 Results")

    # mismatch / match banner
    if fusion_result["mismatch_detected"]:
        st.markdown(f"""<div class="mismatch-banner">
            ⚠️ MISMATCH DETECTED — verbal and facial signals conflict
            &nbsp;&nbsp;|&nbsp;&nbsp; score: {fusion_result['mismatch_score']:.2f}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="match-banner">✅ ALIGNED — verbal and facial signals are consistent</div>',
                    unsafe_allow_html=True)

    res_col1, res_col2 = st.columns(2)

    # ── visual results ─────────────────────────────────────────────────────
    with res_col1:
        st.subheader("🎭 Visual emotion")
        if visual_result:
            top_emotion = visual_result["top_emotion"]
            top_conf    = visual_result["confidence"][top_emotion]
            st.metric("Predicted emotion", top_emotion.capitalize(), f"{top_conf:.0%} confidence")

            st.write("**Confidence per class**")
            for emo, conf in sorted(visual_result["confidence"].items(),
                                    key=lambda x: x[1], reverse=True):
                st.progress(conf, text=f"{emo}: {conf:.0%}")

            if show_gradcam and image_input:
                with st.spinner("Generating Grad-CAM..."):
                    cam_viz = GradCAMVisualiser(visual_analyser.model)
                    heatmap = cam_viz.generate(image_input)
                    st.image(heatmap, caption="Grad-CAM attention heatmap", use_container_width=True)
        else:
            st.info("No image provided — visual analysis skipped.")

    # ── text results ───────────────────────────────────────────────────────
    with res_col2:
        st.subheader("💬 Text sentiment")
        if text_result:
            top_sent = text_result["top_sentiment"]
            top_conf = text_result["confidence"][top_sent]
            st.metric("Predicted sentiment", top_sent.capitalize(), f"{top_conf:.0%} confidence")

            st.write("**Confidence per class**")
            for sent, conf in sorted(text_result["confidence"].items(),
                                     key=lambda x: x[1], reverse=True):
                st.progress(conf, text=f"{sent}: {conf:.0%}")

            if show_tokens and text_result.get("token_attention"):
                st.write("**Token attention weights**")
                html = render_token_attention(text_result["tokens"],
                                              text_result["token_attention"])
                st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("No text provided — text analysis skipped.")

    # ── fusion ─────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔀 Fusion result")
    f_col1, f_col2, f_col3 = st.columns(3)
    f_col1.metric("Visual weight",  f"{visual_weight:.0%}")
    f_col2.metric("Text weight",    f"{text_weight:.0%}")
    f_col3.metric("Mismatch score", f"{fusion_result['mismatch_score']:.2f}")

    if fusion_result.get("fused_emotions"):
        st.write("**Fused emotion distribution**")
        for emo, conf in sorted(fusion_result["fused_emotions"].items(),
                                key=lambda x: x[1], reverse=True):
            st.progress(conf, text=f"{emo}: {conf:.0%}")

    # ── generative summary ─────────────────────────────────────────────────
    st.divider()
    st.subheader("🗣️ Generative summary")
    st.info(f'"{summary}"')

    # ── webcam emotion timeline ─────────────────────────────────────────────
    if enable_webcam and fusion_result.get("timeline"):
        st.divider()
        st.subheader("📈 Emotion timeline (webcam)")
        import pandas as pd
        df = pd.DataFrame(fusion_result["timeline"])
        st.line_chart(df.set_index("second"))