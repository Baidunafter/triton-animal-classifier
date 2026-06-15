"""Streamlit-приложение для классификации животных через Triton API."""
import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://localhost:8080/predict"

st.set_page_config(page_title="Animal Classifier", page_icon="🐾")
st.title("🐾 Animal Classifier (cat / dog / horse)")
st.caption("Powered by ResNet50 + Triton Inference Server")

uploaded = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Загруженное изображение", use_container_width=True)
    with col2:
        with st.spinner("Классифицирую..."):
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG")
            buf.seek(0)
            try:
                resp = requests.post(API_URL, files={"file": buf}, timeout=30)
                resp.raise_for_status()
                result = resp.json()
                st.success(f"### {result['class'].upper()}")
                st.metric("Confidence", f"{result['confidence']*100:.2f}%")
                st.subheader("Распределение вероятностей")
                for cls, prob in result["probabilities"].items():
                    st.progress(prob, text=f"{cls}: {prob*100:.2f}%")
            except Exception as e:
                st.error(f"Ошибка: {e}")
