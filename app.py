import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Pine Flower Phenology Recognition",
    page_icon="🌲",
    layout="wide"
)

st.title("🌲 Pine Flower Phenology Recognition System")
st.markdown("Based on PMC_PhaseNet - Detect elongation, ripening, and decline stages")

uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)
    st.success("✅ Basic functionality working! No OpenCV dependency.")

st.info("PMC_PhaseNet model integration in progress...")
