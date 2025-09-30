import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Load mapping
def load_mapping():
    mapping = {}
    with open("emnist-balanced-mapping.txt", "r") as f:
        for line in f:
            key, val = line.strip().split()
            mapping[int(key)] = chr(int(val))
    return mapping

# Load trained model
@st.cache_resource
def load_trained_model():
    return load_model("emnist_manual_cnn.h5")

model = load_trained_model()
mapping = load_mapping()

st.title("🖐️ Handwritten Character Recognition")
st.markdown("Upload a **28x28 grayscale** handwritten image (A-Z, 0-9).")

uploaded_file = st.file_uploader("📤 Upload your image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")

    # ✅ Rotate and Flip to match EMNIST format
    image_resized = image.resize((28, 28)).transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT)
    image_resized = ImageOps.invert(image_resized)

    st.image(image_resized, caption="Uploaded Image", width=150)

    img_array = np.array(image_resized).reshape(1, 28, 28, 1).astype("float32") / 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = 100 * np.max(prediction)

    predicted_char = mapping[predicted_class]

    st.success(f"✅ Predicted Character: **{predicted_char}**")
    st.progress(int(confidence))
    st.caption(f"Confidence: {confidence:.2f}%")

