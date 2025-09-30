import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# -------------------------
# Load mapping file
# -------------------------
def load_mapping(file_path="emnist-balanced-mapping.txt"):
    mapping = {}
    with open(file_path, "r") as f:
        for line in f:
            idx, code = line.strip().split()
            mapping[int(idx)] = chr(int(code))
    return mapping

# -------------------------
# Preprocess uploaded image
# -------------------------
def preprocess_image(image):
    # Convert to grayscale
    img = image.convert("L")

    # Ensure compatibility with Pillow versions
    if hasattr(Image, "Resampling"):
        RESAMPLE = Image.Resampling.LANCZOS
    else:
        RESAMPLE = Image.ANTIALIAS

    # Resize to 28x28 (like EMNIST dataset)
    img = img.resize((28, 28), RESAMPLE)

    # Invert colors (EMNIST expects white text on black background)
    img = ImageOps.invert(img)

    # Normalize
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array

# -------------------------
# Predict character
# -------------------------
def predict(image, model, mapping):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return mapping.get(predicted_class, "Unknown")

# -------------------------
# Streamlit App
# -------------------------
st.title("✍️ Handwritten Character Recognition — Improved")

# Upload model
uploaded_model = st.file_uploader("Upload Keras model (.h5)", type=["h5"])
# Upload mapping
uploaded_mapping = st.file_uploader("Upload mapping file (.txt)", type=["txt"])

if uploaded_model and uploaded_mapping:
    # Save uploaded files temporarily
    with open("model.h5", "wb") as f:
        f.write(uploaded_model.getbuffer())
    with open("mapping.txt", "wb") as f:
        f.write(uploaded_mapping.getbuffer())

    # Load model + mapping
    model = tf.keras.models.load_model("model.h5")
    mapping = load_mapping("mapping.txt")

    # Upload image
    uploaded_img = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "png", "jpeg"])
    if uploaded_img:
        image = Image.open(uploaded_img)
        st.image(image, caption="Original upload", use_container_width=True)

        if st.button("Predict"):
            try:
                prediction = predict(image, model, mapping)
                st.success(f"✅ Predicted Character: **{prediction}**")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
else:
    st.info("Please upload both the **.h5 model file** and the **mapping file (.txt)** to start.")
