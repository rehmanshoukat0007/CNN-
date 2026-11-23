from pathlib import Path
import pickle

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "skin_disease_cnn_model.pkl"
# flow_from_dataframe assigns class indices alphabetically by default.
CLASS_NAMES = ["Cancer", "Non_cancer"]
IMAGE_SIZE = (128, 128)

st.set_page_config(page_title="Skin Image Classifier", page_icon="🔬", layout="centered")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "skin_disease_cnn_model.pkl is missing. Upload it beside app.py."
        )

    # Only load a pickle model file that you created and trust.
    with MODEL_PATH.open("rb") as model_file:
        model_data = pickle.load(model_file)

    model = tf.keras.Sequential.from_config(model_data["config"])
    model.set_weights(model_data["weights"])
    return model


st.title("Skin Image Classification")
st.write(
    "Upload a skin image to see the class predicted by the saved CNN model. "
    "The model was trained to distinguish the Cancer and Non_cancer dataset classes."
)
st.warning(
    "Research/educational use only. This tool cannot diagnose or rule out skin cancer. "
    "Its prediction may be wrong; consult a qualified healthcare professional about any skin concern."
)

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

uploaded_image = st.file_uploader(
    "Choose a skin image", type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        resized = image.resize(IMAGE_SIZE)
        image_array = np.asarray(resized, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        if st.button("Classify image", type="primary"):
            probabilities = model.predict(image_array, verbose=0)[0]
            predicted_index = int(np.argmax(probabilities))
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(probabilities[predicted_index]) * 100

            st.subheader("Model output")
            st.metric("Predicted dataset class", predicted_class)
            st.write(f"Model confidence score: **{confidence:.2f}%**")
            st.caption(
                "The confidence score is the model's softmax output, not a measure of "
                "medical certainty or diagnostic accuracy."
            )
    except Exception as exc:
        st.error(f"The image could not be processed: {exc}")
