
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "model"
    / "SmartInspect_EfficientNetB2_Final.keras"
)

CLASS_NAMES_PATH = (
    PROJECT_DIR
    / "model"
    / "class_names.json"
)

IMG_SIZE = (260, 260)


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="SmartInspect AI",
    page_icon="🔍",
    layout="centered"
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_classes():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


model = load_model()
class_names = load_classes()


# --------------------------------------------------
# Application UI
# --------------------------------------------------

st.title("🔍 SmartInspect AI")
st.subheader("Industrial Defect Detection")

st.write(
    "CNN + EfficientNetB2 Transfer Learning"
)

st.divider()


uploaded_file = st.file_uploader(
    "Upload an industrial product image",
    type=["jpg", "jpeg", "png", "bmp"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Product Image",
        use_container_width=True
    )

    if st.button(
        "🔍 Inspect Product",
        type="primary",
        use_container_width=True
    ):

        # Resize image
        resized = image.resize(IMG_SIZE)

        # Convert to NumPy
        image_array = np.array(
            resized,
            dtype=np.float32
        )

        # Add batch dimension
        image_batch = np.expand_dims(
            image_array,
            axis=0
        )

        # Prediction
        probabilities = model.predict(
            image_batch,
            verbose=0
        )[0]

        predicted_index = np.argmax(probabilities)

        predicted_class = class_names[
            predicted_index
        ]

        confidence = (
            probabilities[predicted_index] * 100
        )

        # Separate product and status
        product, status = predicted_class.rsplit(
            "_",
            1
        )

        # --------------------------------------------------
        # Results
        # --------------------------------------------------

        st.divider()

        st.header("Inspection Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Product",
                product.replace(
                    "_",
                    " "
                ).title()
            )

        with col2:
            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        if status == "defective":

            st.error(
                "⚠️ DEFECTIVE PRODUCT"
            )

        else:

            st.success(
                "✅ GOOD PRODUCT"
            )

        st.write(
            f"**Predicted Class:** `{predicted_class}`"
        )

        # --------------------------------------------------
        # Top 3 Predictions
        # --------------------------------------------------

        st.subheader(
            "Top 3 Predictions"
        )

        top_indices = np.argsort(
            probabilities
        )[::-1][:3]

        for index in top_indices:

            score = (
                probabilities[index] * 100
            )

            st.write(
                f"**{class_names[index]}** — "
                f"{score:.2f}%"
            )

            st.progress(
                float(probabilities[index])
            )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "SmartInspect AI | "
    "Industrial Defect Detection | "
    "EfficientNetB2 Transfer Learning"
)
