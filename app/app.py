"""
SmartInspect AI — Competition-Ready Streamlit Application
Industrial Defect Detection using EfficientNetB2 Transfer Learning

Run with:
    streamlit run app/app.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import csv
import io
import json
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

# Ensure utils.py (in the same directory) is importable regardless of CWD
sys.path.insert(0, str(Path(__file__).parent))

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import streamlit as st
from PIL import Image

# ── Local utilities ───────────────────────────────────────────────────────────
from utils import (
    format_product_name,
    get_top_predictions,
    is_low_confidence,
    load_pil_image,
    parse_class_name,
    prepare_csv_row,
    preprocess_for_inference,
    results_to_csv_string,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "model" / "SmartInspect_EfficientNetB2_Final.keras"
CLASS_NAMES_PATH = PROJECT_DIR / "model" / "class_names.json"
IMG_SIZE = (260, 260)
LOW_CONFIDENCE_THRESHOLD = 60.0
MAX_BATCH_SIZE = 10
SUPPORTED_PRODUCTS = [
    "Bottle", "BSD", "Cable", "Carpet", "Grid",
    "Leather", "Metal Nut", "Toothbrush", "Wood", "Zipper",
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Page configuration  (must be the FIRST Streamlit call)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="SmartInspect AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model loading (cached across reruns / users)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_resource(show_spinner="Loading SmartInspect AI model…")
def load_model_and_classes():
    """
    Load the Keras model and class-names list exactly once per server process.

    Returns
    -------
    (model, class_names) on success.
    Raises RuntimeError with a user-friendly message on failure.
    """
    import tensorflow as tf  # deferred import — keeps startup fast if TF not needed

    # ── Validate paths ────────────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at:\n\n`{MODEL_PATH}`\n\n"
            "**Setup instructions:**\n"
            "1. Place `SmartInspect_EfficientNetB2_Final.keras` in the `model/` folder.\n"
            "2. Place `class_names.json` in the same `model/` folder.\n"
            "3. Restart the Streamlit server."
        )
    if not CLASS_NAMES_PATH.exists():
        raise RuntimeError(
            f"Class-names file not found at:\n\n`{CLASS_NAMES_PATH}`\n\n"
            "Please place `class_names.json` in the `model/` folder and restart."
        )

    # ── Load class names ──────────────────────────────────────────────────────
    try:
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as fh:
            class_names = json.load(fh)
        if not isinstance(class_names, list) or len(class_names) == 0:
            raise ValueError("class_names.json must be a non-empty JSON array of strings.")
    except Exception as exc:
        raise RuntimeError(f"Could not read class_names.json: {exc}") from exc

    # ── Load Keras model ──────────────────────────────────────────────────────
    try:
        model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load Keras model.\n\nError: {exc}\n\n"
            "Make sure the `.keras` file is not corrupted and was saved with "
            "a compatible version of TensorFlow/Keras."
        ) from exc

    # ── Sanity-check output dimension ─────────────────────────────────────────
    num_model_outputs = model.output_shape[-1]
    if num_model_outputs != len(class_names):
        raise RuntimeError(
            f"Mismatch: model has {num_model_outputs} output neurons but "
            f"class_names.json contains {len(class_names)} entries. "
            "Ensure both files come from the same training run."
        )

    return model, class_names


# ── Attempt to load — surface any error gracefully ───────────────────────────
try:
    model, class_names = load_model_and_classes()
    _model_loaded = True
except RuntimeError as _load_error:
    _model_loaded = False
    _load_error_msg = str(_load_error)
except Exception as _load_error:
    _model_loaded = False
    _load_error_msg = (
        "An unexpected error occurred while loading the model. "
        "Check that TensorFlow is installed and the model file is valid."
    )

if not _model_loaded:
    st.error("### ❌ Model Load Failed")
    st.markdown(_load_error_msg)
    st.info(
        "**Quick fix checklist:**\n"
        "- `model/SmartInspect_EfficientNetB2_Final.keras` exists\n"
        "- `model/class_names.json` exists and is a valid JSON array\n"
        "- TensorFlow ≥ 2.10 is installed (`pip install tensorflow`)\n"
        "- Both files are from the same training run"
    )
    st.stop()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sidebar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("## 🔍 SmartInspect AI")
    st.markdown("*Industrial Defect Detection*")
    st.divider()

    st.markdown("### 📋 Supported Products")
    for product in SUPPORTED_PRODUCTS:
        st.markdown(f"- {product}")
    st.divider()

    st.markdown("### ℹ️ How It Works")
    st.markdown(
        "1. **Upload Image** — Select a product image (JPG / PNG / BMP)\n"
        "2. **Validate** — Image is checked for size and format\n"
        "3. **Analyze** — EfficientNetB2 classifies the product\n"
        "4. **Results** — View defect status and confidence score"
    )
    st.divider()

    st.markdown("### ⚙️ Model Info")
    st.markdown(
        "| Property | Value |\n"
        "|---|---|\n"
        "| Architecture | EfficientNetB2 |\n"
        "| Classes | 20 |\n"
        "| Test Accuracy | 89.15% |\n"
        "| Approach | Transfer Learning |"
    )
    st.divider()

    st.markdown("### ⚠️ Important Notes")
    st.markdown(
        "- This is an **academic prototype** — not validated for real industrial use.\n"
        "- Confidence score ≠ accuracy guarantee.\n"
        "- Only reliable for the 10 listed product categories.\n"
        "- Results may vary with image quality and lighting."
    )
    st.divider()

    st.caption("Academic Prototype — Not for industrial use")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main area
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.title("🔍 SmartInspect AI")
st.markdown(
    "Automated industrial defect detection powered by **EfficientNetB2** transfer learning. "
    "Upload product images to instantly classify them as **Good** or **Defective**."
)

# ── Session state initialisation ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state["history"] = []
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tabs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
tab_single, tab_batch, tab_history = st.tabs(
    ["🔍 Single Inspection", "📦 Batch Inspection", "📋 History"]
)

# ────────────────────────────────────────────────────────────────────────────
# Tab 1 — Single Inspection
# ────────────────────────────────────────────────────────────────────────────
with tab_single:
    st.markdown(
        "Upload a single product image to check whether it is **Good** or **Defective**. "
        "Supported formats: JPG, JPEG, PNG, BMP."
    )

    uploaded_file = st.file_uploader(
        "Choose a product image",
        type=["jpg", "jpeg", "png", "bmp"],
        key="single_uploader",
        help="Upload one image. Maximum recommended resolution: 4000×4000 px.",
    )

    if uploaded_file is not None:
        col_img, col_info = st.columns([2, 1])

        with col_img:
            st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)

        with col_info:
            # Show basic image metadata without running inference yet
            try:
                uploaded_file.seek(0)
                _preview = Image.open(uploaded_file)
                st.markdown("**Image Info**")
                st.markdown(f"- **Filename:** `{uploaded_file.name}`")
                st.markdown(f"- **Dimensions:** {_preview.width} × {_preview.height} px")
                st.markdown(f"- **Color mode:** {_preview.mode}")
                st.markdown(f"- **File size:** {uploaded_file.size / 1024:.1f} KB")
            except Exception:
                st.markdown(f"- **Filename:** `{uploaded_file.name}`")

        st.markdown("")
        inspect_btn = st.button(
            "🔍 Inspect Product",
            key="single_inspect_btn",
            type="primary",
            use_container_width=True,
        )

        if inspect_btn:
            # Reset last result so we always re-predict on a new click
            st.session_state["last_result"] = None

            # ── Load and validate image ───────────────────────────────────────
            uploaded_file.seek(0)
            pil_img, load_error = load_pil_image(uploaded_file)
            if load_error:
                st.error(f"❌ **Image Error:** {load_error}")
                st.stop()

            # ── Run inference ─────────────────────────────────────────────────
            try:
                with st.spinner("🔍 Analyzing image…"):
                    batch = preprocess_for_inference(pil_img, img_size=IMG_SIZE)
                    raw_preds = model.predict(batch, verbose=0)
                    probabilities = raw_preds[0]

                # Guard against NaN / Inf from a corrupted model output
                if not np.all(np.isfinite(probabilities)):
                    st.error(
                        "❌ The model returned invalid probability values. "
                        "This may indicate a corrupted model file."
                    )
                    st.stop()

                predicted_index = int(np.argmax(probabilities))
                confidence = float(probabilities[predicted_index]) * 100.0
                class_name = class_names[predicted_index]
                product_raw, status = parse_class_name(class_name)
                product_display = format_product_name(product_raw)
                low_conf = is_low_confidence(confidence, LOW_CONFIDENCE_THRESHOLD)
                top_preds = get_top_predictions(probabilities, class_names, n=3)
                ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

                # Cache result in session state
                st.session_state["last_result"] = {
                    "filename": uploaded_file.name,
                    "product_display": product_display,
                    "status": status,
                    "class_name": class_name,
                    "confidence": confidence,
                    "low_conf": low_conf,
                    "top_preds": top_preds,
                    "timestamp": ts,
                }

                # ── Add to history (deduplicate same filename from reruns) ────
                history_filenames = [h["filename"] for h in st.session_state["history"]]
                if uploaded_file.name not in history_filenames:
                    csv_row = prepare_csv_row(
                        filename=uploaded_file.name,
                        product=product_display,
                        status=status,
                        predicted_class=class_name,
                        confidence_pct=confidence,
                        is_low_conf=low_conf,
                        timestamp=ts,
                    )
                    st.session_state["history"].append({
                        "filename": uploaded_file.name,
                        "product": product_display,
                        "status": status,
                        "confidence": confidence,
                        "warning": "YES" if low_conf else "NO",
                        "timestamp": ts,
                        "csv_row": csv_row,
                    })

            except Exception as exc:
                st.error(
                    f"❌ **Inference failed:** An unexpected error occurred during analysis.\n\n"
                    f"Details: `{exc}`"
                )
                st.stop()

        # ── Display results (from session state, survives reruns) ─────────────
        result = st.session_state.get("last_result")
        if result:
            st.divider()
            st.subheader("Inspection Result")

            col_product, col_status, col_conf = st.columns(3)

            with col_product:
                st.metric("🏭 Product", result["product_display"])

            with col_status:
                status_emoji = "✅ GOOD" if result["status"] == "good" else (
                    "⚠️ DEFECTIVE" if result["status"] == "defective" else "❓ UNKNOWN"
                )
                st.metric("🔎 Status", status_emoji)

            with col_conf:
                st.metric("📊 Confidence", f"{result['confidence']:.1f}%")

            st.markdown("")

            if result["status"] == "defective":
                st.error("⚠️ **DEFECTIVE** — This product appears to have defects. "
                         "Manual inspection is recommended.")
            elif result["status"] == "good":
                st.success("✅ **GOOD** — This product appears to be in good condition.")
            else:
                st.info("❓ **UNKNOWN** — The model could not confidently classify this product.")

            if result["low_conf"]:
                st.warning(
                    f"⚠️ **Low Confidence ({result['confidence']:.1f}%)** — "
                    "The model is not confident about this result. "
                    "The image may be outside the training distribution, poorly lit, "
                    "or not from a supported product category. Manual verification is advised."
                )

            # ── Top-3 predictions ─────────────────────────────────────────────
            st.markdown("#### Top Predictions")
            for rank, (cname, prod_raw, stat, conf_pct) in enumerate(result["top_preds"], 1):
                label = f"{rank}. **{format_product_name(prod_raw)}** — {stat.capitalize()} ({conf_pct:.1f}%)"
                st.markdown(label)
                st.progress(min(conf_pct / 100.0, 1.0))

            # ── Single-result CSV download ─────────────────────────────────────
            single_csv_row = prepare_csv_row(
                filename=result["filename"],
                product=result["product_display"],
                status=result["status"],
                predicted_class=result["class_name"],
                confidence_pct=result["confidence"],
                is_low_conf=result["low_conf"],
                timestamp=result["timestamp"],
            )
            csv_string = results_to_csv_string([single_csv_row])
            st.download_button(
                label="⬇️ Download Result as CSV",
                data=csv_string,
                file_name=f"smartinspect_{result['filename'].rsplit('.', 1)[0]}_result.csv",
                mime="text/csv",
                key="single_download_btn",
            )

# ────────────────────────────────────────────────────────────────────────────
# Tab 2 — Batch Inspection
# ────────────────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown(
        "Upload up to **10 product images** for batch inspection. "
        "Each image is processed individually and results are shown in a summary table."
    )
    st.info(f"ℹ️ Maximum batch size: **{MAX_BATCH_SIZE} images** per run.")

    batch_files = st.file_uploader(
        "Choose product images (up to 10)",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="batch_uploader",
        help=f"Select up to {MAX_BATCH_SIZE} images at once.",
    )

    if batch_files:
        # Clamp to max
        if len(batch_files) > MAX_BATCH_SIZE:
            st.warning(
                f"⚠️ You uploaded {len(batch_files)} images. "
                f"Only the first {MAX_BATCH_SIZE} will be processed."
            )
            batch_files = batch_files[:MAX_BATCH_SIZE]

        st.markdown(f"**{len(batch_files)} image(s) ready for inspection.**")

        batch_btn = st.button(
            "🔍 Inspect All Products",
            key="batch_inspect_btn",
            type="primary",
            use_container_width=True,
        )

        if batch_btn:
            batch_results = []
            failed_images = []

            progress_bar = st.progress(0, text="Processing images…")

            for i, f in enumerate(batch_files):
                progress_bar.progress(
                    (i + 1) / len(batch_files),
                    text=f"Processing {f.name} ({i + 1}/{len(batch_files)})…",
                )
                try:
                    f.seek(0)
                    pil_img, load_error = load_pil_image(f)
                    if load_error:
                        failed_images.append({"filename": f.name, "error": load_error})
                        continue

                    batch_arr = preprocess_for_inference(pil_img, img_size=IMG_SIZE)
                    raw_preds = model.predict(batch_arr, verbose=0)
                    probs = raw_preds[0]

                    if not np.all(np.isfinite(probs)):
                        failed_images.append({
                            "filename": f.name,
                            "error": "Model returned non-finite probability values.",
                        })
                        continue

                    pred_idx = int(np.argmax(probs))
                    conf = float(probs[pred_idx]) * 100.0
                    cname = class_names[pred_idx]
                    prod_raw, stat = parse_class_name(cname)
                    prod_disp = format_product_name(prod_raw)
                    low_conf = is_low_confidence(conf, LOW_CONFIDENCE_THRESHOLD)
                    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

                    status_display = (
                        "✅ Good" if stat == "good" else
                        "⚠️ Defective" if stat == "defective" else
                        "❓ Unknown"
                    )

                    csv_row = prepare_csv_row(
                        filename=f.name,
                        product=prod_disp,
                        status=stat,
                        predicted_class=cname,
                        confidence_pct=conf,
                        is_low_conf=low_conf,
                        timestamp=ts,
                    )

                    batch_results.append({
                        "Filename": f.name,
                        "Product": prod_disp,
                        "Status": status_display,
                        "Confidence (%)": f"{conf:.1f}",
                        "Low Confidence Warning": "⚠️ Yes" if low_conf else "No",
                        "_csv_row": csv_row,
                        "_stat": stat,
                        "_ts": ts,
                    })

                    # Add to session history (deduplicate)
                    history_filenames = [h["filename"] for h in st.session_state["history"]]
                    if f.name not in history_filenames:
                        st.session_state["history"].append({
                            "filename": f.name,
                            "product": prod_disp,
                            "status": stat,
                            "confidence": conf,
                            "warning": "YES" if low_conf else "NO",
                            "timestamp": ts,
                            "csv_row": csv_row,
                        })

                except Exception as exc:
                    failed_images.append({"filename": f.name, "error": str(exc)})

            progress_bar.empty()

            # ── Results table ─────────────────────────────────────────────────
            if batch_results:
                st.subheader("Batch Results")

                display_rows = [
                    {k: v for k, v in r.items() if not k.startswith("_")}
                    for r in batch_results
                ]
                st.dataframe(display_rows, use_container_width=True)

                # Summary stats
                total = len(batch_results)
                good_count = sum(1 for r in batch_results if r["_stat"] == "good")
                defective_count = sum(1 for r in batch_results if r["_stat"] == "defective")
                warning_count = sum(
                    1 for r in batch_results if r["Low Confidence Warning"] != "No"
                )

                st.markdown("#### Summary")
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Total Processed", total)
                s2.metric("✅ Good", good_count)
                s3.metric("⚠️ Defective", defective_count)
                s4.metric("🔔 Low Confidence", warning_count)

                # Batch CSV download
                csv_rows = [r["_csv_row"] for r in batch_results]
                batch_csv = results_to_csv_string(csv_rows)
                st.download_button(
                    label="⬇️ Download All Results as CSV",
                    data=batch_csv,
                    file_name="smartinspect_batch_results.csv",
                    mime="text/csv",
                    key="batch_download_btn",
                )

            # ── Failed images ─────────────────────────────────────────────────
            if failed_images:
                st.subheader("⚠️ Failed Images")
                st.markdown(
                    f"{len(failed_images)} image(s) could not be processed:"
                )
                for fail in failed_images:
                    st.error(f"**{fail['filename']}** — {fail['error']}")

            if not batch_results and not failed_images:
                st.info("No results to display.")

# ────────────────────────────────────────────────────────────────────────────
# Tab 3 — History
# ────────────────────────────────────────────────────────────────────────────
with tab_history:
    st.subheader("📋 Inspection History")
    st.caption(
        "Session history is temporary and will be cleared when you close or refresh the browser."
    )

    history = st.session_state.get("history", [])

    if not history:
        st.info(
            "📭 No inspections yet. "
            "Use the **Single Inspection** or **Batch Inspection** tabs above to get started."
        )
    else:
        # Build display-friendly records
        display_history = []
        for entry in history:
            status_disp = (
                "✅ Good" if entry["status"] == "good" else
                "⚠️ Defective" if entry["status"] == "defective" else
                "❓ Unknown"
            )
            display_history.append({
                "Timestamp": entry.get("timestamp", "—"),
                "Filename": entry["filename"],
                "Product": entry["product"],
                "Status": status_disp,
                "Confidence (%)": f"{entry['confidence']:.1f}",
                "Low Confidence Warning": entry.get("warning", "NO"),
            })

        st.dataframe(display_history, use_container_width=True)
        st.markdown(f"**{len(history)} inspection(s)** recorded this session.")

        col_clear, col_dl = st.columns(2)

        with col_clear:
            if st.button("🗑️ Clear History", key="clear_history_btn", use_container_width=True):
                st.session_state["history"] = []
                st.session_state["last_result"] = None
                st.rerun()

        with col_dl:
            history_csv_rows = [h["csv_row"] for h in history if "csv_row" in h]
            if history_csv_rows:
                history_csv = results_to_csv_string(history_csv_rows)
                st.download_button(
                    label="⬇️ Download History as CSV",
                    data=history_csv,
                    file_name="smartinspect_history.csv",
                    mime="text/csv",
                    key="history_download_btn",
                    use_container_width=True,
                )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Footer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 0.85em;'>"
    "SmartInspect AI &nbsp;|&nbsp; Industrial Defect Detection &nbsp;|&nbsp; "
    "EfficientNetB2 Transfer Learning &nbsp;|&nbsp; Academic Prototype"
    "</div>",
    unsafe_allow_html=True,
)
