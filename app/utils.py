# app/utils.py
"""
SmartInspect AI — Utility functions for inference pipeline.

VERIFIED PREPROCESSING FACTS (do not change without re-verifying):
- The saved model (SmartInspect_EfficientNetB2_Final.keras) contains its OWN
  preprocessing pipeline inside the EfficientNetB2 sub-model:
    1. rescaling_1: scale=1/255 (float32 0-255 -> 0-1)
    2. normalization: ImageNet channel normalization
    3. rescaling_2: per-channel scaling
- Therefore, the correct inference input is raw uint8/float32 pixels in [0, 255].
- Do NOT apply tf.keras.applications.efficientnet.preprocess_input externally.
- Training used tf.keras.utils.image_dataset_from_directory which loads raw pixels (0-255).
- Image size is 260x260 pixels.
"""

import csv
import io
from collections import OrderedDict
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError


def load_pil_image(uploaded_file) -> Tuple[Optional[Image.Image], Optional[str]]:
    """
    Open a PIL image from a Streamlit UploadedFile object.

    Validates that:
      - The file is decodable as an image.
      - The image is at least 50x50 pixels.

    All color modes (RGBA, L, palette, etc.) are converted to RGB before returning.

    Returns
    -------
    (pil_img, None)   on success
    (None, error_msg) on failure
    """
    if uploaded_file is None:
        return None, "No file provided."

    try:
        uploaded_file.seek(0)
        pil_img = Image.open(uploaded_file)
        # Force decoding to catch corrupt files early
        pil_img.verify()
        # After verify() the file handle is exhausted; reopen from the raw bytes
        uploaded_file.seek(0)
        pil_img = Image.open(uploaded_file)
        pil_img.load()
    except UnidentifiedImageError:
        return None, (
            "The uploaded file could not be identified as an image. "
            "Please upload a valid JPG, PNG, or BMP file."
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"Failed to open image: {exc}"

    # Minimum size check
    width, height = pil_img.size
    if width < 50 or height < 50:
        return None, (
            f"Image is too small ({width}×{height} px). "
            "Please upload an image that is at least 50×50 pixels."
        )

    # Normalise color mode to RGB
    if pil_img.mode != "RGB":
        try:
            pil_img = pil_img.convert("RGB")
        except Exception as exc:  # noqa: BLE001
            return None, f"Could not convert image to RGB: {exc}"

    return pil_img, None


def preprocess_for_inference(
    pil_image: Image.Image, img_size: Tuple[int, int] = (260, 260)
) -> np.ndarray:
    """
    Prepare a PIL RGB image for model inference.

    Steps
    -----
    1. Resize to *img_size* using high-quality LANCZOS resampling.
    2. Convert to a float32 NumPy array — values stay in **[0, 255]**.
    3. Add a batch dimension → shape (1, H, W, 3).

    WHY no external normalisation?
    --------------------------------
    The saved Keras model (SmartInspect_EfficientNetB2_Final.keras) embeds its
    own preprocessing layers inside the EfficientNetB2 sub-model:
      • rescaling_1  : divides by 255  (maps [0,255] → [0,1])
      • normalization: subtracts ImageNet channel means, divides by stds
      • rescaling_2  : per-channel affine scaling
    Applying tf.keras.applications.efficientnet.preprocess_input externally
    would double-normalise the inputs and produce wrong predictions.

    Parameters
    ----------
    pil_image : PIL.Image.Image
        An RGB image (already validated/converted by load_pil_image).
    img_size : tuple of int
        Target (width, height). Defaults to (260, 260).

    Returns
    -------
    np.ndarray
        Shape (1, img_size[1], img_size[0], 3), dtype float32, values in [0, 255].
    """
    resized = pil_image.resize(img_size, Image.LANCZOS)
    arr = np.array(resized, dtype=np.float32)   # shape (H, W, 3), values 0-255
    batch = np.expand_dims(arr, axis=0)         # shape (1, H, W, 3)
    return batch


def parse_class_name(class_name: str) -> Tuple[str, str]:
    """
    Safely parse a "product_status" label into its component parts.

    Handles the special case where the product name itself contains underscores
    (e.g. "metal_nut_defective" → product="metal_nut", status="defective").

    Rules
    -----
    - If *class_name* ends with "_defective":
        product = everything before the last "_defective" suffix
        status  = "defective"
    - If *class_name* ends with "_good":
        product = everything before the last "_good" suffix
        status  = "good"
    - Otherwise:
        product = class_name
        status  = "unknown"

    Parameters
    ----------
    class_name : str
        Raw label string from class_names list (e.g. "metal_nut_defective").

    Returns
    -------
    (product_str, status_str)
    """
    if class_name.endswith("_defective"):
        product = class_name[: -len("_defective")]
        return product, "defective"
    if class_name.endswith("_good"):
        product = class_name[: -len("_good")]
        return product, "good"
    return class_name, "unknown"


def format_product_name(product_str: str) -> str:
    """
    Convert an internal product identifier to a human-readable display name.

    Examples
    --------
    "metal_nut" → "Metal Nut"
    "BSD"       → "BSD"   (already all-uppercase, title() preserves it)
    "bottle"    → "Bottle"
    "toothbrush"→ "Toothbrush"

    Parameters
    ----------
    product_str : str
        Internal product string, potentially containing underscores.

    Returns
    -------
    str
        Display-friendly product name.
    """
    return product_str.replace("_", " ").title()


def get_top_predictions(
    probabilities: np.ndarray,
    class_names: List[str],
    n: int = 3,
) -> List[Tuple[str, str, str, float]]:
    """
    Return the top-n predictions sorted by confidence descending.

    Parameters
    ----------
    probabilities : np.ndarray
        1-D array of softmax probabilities (one per class).
    class_names : list of str
        Ordered list of class label strings.
    n : int
        Number of top predictions to return.

    Returns
    -------
    List of (class_name, product, status, confidence_pct) tuples.
    """
    n = min(n, len(class_names))
    top_indices = np.argsort(probabilities)[::-1][:n]
    results = []
    for idx in top_indices:
        cname = class_names[idx]
        product, status = parse_class_name(cname)
        confidence_pct = float(probabilities[idx]) * 100.0
        results.append((cname, product, status, confidence_pct))
    return results


def is_low_confidence(confidence_pct: float, threshold: float = 60.0) -> bool:
    """
    Return True when the top-1 confidence falls below *threshold* percent.

    Parameters
    ----------
    confidence_pct : float
        Confidence as a percentage (0–100).
    threshold : float
        Minimum acceptable confidence. Defaults to 60.0.

    Returns
    -------
    bool
    """
    return confidence_pct < threshold


def prepare_csv_row(
    filename: str,
    product: str,
    status: str,
    predicted_class: str,
    confidence_pct: float,
    is_low_conf: bool,
    timestamp: Optional[str] = None,
) -> OrderedDict:
    """
    Build an OrderedDict representing one inspection result row.

    Parameters
    ----------
    filename : str
    product : str
        Formatted product display name.
    status : str
        "good", "defective", or "unknown".
    predicted_class : str
        Raw class label (e.g. "metal_nut_defective").
    confidence_pct : float
        Top-1 confidence percentage.
    is_low_conf : bool
        Whether the result falls below the confidence threshold.
    timestamp : str or None
        ISO 8601 timestamp string. If None, uses current UTC time.

    Returns
    -------
    OrderedDict with keys in display order.
    """
    if timestamp is None:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    return OrderedDict([
        ("timestamp", timestamp),
        ("filename", filename),
        ("product", product),
        ("status", status),
        ("predicted_class", predicted_class),
        ("confidence_pct", f"{confidence_pct:.2f}"),
        ("low_confidence_warning", "YES" if is_low_conf else "NO"),
    ])


def results_to_csv_string(rows: List[OrderedDict]) -> str:
    """
    Serialise a list of inspection result rows to a CSV string.

    Uses csv.DictWriter with a consistent, ordered field list so that the
    output columns are always in the same order regardless of dict insertion
    order on different Python versions.

    Parameters
    ----------
    rows : list of OrderedDict
        Each dict produced by :func:`prepare_csv_row`.

    Returns
    -------
    str
        UTF-8 encoded CSV text (with CRLF line endings as per RFC 4180).
    """
    fieldnames = [
        "timestamp",
        "filename",
        "product",
        "status",
        "predicted_class",
        "confidence_pct",
        "low_confidence_warning",
    ]

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames,
        extrasaction="ignore",
        lineterminator="\r\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()
