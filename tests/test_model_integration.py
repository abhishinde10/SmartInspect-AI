"""
Integration tests for SmartInspect AI — model loading and inference.

These tests load the actual saved Keras model and verify end-to-end behaviour.

All tests in this module are SKIPPED automatically if:
  - The model file is not present at model/SmartInspect_EfficientNetB2_Final.keras
  - TensorFlow cannot be imported

This allows the test suite to run in CI environments without the large model
file, while still providing full validation when the model is available.

Run only these tests:
    pytest tests/test_model_integration.py -v

Skip reason is reported clearly for each skipped test.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent.parent
MODEL_PATH = PROJECT_DIR / "model" / "SmartInspect_EfficientNetB2_Final.keras"
CLASS_NAMES_PATH = PROJECT_DIR / "model" / "class_names.json"

sys.path.insert(0, str(PROJECT_DIR / "app"))

# ---------------------------------------------------------------------------
# Conditional import of TensorFlow
# ---------------------------------------------------------------------------
try:
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault(
        "KERAS_HOME", str(PROJECT_DIR / ".keras_home")
    )
    import tensorflow as tf  # noqa: E402

    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# ---------------------------------------------------------------------------
# Shared skip markers
# ---------------------------------------------------------------------------
skip_no_tf = pytest.mark.skipif(
    not TF_AVAILABLE,
    reason="TensorFlow is not importable in this environment.",
)

skip_no_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=f"Model file not found: {MODEL_PATH}",
)

skip_no_classnames = pytest.mark.skipif(
    not CLASS_NAMES_PATH.exists(),
    reason=f"class_names.json not found: {CLASS_NAMES_PATH}",
)

# Composed: skip if either model or class-names is absent
requires_model = pytest.mark.skipif(
    not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists() or not TF_AVAILABLE,
    reason=(
        "Skipped: requires TensorFlow AND model/SmartInspect_EfficientNetB2_Final.keras "
        "AND model/class_names.json to be present."
    ),
)

# ---------------------------------------------------------------------------
# Fixture: loaded class names (no TF needed)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def class_names():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Fixture: loaded Keras model (TF needed)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    return tf.keras.models.load_model(str(MODEL_PATH), compile=False)


# ===========================================================================
# Tests: class_names.json (no model, no TF needed)
# ===========================================================================


@skip_no_classnames
class TestClassNamesJson:
    """Validate the class-names JSON file without loading the model."""

    def test_file_exists(self):
        assert CLASS_NAMES_PATH.exists()

    def test_is_valid_json_list(self):
        with open(CLASS_NAMES_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        assert isinstance(data, list)
        assert len(data) > 0

    def test_exactly_20_classes(self):
        with open(CLASS_NAMES_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        assert len(data) == 20, f"Expected 20 classes, got {len(data)}"

    def test_all_entries_are_strings(self):
        with open(CLASS_NAMES_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        for entry in data:
            assert isinstance(entry, str), f"Non-string entry: {entry!r}"

    def test_all_classes_end_with_good_or_defective(self):
        with open(CLASS_NAMES_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        for name in data:
            assert name.endswith("_good") or name.endswith("_defective"), (
                f"Class '{name}' does not end with '_good' or '_defective'"
            )

    def test_all_10_products_represented(self):
        expected_products = {
            "bottle", "BSD", "cable", "carpet", "grid",
            "leather", "metal_nut", "toothbrush", "wood", "zipper",
        }
        with open(CLASS_NAMES_PATH, encoding="utf-8") as fh:
            data = json.load(fh)

        found_products = set()
        for name in data:
            if name.endswith("_good"):
                found_products.add(name[: -len("_good")])
            elif name.endswith("_defective"):
                found_products.add(name[: -len("_defective")])

        missing = expected_products - found_products
        assert not missing, f"Missing products in class_names.json: {missing}"

    def test_each_product_has_both_good_and_defective(self):
        with open(CLASS_NAMES_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        good_classes = {n[: -len("_good")] for n in data if n.endswith("_good")}
        defective_classes = {n[: -len("_defective")] for n in data if n.endswith("_defective")}
        assert good_classes == defective_classes, (
            f"Good/defective mismatch — only_good={good_classes - defective_classes}, "
            f"only_defective={defective_classes - good_classes}"
        )

    def test_expected_class_order(self):
        """Verify the exact class order matches training (alphabetical sorted)."""
        expected = [
            "BSD_defective", "BSD_good",
            "bottle_defective", "bottle_good",
            "cable_defective", "cable_good",
            "carpet_defective", "carpet_good",
            "grid_defective", "grid_good",
            "leather_defective", "leather_good",
            "metal_nut_defective", "metal_nut_good",
            "toothbrush_defective", "toothbrush_good",
            "wood_defective", "wood_good",
            "zipper_defective", "zipper_good",
        ]
        with open(CLASS_NAMES_PATH, encoding="utf-8") as fh:
            actual = json.load(fh)
        assert actual == expected, (
            "Class order does not match expected training order. "
            "This would cause silent misclassification."
        )


# ===========================================================================
# Tests: Model architecture (TF + model required)
# ===========================================================================


@requires_model
class TestModelArchitecture:
    """Verify the saved model's shape and layer structure."""

    def test_model_loads_without_error(self, model):
        assert model is not None

    def test_model_input_shape(self, model):
        assert model.input_shape == (None, 260, 260, 3), (
            f"Unexpected input shape: {model.input_shape}"
        )

    def test_model_output_shape(self, model):
        assert model.output_shape == (None, 20), (
            f"Unexpected output shape: {model.output_shape}"
        )

    def test_output_classes_match_class_names(self, model, class_names):
        num_outputs = model.output_shape[-1]
        assert num_outputs == len(class_names), (
            f"Model has {num_outputs} outputs but class_names.json has "
            f"{len(class_names)} entries."
        )

    def test_model_has_efficientnetb2_backbone(self, model):
        """Verify EfficientNetB2 is present as a sub-model."""
        layer_names = [layer.name for layer in model.layers]
        has_efficientnet = any("efficientnet" in n.lower() for n in layer_names)
        assert has_efficientnet, (
            f"No EfficientNet layer found. Layers: {layer_names}"
        )

    def test_model_has_preprocessing_inside(self, model):
        """
        Verify that the model contains its own preprocessing layers,
        meaning inference should use raw [0, 255] pixel values.
        """
        efficientnet = model.get_layer("efficientnetb2")
        inner_layer_names = [l.name for l in efficientnet.layers]
        has_rescaling = any("rescal" in n.lower() for n in inner_layer_names)
        assert has_rescaling, (
            "Expected a Rescaling layer inside EfficientNetB2 "
            "but none found. Preprocessing assumptions may be wrong."
        )


# ===========================================================================
# Tests: Single-image inference (TF + model required)
# ===========================================================================


@requires_model
class TestSingleImageInference:
    """End-to-end inference tests using synthetic images."""

    def _make_batch(self, value: float = 128.0) -> np.ndarray:
        """Create a uniform-colour (1, 260, 260, 3) float32 batch."""
        return np.full((1, 260, 260, 3), value, dtype=np.float32)

    def test_inference_returns_correct_shape(self, model):
        batch = self._make_batch()
        preds = model.predict(batch, verbose=0)
        assert preds.shape == (1, 20), f"Unexpected output shape: {preds.shape}"

    def test_probabilities_are_finite(self, model):
        batch = self._make_batch()
        preds = model.predict(batch, verbose=0)[0]
        assert np.all(np.isfinite(preds)), "Model output contains NaN or Inf values."

    def test_probabilities_sum_to_one(self, model):
        batch = self._make_batch()
        preds = model.predict(batch, verbose=0)[0]
        assert abs(preds.sum() - 1.0) < 1e-4, (
            f"Probabilities do not sum to 1.0: sum={preds.sum():.6f}"
        )

    def test_probabilities_are_non_negative(self, model):
        batch = self._make_batch()
        preds = model.predict(batch, verbose=0)[0]
        assert np.all(preds >= 0), "Some probability values are negative."

    def test_argmax_is_valid_index(self, model, class_names):
        batch = self._make_batch()
        preds = model.predict(batch, verbose=0)[0]
        idx = int(np.argmax(preds))
        assert 0 <= idx < len(class_names), (
            f"Predicted index {idx} is out of range [0, {len(class_names)})"
        )

    def test_preprocessing_mismatch_detection(self, model):
        """
        Verify that raw [0,255] inputs and pre-normalized [0,1] inputs
        produce DIFFERENT outputs, confirming the model handles its own
        preprocessing (and double-normalization would be wrong).
        """
        raw_batch = np.full((1, 260, 260, 3), 200.0, dtype=np.float32)
        norm_batch = raw_batch / 255.0  # simulate external preprocess_input

        preds_raw = model.predict(raw_batch, verbose=0)[0]
        preds_norm = model.predict(norm_batch, verbose=0)[0]

        max_diff = np.max(np.abs(preds_raw - preds_norm))
        assert max_diff > 0.01, (
            "Raw and pre-normalized inputs produce nearly identical outputs. "
            "This is unexpected if the model contains its own preprocessing. "
            f"Max diff: {max_diff:.6f}"
        )

    def test_black_image_does_not_crash(self, model):
        """A completely black image should not crash inference."""
        black_batch = np.zeros((1, 260, 260, 3), dtype=np.float32)
        preds = model.predict(black_batch, verbose=0)[0]
        assert preds.shape == (20,)
        assert np.all(np.isfinite(preds))

    def test_white_image_does_not_crash(self, model):
        """A completely white image should not crash inference."""
        white_batch = np.full((1, 260, 260, 3), 255.0, dtype=np.float32)
        preds = model.predict(white_batch, verbose=0)[0]
        assert preds.shape == (20,)
        assert np.all(np.isfinite(preds))


# ===========================================================================
# Tests: Preprocessing pipeline consistency (TF + model required)
# ===========================================================================


@requires_model
class TestPreprocessingPipeline:
    """Verify that the utils preprocessing pipeline produces valid model input."""

    def test_utils_preprocess_output_shape(self):
        from PIL import Image
        from utils import preprocess_for_inference

        img = Image.new("RGB", (400, 300), color=(128, 64, 32))
        batch = preprocess_for_inference(img, img_size=(260, 260))
        assert batch.shape == (1, 260, 260, 3)

    def test_utils_preprocess_pixel_range(self):
        from PIL import Image
        from utils import preprocess_for_inference

        img = Image.new("RGB", (100, 100), color=(200, 100, 50))
        batch = preprocess_for_inference(img)
        assert batch.min() >= 0.0
        assert batch.max() <= 255.0

    def test_full_pipeline_produces_finite_predictions(self, model):
        from PIL import Image
        from utils import preprocess_for_inference

        img = Image.new("RGB", (300, 300), color=(150, 100, 50))
        batch = preprocess_for_inference(img)
        preds = model.predict(batch, verbose=0)[0]
        assert np.all(np.isfinite(preds))
        assert abs(preds.sum() - 1.0) < 1e-4
