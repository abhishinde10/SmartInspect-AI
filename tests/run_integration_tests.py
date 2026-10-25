"""
Standalone integration test runner — runs model tests without needing
pytest installed in the venv. Uses the venv's TensorFlow directly.

Run from project root:
    .venv\\Scripts\\python.exe tests/run_integration_tests.py
"""

import json
import os
import sys
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ.setdefault("KERAS_HOME", str(Path(__file__).parent.parent / ".keras_home"))

# Ensure stdout uses UTF-8 regardless of console code page (Windows fix)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_DIR = Path(__file__).parent.parent
MODEL_PATH = PROJECT_DIR / "model" / "SmartInspect_EfficientNetB2_Final.keras"
CLASS_NAMES_PATH = PROJECT_DIR / "model" / "class_names.json"

sys.path.insert(0, str(PROJECT_DIR / "app"))

PASS_MARK = "  PASS"
FAIL_MARK = "  FAIL"

results = []


def assert_(cond, msg=""):
    if not cond:
        raise AssertionError(msg)


def run_test(name, fn):
    try:
        fn()
        results.append((name, "PASS", None))
        print(f"{PASS_MARK}  {name}")
    except AssertionError as e:
        results.append((name, "FAIL", str(e)))
        print(f"{FAIL_MARK}  {name}\n       {e}")
    except Exception as e:
        results.append((name, "FAIL", str(e)))
        print(f"{FAIL_MARK}  {name}\n       {type(e).__name__}: {e}")


# --- Class names tests (no TF) -----------------------------------------------

print("\n=== CLASS NAMES JSON ===")

with open(CLASS_NAMES_PATH, encoding="utf-8") as fh:
    class_names = json.load(fh)

run_test("class_names.json exists", lambda: assert_(CLASS_NAMES_PATH.exists()))
run_test("Exactly 20 classes", lambda: assert_(len(class_names) == 20, f"Got {len(class_names)}"))
run_test("All strings", lambda: [assert_(isinstance(n, str)) for n in class_names])
run_test("All end _good or _defective", lambda: [
    assert_(n.endswith("_good") or n.endswith("_defective"), f"Bad: {n}")
    for n in class_names
])

expected_order = [
    "BSD_defective", "BSD_good", "bottle_defective", "bottle_good",
    "cable_defective", "cable_good", "carpet_defective", "carpet_good",
    "grid_defective", "grid_good", "leather_defective", "leather_good",
    "metal_nut_defective", "metal_nut_good", "toothbrush_defective", "toothbrush_good",
    "wood_defective", "wood_good", "zipper_defective", "zipper_good",
]
run_test("Class order matches training", lambda: assert_(class_names == expected_order))

# --- TF model tests ----------------------------------------------------------

print("\n=== MODEL LOADING ===")

try:
    import tensorflow as tf
    import numpy as np
    TF_OK = True
    print("  TensorFlow version:", tf.__version__)
except Exception as e:
    TF_OK = False
    print(f"  WARNING: TensorFlow import failed: {e}")

if TF_OK:
    try:
        model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        MODEL_OK = True
        print(f"  Model loaded: {MODEL_PATH.name}")
    except Exception as e:
        MODEL_OK = False
        model = None
        print(f"  FAIL: Model load failed: {e}")

    if MODEL_OK:
        print("\n=== MODEL ARCHITECTURE ===")
        run_test("Input shape (None,260,260,3)",
                 lambda: assert_(model.input_shape == (None, 260, 260, 3),
                                  f"Got {model.input_shape}"))
        run_test("Output shape (None,20)",
                 lambda: assert_(model.output_shape == (None, 20),
                                  f"Got {model.output_shape}"))
        run_test("20 outputs match 20 class names",
                 lambda: assert_(model.output_shape[-1] == len(class_names),
                                  f"{model.output_shape[-1]} vs {len(class_names)}"))

        layer_names = [lyr.name for lyr in model.layers]
        run_test("EfficientNetB2 backbone present",
                 lambda: assert_(any("efficientnet" in n.lower() for n in layer_names),
                                  f"Layers: {layer_names}"))

        effnet = model.get_layer("efficientnetb2")
        inner_names = [lyr.name for lyr in effnet.layers]
        run_test("Rescaling layer inside EfficientNetB2 (own preprocessing)",
                 lambda: assert_(any("rescal" in n.lower() for n in inner_names),
                                  f"Inner layers: {inner_names}"))

        print("\n=== INFERENCE ===")
        batch_128 = np.full((1, 260, 260, 3), 128.0, dtype=np.float32)

        def test_inference_shape():
            p = model.predict(batch_128, verbose=0)
            assert_(p.shape == (1, 20), f"Got {p.shape}")
        run_test("Inference output shape (1,20)", test_inference_shape)

        def test_finite():
            p = model.predict(batch_128, verbose=0)[0]
            assert_(np.all(np.isfinite(p)), f"Non-finite values found")
        run_test("All probabilities finite", test_finite)

        def test_sum():
            p = model.predict(batch_128, verbose=0)[0]
            assert_(abs(p.sum() - 1.0) < 1e-4, f"Sum={p.sum():.6f}")
        run_test("Probabilities sum to ~1.0", test_sum)

        def test_nonneg():
            p = model.predict(batch_128, verbose=0)[0]
            assert_(np.all(p >= 0), "Negative values found")
        run_test("All probabilities non-negative", test_nonneg)

        def test_preprocessing_mismatch():
            raw = np.full((1, 260, 260, 3), 200.0, dtype=np.float32)
            norm = raw / 255.0
            p_raw = model.predict(raw, verbose=0)[0]
            p_norm = model.predict(norm, verbose=0)[0]
            diff = np.max(np.abs(p_raw - p_norm))
            assert_(diff > 0.01,
                    f"Raw vs normalized outputs too similar (diff={diff:.6f}). "
                    "Model may NOT handle its own preprocessing.")
        run_test("Raw [0-255] != pre-normalized [0-1] (confirms internal preprocessing)",
                 test_preprocessing_mismatch)

        def test_black():
            p = model.predict(np.zeros((1, 260, 260, 3), dtype=np.float32), verbose=0)[0]
            assert_(p.shape == (20,) and np.all(np.isfinite(p)))
        run_test("Black image inference succeeds", test_black)

        print("\n=== PREPROCESSING PIPELINE (utils.py) ===")
        from PIL import Image
        from utils import preprocess_for_inference, parse_class_name, get_top_predictions

        def test_utils_shape():
            img = Image.new("RGB", (400, 300))
            b = preprocess_for_inference(img)
            assert_(b.shape == (1, 260, 260, 3), f"Got {b.shape}")
        run_test("preprocess_for_inference output shape", test_utils_shape)

        def test_utils_range():
            img = Image.new("RGB", (200, 200), (200, 100, 50))
            b = preprocess_for_inference(img)
            assert_(b.min() >= 0 and b.max() <= 255.0,
                    f"Range: [{b.min()}, {b.max()}]")
        run_test("preprocess_for_inference pixel range [0,255]", test_utils_range)

        def test_full_pipeline():
            img = Image.new("RGB", (300, 300), (150, 100, 50))
            b = preprocess_for_inference(img)
            p = model.predict(b, verbose=0)[0]
            assert_(np.all(np.isfinite(p)) and abs(p.sum() - 1.0) < 1e-4)
        run_test("Full pipeline: PIL -> preprocess -> predict -> valid probs", test_full_pipeline)

# --- Summary -----------------------------------------------------------------

print("\n" + "=" * 60)
passed = sum(1 for _, s, _ in results if s == "PASS")
failed = sum(1 for _, s, _ in results if s == "FAIL")
print(f"RESULTS: {passed} passed, {failed} failed out of {len(results)} tests")
if failed == 0:
    print("All tests passed!")
else:
    print("Some tests failed. See details above.")
    sys.exit(1)
