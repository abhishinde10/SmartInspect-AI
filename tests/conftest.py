import os
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ.setdefault("KERAS_HOME", str(Path(__file__).parent.parent / ".keras_home"))
