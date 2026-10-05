from pathlib import Path
import os
MODEL_ROOT = Path(os.environ.get("PUPU_MODEL_DIR", Path(__file__).resolve().parent.parent / "models")).resolve()
