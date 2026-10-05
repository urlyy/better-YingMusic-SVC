"""Repository paths and optional overrides for a developer-managed environment."""
from pathlib import Path
import os
import sys

WORKSPACE = Path(__file__).resolve().parent.parent
ENGINE = Path(os.environ.get('PUPU_ENGINE_DIR', WORKSPACE / 'inference')).resolve()

MODEL_ROOT = Path(os.environ.get('PUPU_MODEL_DIR', WORKSPACE / 'models')).resolve()

def python_path():
    override = os.environ.get('PUPU_PYTHON')
    if override:
        return Path(override).resolve()
    return Path(sys.executable).resolve()

def hub_cache():
    override = os.environ.get('PUPU_HF_CACHE') or os.environ.get('HF_HUB_CACHE')
    if override:
        return Path(override).expanduser().resolve()
    return MODEL_ROOT / 'huggingface/hub'
