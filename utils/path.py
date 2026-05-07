from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def get_model_path(filename):
    return BASE_DIR / "Models" / filename