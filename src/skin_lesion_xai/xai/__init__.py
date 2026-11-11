"""Explanations (Grad-CAM, attention maps) and their localisation scores against lesion masks."""
from pathlib import Path


def xai_dir(run_dir: Path, split: str, random_weights: bool = False, smoke: bool = False) -> Path:
    """<run>/xai_<split>[_random][_smoke]: shared by scripts/explain.py and scripts/eval_xai.py."""
    return run_dir / f"xai_{split}{'_random' if random_weights else ''}{'_smoke' if smoke else ''}"


def trust_dir(run_dir: Path, split: str, smoke: bool = False, method: str = "gradcam") -> Path:
    """<run>/trust_<split>[_<method>][_smoke]: written by scripts/eval_trust.py (no suffix for Grad-CAM, M4)."""
    return run_dir / f"trust_{split}{'' if method == 'gradcam' else '_' + method}{'_smoke' if smoke else ''}"
