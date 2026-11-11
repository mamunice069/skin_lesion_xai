"""Kaggle script: train one model on ISIC 2020 with the project code (paste into a Kaggle script, GPU + internet on).

Inputs to attach (Kaggle "Add Input"):
  - private dataset "skin-lesion-xai-code": the zip made by the packaging script (src/, configs/, scripts/,
    pyproject.toml, requirements.txt, data/splits/isic2020*.csv, kaggle/)
  - private dataset "isic2020-256-local": an unchanged upload of datasets/isic2020/ (images/ + train-metadata.csv)
  - to continue a run that hit the time limit: the output of the previous version of this script

What it does:
  1. logs the GPU, pins timm / albumentations / scikit-learn to the local versions (torch stays Kaggle's);
  2. copies the code to /kaggle/working/skin_lesion_xai (outputs land in its outputs/runs/);
  3. checks that every image id in the split CSV exists;
  4. copies the previous version's run folder if one is attached and adds --resume;
  5. runs scripts/train.py with configs/train/kaggle_16gb.yaml as a subprocess with a time limit below the 12 h
     session limit. On the limit it is stopped; last.pt (saved atomically after every epoch) holds the last
     finished epoch, the session ends normally so the output is kept, and the next version resumes;
  6. deletes last.pt when the run has finished, so the output is best.pt + logs.
The recipe is configs/train/default.yaml (same as HAM10000); only the hardware profile differs.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

MODEL = "effnetv2_s_pnorm"  # "convnext_t", "hybrid" or "effnetv2_s_pnorm" (configs/model/<MODEL>.yaml)
SMOKE = False  # True: 1 epoch on a small subset to check GPU memory and speed (run name smoke_...)
TIME_LIMIT_H = 11.0  # stop training after this many hours (Kaggle sessions end at 12 h)

PINNED = {"timm": "1.0.30", "albumentations": "2.0.8", "scikit-learn": "1.9.1"}  # versions used locally
LOCAL_TORCH = "2.6.0+cu124"
INPUT, WORK = Path("/kaggle/input"), Path("/kaggle/working")
CODE = WORK / "skin_lesion_xai"
DATA_YAML = "configs/data/isic2020_kaggle.yaml"
RUN = f"smoke_isic2020_{MODEL}_kaggle" if SMOKE else f"isic2020_{MODEL}"


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def installed_version(dist: str) -> str | None:
    from importlib import metadata
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def pin_packages() -> dict:
    for dist, version in PINNED.items():
        if installed_version(dist) != version:
            log(f"installing {dist}=={version} (found {installed_version(dist)})")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"{dist}=={version}"], check=True)
    check = "import torch, timm, albumentations, sklearn, cv2; print(torch.__version__, torch.cuda.is_available())"
    out = subprocess.run([sys.executable, "-c", check], check=True, capture_output=True, text=True).stdout.split()
    versions = {d: installed_version(d) for d in PINNED} | {"torch": out[0]}
    if out[1] != "True":
        sys.exit("no CUDA GPU: set Accelerator to GPU in the notebook settings")
    if versions["torch"] != LOCAL_TORCH:
        log(f"note: torch {versions['torch']} here vs {LOCAL_TORCH} locally (results are not bit-identical anyway)")
    return versions


def find_one(pattern: str, what: str) -> Path:
    hits = sorted(INPUT.glob(pattern))
    if not hits:
        sys.exit(f"cannot find {what} under {INPUT} (pattern {pattern}); is the input attached?")
    if len(hits) > 1:
        log(f"several {what} found, using {hits[0]}: {[str(h) for h in hits]}")
    return hits[0]


def setup_code() -> None:
    src = find_one("**/scripts/train.py", "the project code").parents[1]
    if CODE.exists():
        shutil.rmtree(CODE)
    shutil.copytree(src, CODE, ignore=shutil.ignore_patterns("outputs", "__pycache__", ".venv"))
    log(f"code copied from {src}")


def setup_data() -> None:
    import yaml

    cfg_path = CODE / DATA_YAML
    cfg = yaml.safe_load(cfg_path.read_text())
    root = Path(cfg["data_root"])
    if not (root / cfg["image_dir"]).is_dir():
        found = find_one(f"**/{cfg['metadata_csv']}", "train-metadata.csv").parent
        if not (found / cfg["image_dir"]).is_dir():
            sys.exit(f"{found} has {cfg['metadata_csv']} but no {cfg['image_dir']}/ folder")
        log(f"data found at {found} instead of {root}; updating {DATA_YAML}")
        cfg_path.write_text(cfg_path.read_text().replace(f"data_root: {root.as_posix()}", f"data_root: {found.as_posix()}"))
        root = found
    import pandas as pd
    split = pd.read_csv(CODE / cfg["split_csv"])
    images = root / cfg["image_dir"]
    missing = [i for i in split["image_id"] if not (images / f"{i}{cfg['image_ext']}").exists()]
    if missing:
        sys.exit(f"{len(missing)} of {len(split)} split images missing in {images}, e.g. {missing[:3]}")
    log(f"all {len(split)} split images present in {images}")


def previous_run() -> Path | None:
    """The run folder from an attached output of an earlier version, if any."""
    hits = sorted(INPUT.glob(f"**/outputs/runs/{RUN}/config.json"))
    return hits[0].parent if hits else None


def main() -> None:
    started = time.time()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    log(f"GPU: {gpu or 'none'}  |  run {RUN}")
    versions = pin_packages()
    setup_code()
    setup_data()

    run_dir = CODE / "outputs" / "runs" / RUN
    args = ["scripts/train.py", "--data", DATA_YAML, "--model", f"configs/model/{MODEL}.yaml",
            "--profile", "configs/train/kaggle_16gb.yaml", "--run-name", RUN]
    if SMOKE:
        args += ["--epochs", "1", "--max-train", "500", "--max-val", "200"]
    prev = previous_run()
    if prev is not None:
        if (prev / "train_summary.json").exists():
            log(f"{RUN} already finished in the attached output ({prev}); nothing to do")
            return
        shutil.copytree(prev, run_dir)
        args.append("--resume")
        log(f"continuing {RUN} from {prev}")

    env = os.environ | {"PYTHONPATH": str(CODE / "src"), "PYTHONUNBUFFERED": "1"}
    budget = TIME_LIMIT_H * 3600 - (time.time() - started)
    log(f"python {' '.join(args)}  (time limit {budget / 3600:.2f} h)")
    proc = subprocess.Popen([sys.executable, *args], cwd=CODE, env=env)
    try:
        code = proc.wait(timeout=budget)
        status = "finished" if (run_dir / "train_summary.json").exists() else f"exited with code {code}"
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
        status = "stopped at the time limit; attach this output to the next version to resume"
    log(f"{RUN}: {status}")

    if (run_dir / "train_summary.json").exists() and (run_dir / "last.pt").exists():
        (run_dir / "last.pt").unlink()  # finished: best.pt is the model; last.pt is only for resuming
    if run_dir.exists():
        sessions = run_dir / "kaggle_sessions.json"
        history = json.loads(sessions.read_text()) if sessions.exists() else []
        history.append({"started": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
                        "ended": datetime.now().isoformat(timespec="seconds"), "gpu": gpu,
                        "versions": versions, "resumed": prev is not None, "status": status})
        sessions.write_text(json.dumps(history, indent=2))
        print((run_dir / "train.log").read_text()[-3000:] if (run_dir / "train.log").exists() else "")


if __name__ == "__main__":
    main()
