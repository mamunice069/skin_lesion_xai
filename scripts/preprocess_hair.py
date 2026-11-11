"""Run DullRazor hair removal once over a dataset and cache the cleaned images.

Writes <hair_removed_dir>/<image_id>.jpg (same names as the originals) and a
before / hair-mask / after grid to <hair_removed_dir>/_samples.png for visual checking.
Existing outputs are skipped, so the script can be resumed. The DullRazor parameters are
stored in <hair_removed_dir>/_params.json; if the config changes, the script stops instead of
mixing images made with different parameters (pass --overwrite to clear the cache and redo it).

Usage: python scripts/preprocess_hair.py --config configs/data/ham10000.yaml [--workers 4] [--limit N]
Then set `use_hair_removal: true` in the data config to train on the cleaned images.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
from tqdm import tqdm

from skin_lesion_xai.config import DataConfig
from skin_lesion_xai.data.dullrazor import remove_hair
from skin_lesion_xai.data.io import read_rgb  # torch-free: each worker process imports this script
from skin_lesion_xai.data.splits import load_split_frame
from skin_lesion_xai.utils.logging import get_logger

log = get_logger("preprocess_hair")


def _process(job: tuple[str, str, dict]) -> str | None:
    src, dst, params = job
    if Path(dst).exists():
        return None
    try:
        cleaned = remove_hair(read_rgb(Path(src)), **params)
    except Exception as exc:  # report and continue; the summary lists failures
        return f"{src}: {exc}"
    if not cv2.imwrite(dst, cv2.cvtColor(cleaned, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95]):
        return f"{dst}: write failed"
    return None


def save_sample_grid(paths: list[Path], params: dict, out_file: Path) -> None:
    # imported here, not at module level: every worker process re-imports this script
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(paths), 3, figsize=(9, 3 * len(paths)))
    for row, path in zip(axes, paths):
        image = read_rgb(path)
        cleaned, mask = remove_hair(image, return_mask=True, **params)
        for ax, img, title in zip(row, (image, mask, cleaned), ("original", "hair mask", "DullRazor")):
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.set_title(f"{path.stem} - {title}", fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_file, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=3, help="keep low on machines with little RAM")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N images (for a quick check)")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true", help="delete cached images made with other parameters")
    args = parser.parse_args()

    cfg = DataConfig.from_yaml(args.config)
    src_root = cfg.raw_path(cfg.image_dir)
    out_root = cfg.project_path(cfg.hair_removed_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    params_file = out_root / "_params.json"
    cached = list(out_root.glob(f"*{cfg.image_ext}"))
    stored = json.loads(params_file.read_text()) if params_file.exists() else None
    if cached and stored != cfg.dullrazor:
        if not args.overwrite:
            raise SystemExit(f"{len(cached)} cached images in {out_root} were made with {stored}, "
                             f"config now has {cfg.dullrazor}. Pass --overwrite to delete and redo them.")
        for path in cached:
            path.unlink()
        log.info("deleted %d cached images made with %s", len(cached), stored)
    params_file.write_text(json.dumps(cfg.dullrazor, indent=2))

    ids = load_split_frame(cfg)["image_id"].tolist()[: args.limit]
    jobs = [(str(src_root / f"{i}{cfg.image_ext}"), str(out_root / f"{i}{cfg.image_ext}"), cfg.dullrazor) for i in ids]
    log.info("%s: %d images -> %s (params %s)", cfg.name, len(jobs), out_root, cfg.dullrazor)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        errors = [e for e in tqdm(pool.map(_process, jobs, chunksize=32), total=len(jobs)) if e]
    for e in errors:
        log.error(e)
    log.info("done, %d failures", len(errors))

    sample_paths = [src_root / f"{i}{cfg.image_ext}" for i in ids[: args.n_samples]]
    save_sample_grid(sample_paths, cfg.dullrazor, out_root / "_samples.png")
    log.info("sample grid: %s", out_root / "_samples.png")


if __name__ == "__main__":
    main()
