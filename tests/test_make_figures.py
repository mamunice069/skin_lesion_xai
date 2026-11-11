"""scripts/make_figures.py: skips figures whose inputs are missing, and draws from synthetic run files."""
import importlib.util
import json

import pandas as pd

from skin_lesion_xai.config import PROJECT_DIR


def load():
    spec = importlib.util.spec_from_file_location("make_figures", PROJECT_DIR / "scripts" / "make_figures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_missing_inputs_are_skipped(tmp_path, monkeypatch, capsys):
    mod = load()
    monkeypatch.setattr(mod, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(mod, "CMP_DIR", tmp_path / "comparisons")
    figs = mod.Figures(tmp_path)
    for draw in mod.FIGURES.values():
        draw(figs)
    figs.write_captions()
    out = capsys.readouterr().out
    assert out.count("skipped") == len(mod.FIGURES)
    assert not list(tmp_path.glob("*.png"))


def test_classification_and_training_figures_from_synthetic_runs(tmp_path, monkeypatch):
    mod = load()
    runs = tmp_path / "runs"
    for k, (run, _, _) in enumerate(mod.HAM_RUNS[:2]):
        (runs / run / "eval").mkdir(parents=True)
        (runs / run / "train_summary.json").write_text(json.dumps({"best_val_score": 0.8, "best_epoch": 2}))
        (runs / run / "eval" / "metrics.json").write_text(json.dumps(
            {"test": {"macro_f1": 0.7 + 0.01 * k}, "test_ci95": {"macro_f1": {"low": 0.65, "high": 0.75}}}))
    hist = pd.DataFrame({"epoch": [1, 2, 3], "train_loss": [1.5, 1.2, 1.1], "val_loss": [1.4, 1.3, 1.35],
                         "val_macro_f1": [0.6, 0.8, 0.75]})
    (runs / "ham10000_convnext_t").mkdir(exist_ok=True)
    hist.to_csv(runs / "ham10000_convnext_t" / "history.csv", index=False)
    monkeypatch.setattr(mod, "RUNS_DIR", runs)
    figs = mod.Figures(tmp_path)
    mod.f1_classification(figs)
    mod.f8_training(figs)
    figs.write_captions()
    for name in ("F1_ham10000_classification", "F8_training_curves"):
        assert (tmp_path / f"{name}.png").stat().st_size > 0 and (tmp_path / f"{name}.pdf").exists()
    assert "F8_training_curves" in (tmp_path / "captions.md").read_text(encoding="utf-8")
