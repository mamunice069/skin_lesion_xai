"""LaTeX tables for the report (report/tables/*.tex), generated only from files the runs wrote.

Usage: python scripts/make_report_tables.py [--out report/tables]

Each file holds one fully bordered tabular with a bold header row (the style of the previous IEEE report; no float,
no caption: those stay in report/main.tex) and starts with a comment naming its source files. Run lists, comparison
files and the decision rule come from skin_lesion_xai/eval/results.py, as in scripts/compare_results.py. A run or
comparison that does not exist yet is written as a \\pending{...} row. The tables use these macros from main.tex:
\\better, \\worse (significantly better / worse for B after Holm), \\expl (EXPLORATORY mark), \\pending{text}.
  t0_splits   t1_ham_classification   t2_ham_paired   t3a_isic   t3b_isic_paired (vs ConvNeXt-T and vs EffNetV2-S)
  t4_localisation (condensed)   t4_localisation_full   t5_trust_test (condensed)   t5_trust_{val,test}_full
  t6a_step4   t6b_step5_selection   t6c_step5   t7_training (from the config files)
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import yaml

from skin_lesion_xai.config import PROJECT_DIR
from skin_lesion_xai.eval.metrics import holm
from skin_lesion_xai.eval.results import (
    CMP_DIR, EXPLORATORY, HAM_PAIRED, HAM_RUNS, ISIC_PAIRED, ISIC_PAIRED_LOCAL, ISIC_RUNS, M4_STEM, MAP_LABELS, MIN_P,
    RUNS_DIR, STEP4, STEP4_METHODS, STEP5, XAI_RUNS, decision_rule, direction, holm_bound, is_nan, read_json, rel,
)

SPLITS_DIR = PROJECT_DIR / "data" / "splits"
ROLE_TEX = {"baseline": "baseline", "baseline; global branch alone": "baseline", "baseline (ImageNet norm)": "baseline$^a$",
            "local branch alone (norm fix)": "local branch", "norm fix": "norm fix", "proposed": "\\textbf{proposed}",
            "fusion ablation": "ablation", "hair ablation": "hair abl.",
            "hair ablation; best on test, not pre-specified": "hair abl.$^b$",
            f"{EXPLORATORY} (Part B step 5)": "\\expl"}
SHORT_MAP = {"gradcam": "ConvNeXt-T", "gradcam_local": "local", "gradcam_fused": "fused"}
ENDPOINT_TEX = {"deletion_vs_random": "Deletion $-$ random", "insertion_vs_random": "Insertion $-$ random",
                "rand_full_ssim": "SSIM to randomised", "stab_spearman_mean": "Stability ($\\rho$)", "iou": "IoU",
                "iou_gap": "IoU gap (correct $-$ wrong)", "deletion_auc": "Deletion AUC", "insertion_auc": "Insertion AUC",
                "rand_full_abs_spearman": "$|\\rho|$ to randomised", "rand_full_abs_hog": "$|\\rho|$ HOG to randomised",
                "stab_ssim_mean": "Stability (SSIM)", "energy_in_mask": "Energy in mask", "pointing": "Pointing",
                "stab_spearman_no_noise": "Stability ($\\rho$, no noise)"}
ENDPOINT_SHORT = {"deletion_vs_random": "Del.", "insertion_vs_random": "Ins.", "rand_full_ssim": "Rand. SSIM",
                  "stab_spearman_mean": "Stab. $\\rho$", "iou": "IoU", "iou_gap": "IoU gap"}
T4_CONDENSED = [("ham10000_resnet50", "gradcam"), ("ham10000_convnext_t", "gradcam"), ("ham10000_swin_t", "gradcam"),
                ("ham10000_effnetv2_s_pnorm", "gradcam"), ("ham10000_vit_b16_pnorm_lr3e-5", "gradcam"),
                ("ham10000_hybrid", "gradcam_local"), ("ham10000_hybrid", "gradcam_global"),
                ("ham10000_hybrid", "gradcam_fused"), ("ham10000_hybrid", "xattn_local_to_global"),
                ("ham10000_hybrid", "xattn_global_to_local")]
CONFIGS = PROJECT_DIR / "configs"


# ----------------------------------------------------------------------------------------------- formatting
def esc(text: str) -> str:
    """Plain label text -> LaTeX (run names, map labels, comparison labels)."""
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"),
                 ("−", "$-$"), ("λ", "$\\lambda$"), ("μ", "$\\mu$"), ("≥", "$\\geq$"), ("→", "$\\rightarrow$"),
                 ("×", "$\\times$")):
        text = text.replace(a, b)
    return text


def num(v, d: int = 3) -> str:
    return "--" if is_nan(v) else f"${v:.{d}f}$"


def sgn(v, d: int = 3) -> str:
    return "--" if is_nan(v) else f"${v:+.{d}f}$"


def ci(c: dict | None, d: int = 3) -> str:
    if not c or is_nan(c.get("low")):
        return ""
    return f"$[{c['low']:.{d}f}, {c['high']:.{d}f}]$"


def pval(p, bound: float = MIN_P) -> str:
    if is_nan(p):
        return "--"
    return f"$<{bound:.3f}$" if p == 0 else f"${p:.3f}$"


def marked(r: dict | None, d: int = 3) -> str:
    if r is None:
        return "--"
    return {1: "\\better", -1: "\\worse", 0: ""}[direction(r)] + sgn(r["diff"], d)


def verdict(text: str) -> str:
    return {"met": "\\textbf{met}", "not met": "not met"}.get(text, text)


def mc(n: int, align: str, text: str, first: bool = False) -> str:
    """Bordered \\multicolumn cell (first=True when it starts the row)."""
    return f"\\multicolumn{{{n}}}{{{'|' if first else ''}{align}|}}{{{text}}}"


def bold(cell: str) -> str:
    return cell if (not cell or cell.startswith("\\multicolumn")) else f"\\textbf{{{cell}}}"


RULE = "\\hline"  # a raw row: a horizontal rule between groups of body rows


def bold_best(rows: list, cols: list[int], lower: set[int] = frozenset()) -> None:
    """Set the best value of each given column in bold (highest; lowest for columns in `lower`), as the related papers
    mark their best results. Cells are parsed from their first number; rows that are raw lines or have no number in
    that column are skipped. Ties are all bolded."""
    for col in cols:
        values = {}
        for i, r in enumerate(rows):
            if isinstance(r, str) or col >= len(r):
                continue
            m = re.search(r"[-+]?\d+\.\d+", r[col])
            if m:
                values[i] = float(m.group())
        if not values:
            continue
        best = (min if col in lower else max)(values.values())
        for i, v in values.items():
            if v == best:
                cell = rows[i][col]
                head, _, tail = cell.partition(" ")  # bold the value, not its bracketed CI
                rows[i][col] = f"{{\\boldmath {head}}}" + (f" {tail}" if tail else "")


class Writer:
    def __init__(self, out_dir: Path):
        self.out_dir, self.written = out_dir, []

    def table(self, name: str, colspec, header: list, rows: list, sources: list[str],
              size: str = "\\small", width: str | None = None, stack_ci: bool = False) -> None:
        """Boxed tabular in the style of the previous course report: outer frame and column rules, bold header rows
        each closed by a rule, body rows without rules between them. header / rows: lists of cells (a row), or a
        raw line such as '\\cline{2-4}' or RULE (a rule between groups of body rows).
        colspec: one letter per column, or a list of column specs. width: write a tabularx of exactly this width
        (e.g. '\\columnwidth'); its 'X' columns take the remaining width and wrap their text (left-aligned), so the
        table cannot overflow. stack_ci: put a cell's bracketed CI on a second line under its value (makecell)."""
        cols = list(colspec) if isinstance(colspec, str) else colspec
        cols = [">{\\raggedright\\arraybackslash}X" if c == "X" else c for c in cols]
        spec = "|" + "|".join(cols) + "|"
        begin = f"\\begin{{tabularx}}{{{width}}}{{{spec}}}" if width else f"\\begin{{tabular}}{{{spec}}}"
        end = "\\end{tabularx}" if width else "\\end{tabular}"
        if stack_ci:
            rows = [r if isinstance(r, str) else [re.sub(r"^(.*\$) (\$\[.*\]\$)$", r"\\makecell[l]{\1\\\\\2}", c)
                                                  for c in r] for r in rows]
        lines = ["% Generated by scripts/make_report_tables.py; do not edit by hand.",
                 "% Sources: " + ", ".join(sorted(set(sources))) if sources else "% Sources: none (pending)",
                 size, "\\setlength{\\tabcolsep}{3pt}", "\\renewcommand{\\arraystretch}{1.1}", begin, "\\hline"]
        for i, item in enumerate(header):
            if isinstance(item, str):
                lines.append(item)
                continue
            nxt = header[i + 1] if i + 1 < len(header) else None
            lines.append(" & ".join(bold(c) for c in item) + " \\\\" + ("" if isinstance(nxt, str) else " \\hline"))
        body = [r for k, r in enumerate(rows) if not (r == RULE and (k == 0 or k == len(rows) - 1))]
        for item in body:
            lines.append(item if isinstance(item, str) else " & ".join(item) + " \\\\")
        lines += ["\\hline", end]
        (self.out_dir / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.written.append(name)


def pending_row(text: str, ncols: int, first: str = "") -> list[str]:
    if first:
        return [first, mc(ncols - 1, "c", f"\\pending{{{text}}}")]
    return [mc(ncols, "c", f"\\pending{{{text}}}", first=True)]


# ----------------------------------------------------------------------------------------------- tables
def t0_splits(w: Writer) -> None:
    rows, src = [], []
    for name, title in (("ham10000", "HAM10000 (grouped by lesion)"), ("isic2020", "ISIC 2020 (grouped by patient)")):
        path = SPLITS_DIR / f"{name}_summary.csv"
        if not path.exists():
            continue
        src.append(rel(path))
        df = pd.read_csv(path)
        rows += [RULE, [mc(4, "l", f"\\textit{{{title}}}", first=True)], RULE]
        for _, r in df.iterrows():
            label = esc(str(r["label_name"]))
            label = f"\\textit{{{label}}}" if label.startswith(("total", "groups")) else label
            rows.append([label, *(f"{int(r[s]):,}".replace(",", "{,}") for s in ("train", "val", "test"))])
    w.table("t0_splits", "Xrrr", [["Class", "Train", "Val", "Test"]], rows, src, width="\\columnwidth")


def t1_ham(w: Writer) -> None:
    rows, src = [], []
    for run, label, role in HAM_RUNS:
        s, m = read_json(RUNS_DIR / run / "train_summary.json"), read_json(RUNS_DIR / run / "eval" / "metrics.json")
        if s is None or m is None:
            rows.append(pending_row(esc(label), 10, esc(label)))
            continue
        src += [rel(RUNS_DIR / run / "train_summary.json"), rel(RUNS_DIR / run / "eval" / "metrics.json")]
        t = m["test"]
        rows.append([esc(label), ROLE_TEX.get(role, esc(role)), str(m["checkpoint_epoch"]), num(s["best_val_score"], 4),
                     f"{num(t['macro_f1'], 4)} {ci(m['test_ci95']['macro_f1'], 4)}", num(t["balanced_accuracy"]),
                     num(t["accuracy"]), num(t["auc"]), num(t["mcc"]), num(t["per_class"]["mel"]["sensitivity"])])
    bold_best(rows, [3, 4, 5, 6, 7, 8, 9])
    w.table("t1_ham_classification", "Xlrrlrrrrr",
            [["Model", "Role", "Ep.", "Val F1", "Test macro-F1 [95\\% CI]", "Bal. acc.", "Acc.", "AUC", "MCC",
              "Mel. sens."]], rows, src, width="\\textwidth")


def t2_ham_paired(w: Writer) -> None:
    loaded = {stem: read_json(CMP_DIR / f"{stem}.json") for stem, _, _ in HAM_PAIRED}
    adj = holm({stem: loaded[stem]["metrics"]["macro_f1"]["p_two_sided"]
                for stem, _, fam in HAM_PAIRED if fam and loaded[stem]})
    rows, src = [], []
    for stem, label, fam in HAM_PAIRED:
        cmp = loaded[stem]
        name = esc(label.replace(f" ({EXPLORATORY})", "")) + ("\\expl" if fam is None else "")
        if cmp is None:
            rows.append(pending_row("not run", 4, name))
            continue
        src.append(rel(CMP_DIR / f"{stem}.json"))
        r = cmp["metrics"]["macro_f1"]
        rows.append([name, f"{sgn(r['diff'], 4)} {ci(r['diff_ci'], 4)}", pval(r["p_two_sided"]),
                     pval(adj[stem], 2 * MIN_P) if stem in adj else "--"])
    w.table("t2_ham_paired", "Xlrr", [["$B-A$", "$\\Delta$ macro-F1 [95\\% CI]", "$p$", "Holm $p$"]], rows, src,
            width="\\columnwidth", stack_ci=True)


def t3_isic(w: Writer) -> None:
    rows, src = [], []
    for run, label in ISIC_RUNS:
        m, s = read_json(RUNS_DIR / run / "eval" / "metrics.json"), read_json(RUNS_DIR / run / "train_summary.json")
        if m is None:
            rows.append(pending_row(f"{esc(label)} not evaluated yet", 8, esc(label)))
            continue
        src += [rel(RUNS_DIR / run / "eval" / "metrics.json")] + ([rel(RUNS_DIR / run / "train_summary.json")] if s else [])
        t, c, cut = m["test"], m["test_ci95"], m["test_at_val_thresholds"]
        pair = lambda mm: f"{num(mm['sensitivity'])}/{num(mm['specificity'])}"
        rows.append([esc(label), str(m["checkpoint_epoch"]), num(s["best_val_score"]) if s else "--",
                     f"{num(t['auc'])} {ci(c['auc'])}", f"{num(t['average_precision'])} {ci(c['average_precision'])}",
                     pair(t), pair(cut["youden"]["metrics"]), pair(cut["sensitivity_0.80"]["metrics"])])
    bold_best(rows, [2, 3, 4])
    w.table("t3a_isic", "lrrllccc",
            [["Model", "Ep.", "Val AUC", "Test AUC [95\\% CI]", "AP [95\\% CI]",
              mc(3, "c", "\\textbf{Sens./spec. at cutoff}")], "\\cline{6-8}",
             ["", "", "", "", "", "argmax", "Youden", "sens.\\,$\\geq$\\,0.80"]], rows, src)

    metrics = [("auc", "AUC"), ("average_precision", "AP"),
               ("sensitivity_0.80_specificity", "Spec. @ sens.\\,$\\geq$\\,0.80"),
               ("sensitivity_0.80_sensitivity", "Sens. @ sens.\\,$\\geq$\\,0.80"),
               ("youden_specificity", "Spec. @ Youden"), ("youden_sensitivity", "Sens. @ Youden"),
               ("argmax_specificity", "Spec. @ argmax"), ("argmax_sensitivity", "Sens. @ argmax")]
    # one full-width table, the hybrid against each comparator side by side (grouped headers, as in WA-NET Table 1)
    comparators = [(ISIC_PAIRED, "ConvNeXt-T"), (ISIC_PAIRED_LOCAL, "EfficientNetV2-S")]
    cmps = [read_json(CMP_DIR / f"{stem}.json") for stem, _ in comparators]
    header = [["", ""] + [mc(4, "c", f"\\textbf{{Hybrid $-$ {a}}}") for _, a in comparators], "\\cline{3-10}",
              ["Metric", "Hybrid", "ConvNeXt-T", "$\\Delta$", "[95\\% CI]", "$p$",
               "EffNetV2-S", "$\\Delta$", "[95\\% CI]", "$p$"]]
    rows = []
    for k, label in metrics:
        hyb = next((c["metrics"][k]["b"] for c in cmps if c), None)
        row = [label, num(hyb)]
        for c, (_, a) in zip(cmps, comparators):
            if c is None:
                row += [mc(4, "c", f"\\pending{{{a}}}")]
                continue
            r = c["metrics"][k]
            row += [num(r["a"]), sgn(r["diff"]), ci(r["diff_ci"]), pval(r["p_two_sided"])]
        rows.append(row)
    w.table("t3b_isic_paired", "lrrrlrrrlr", header, rows,
            [rel(CMP_DIR / f"{stem}.json") for (stem, _), c in zip(comparators, cmps) if c])


def t4_localisation(w: Writer) -> None:
    summaries, src = {}, []
    for run, label, _ in XAI_RUNS:
        s = read_json(RUNS_DIR / run / "xai_val" / "summary.json")
        r = read_json(RUNS_DIR / run / "xai_val_random" / "summary.json")
        if s:
            summaries[run] = (label, s, r)
            src += [rel(RUNS_DIR / run / "xai_val" / "summary.json")]
            src += [rel(RUNS_DIR / run / "xai_val_random" / "summary.json")] if r else []
    if not summaries:
        return

    def cells(run, key, full):
        label, s, r = summaries[run]
        a = s["maps"][key]["all"]
        ra = r["maps"].get(key, {}).get("all", {}) if r else {}
        name = key.split("@")[0]
        if name == "gradcam":
            text = esc(label)
        else:
            text = f"{esc(label.split(',')[0] if run == 'ham10000_hybrid' else label)}: {esc(MAP_LABELS.get(name, name))}"
        out = [text, num(a["iou"])] + ([num(a["dice"]), num(a["bbox_iou"])] if full else [])
        out += [num(a["pointing"]), num(a["energy_in_mask"]), num(ra.get("iou"))]
        return out + ([num(ra.get("pointing")), num(ra.get("energy_in_mask"))] if full else [])

    refs = next(iter(summaries.values()))[1]["maps"]

    def ref_rows(full):
        rows = [RULE]
        for key, label in (("ref_centre@none", "\\textit{Reference: centred Gaussian}"),
                           ("ref_uniform@none", "\\textit{Reference: uniform map}")):
            a = refs[key]["all"]
            rows.append([label, num(a["iou"])] + ([num(a["dice"]), num(a["bbox_iou"])] if full else [])
                        + [num(a["pointing"]), num(a["energy_in_mask"]), "--"] + (["--", "--"] if full else []))
        return rows

    rows = [cells(run, f"{m}@{'none' if m.startswith('xattn') else 'pred'}", False)
            for run, m in T4_CONDENSED if run in summaries]
    bold_best(rows, [1, 2, 3])
    w.table("t4_localisation", "Xrrrr", [["Model / map", "IoU", "Point.", "Energy", "Rand. IoU"]],
            rows + ref_rows(False), src, width="\\columnwidth")
    rows = [cells(run, key, True) for run in summaries for key in summaries[run][1]["maps"]
            if not key.startswith("ref_") and not key.endswith("@true")]
    w.table("t4_localisation_full", "lrrrrrrrr",
            [["", mc(5, "c", "\\textbf{Trained}"), mc(3, "c", "\\textbf{Random weights}")], "\\cline{2-9}",
             ["Model / map", "IoU", "Dice", "Box IoU", "Point.", "Energy", "IoU", "Point.", "Energy"]],
            rows + ref_rows(True), src)


def t5_trust(w: Writer) -> None:
    for split in ("val", "test"):
        path = CMP_DIR / f"{M4_STEM.format(split=split)}.json"
        cmp = read_json(path)
        if cmp is None:
            continue
        bound, maps, res = holm_bound(cmp), list(cmp["results"]), cmp["results"]

        def row(metric, full):
            first = res[maps[0]][metric]
            arrow = "$\\downarrow$" if first["better"] == "lower" else "$\\uparrow$"  # which direction is better
            label = ENDPOINT_TEX.get(metric, esc(metric)) + (f" {arrow}" if not full else "")
            out = [label + ("$^*$" if full and metric in cmp["primary"] else ""), num(first["a"])]
            for m in maps:
                r = res[m][metric]
                out += [num(r["b"]), f"{marked(r)} {ci(r['diff_ci'])}",
                        pval(r["p_holm"], bound) if "p_holm" in r else pval(r["p_two_sided"])]
            return out

        title = {m: f"\\textbf{{Hybrid, {SHORT_MAP[m]} map}}" for m in maps}
        p_head = "Holm $p$" if split == "test" else "$p$ (Holm)"
        header = [["", "", mc(3, "c", title[maps[0]]), mc(3, "c", title[maps[1]])], "\\cline{3-8}",
                  ["Endpoint", "ConvNeXt-T", "Hybrid", "$B-A$ [95\\% CI]", p_head, "Hybrid", "$B-A$ [95\\% CI]", p_head]]
        rule = [RULE, ["\\textit{Frozen decision rule}", "", mc(3, "c", verdict(decision_rule(res[maps[0]], cmp["primary"]))),
                 mc(3, "c", verdict(decision_rule(res[maps[1]], cmp["primary"])))]]
        if split == "test":
            w.table("t5_trust_test", "Xrrllrll", header, [row(m, False) for m in cmp["primary"]] + rule, [rel(path)],
                    width="\\textwidth")
        full_header = header[:2] + [header[2][:4] + ["$p$ (Holm)", "Hybrid", "$B-A$ [95\\% CI]", "$p$ (Holm)"]]
        w.table(f"t5_trust_{split}_full", "lrrllrll", full_header,
                [row(m, True) for m in res[maps[0]]] + rule, [rel(path)], size="\\scriptsize")


def _short_cmp(label: str) -> str:
    for key, short in (("hybrid vs ConvNeXt-T", " vs ConvNeXt-T"), ("hybrid:", " vs own Grad-CAM"),
                       ("ConvNeXt-T:", " ConvNeXt-T vs own Grad-CAM"), ("mask-guided vs hybrid", " vs source hybrid"),
                       ("mask-guided vs ConvNeXt-T", " vs ConvNeXt-T")):
        if key in label:
            return short
    return ""


def compact_marked(specs, fmt, label_fmt, prefix=None) -> tuple[list, list]:
    rows, src = [], []
    for label, stem, ruled in specs:
        text = label.format(**label_fmt)
        short = esc(text[:3] + _short_cmp(text))
        cmps = {}
        for split in ("val", "test"):
            path = CMP_DIR / f"{stem.format(split=split, **fmt)}.json"
            cmps[split] = read_json(path)
            src += [rel(path)] if cmps[split] else []
        first = cmps["val"] or cmps["test"]
        if first is None:
            rows.append(pending_row("not run", 10 + (prefix is not None), short))
            continue
        for m in first["results"]:
            for k, split in enumerate(("val", "test")):
                c = cmps[split]
                lead = ([prefix if k == 0 else ""] if prefix is not None else [])
                lead += [short if k == 0 else "", SHORT_MAP[m] if k == 0 else "", split]
                cells = [marked(c["results"][m].get(metric)) if c else "--" for metric in first["primary"]]
                rule = (verdict(decision_rule(c["results"][m], c["primary"])) if ruled else "descr.") if c else "--"
                rows.append(lead + cells + [rule])
            rows.append(RULE)
    return rows, src


def t6_part_b(w: Writer) -> None:
    head = ["Split", *ENDPOINT_SHORT.values(), "Rule"]
    rows, src = [], []
    for m, name in STEP4_METHODS:
        r, s = compact_marked(STEP4, {"m": m}, {"M": name}, prefix=esc(name))
        rows += r
        src += s
    w.table("t6a_step4", "lllcrrrrrrc", [["Method", "Comparison", "Map", *head]], rows, src, size="\\scriptsize")

    sel_rows, sel_src, seen = [], [], set()
    for stage in ("lambda", "consistency"):
        path = CMP_DIR / f"ham10000_mask_guided_{stage}.json"
        sel = read_json(path)
        if sel is None:
            continue
        sel_src.append(rel(path))
        cands = ([sel["source"] | {"eligible": None}] if stage == "lambda" else []) + sel["candidates"]
        for c in cands:
            if c["run"] in seen:
                continue
            seen.add(c["run"])
            lm = re.search(r"_l([\d.]+)_m([\d.]+)$", c["run"])
            lam, mu = (f"${lm.group(1)}$", f"${lm.group(2)}$") if lm else ("--", "--")
            sel_rows.append(["source" if lm is None else {"lambda": "1 ($\lambda$)", "consistency": "2 ($\mu$)"}[stage], lam, mu, num(c["val_macro_f1"], 4),
                             "--" if c["eligible"] is None else ("yes" if c["eligible"] else "no"),
                             num(c["insertion_vs_random"]), num(c["iou"]), "", c["run"]])
    if sel_rows:
        final = read_json(CMP_DIR / "ham10000_mask_guided_consistency.json")
        for r in sel_rows:
            if final and r[-1] == final.get("chosen"):
                r[-2] = "$\\checkmark$"
        sel_rows = [r[:-1] for r in sel_rows]  # drop the run name used for matching
        w.table("t6b_step5_selection", "Xrrrcrrc",
                [["Stage", "$\\lambda$", "$\\mu$", "Val F1", "Elig.", "Ins.\\,$-$\\,rand.", "IoU", "Chosen"]],
                sel_rows, sel_src, width="\\columnwidth")

    rows, src = compact_marked(STEP5, {}, {})
    w.table("t6c_step5", "llcrrrrrrc", [["Comparison", "Map", *head]], rows, src, size="\\scriptsize")


def t7_training(w: Writer) -> None:
    """Training settings, read from the config files (not results)."""
    paths = {k: CONFIGS / "train" / f"{k}.yaml" for k in ("default", "local_4gb", "kaggle_16gb", "mask_finetune")}
    cfg = {k: yaml.safe_load(p.read_text()) for k, p in paths.items()}
    hyb = yaml.safe_load((CONFIGS / "model" / "hybrid.yaml").read_text())
    data = yaml.safe_load((CONFIGS / "data" / "ham10000.yaml").read_text())
    d, loc, kag, ft = cfg["default"], cfg["local_4gb"], cfg["kaggle_16gb"], cfg["mask_finetune"]
    def sci(v) -> str:
        mantissa, exponent = f"{float(v):.0e}".split("e")
        return f"${mantissa}\\times10^{{{int(exponent)}}}$"

    rows = [
        RULE, [mc(2, "l", "\\textit{Optimisation (all models)}", first=True)], RULE,
        ["Optimiser", "AdamW"],
        ["Learning rate, backbone / fusion + head", f"{sci(d['backbone_lr'])} / {sci(d['head_lr'])}"],
        ["Weight decay", f"${d['weight_decay']}$"],
        ["Schedule", f"{d['warmup_epochs']} warm-up epochs, cosine to {d['min_lr_ratio']:g}$\\times$"],
        ["Maximum epochs; early stopping", f"{d['epochs']}; patience {d['early_stop_patience']} on val macro-F1 (AUC on ISIC)"],
        ["Loss", f"cross-entropy, label smoothing {d['label_smoothing']}, class weights (1/freq)$^{{{d['class_weight_power']}}}$"],
        ["Gradient clipping; EMA decay", f"{d['grad_clip']}; {d['ema_decay']}"],
        ["Mixed precision; seed", f"{'yes' if d['amp'] else 'no'}; {d['seed']}"],
        ["Input size", f"${data['image_size']}\\times{data['image_size']}$"],
        RULE, [mc(2, "l", "\\textit{Hardware profiles}", first=True)], RULE,
        ["Local (RTX 3050, 4 GB)", f"batch {loc['batch_size']} $\\times$ {loc['accum_steps']} accumulation, gradient checkpointing"],
        ["Kaggle (ISIC 2020)", f"batch {kag['batch_size']}, no accumulation"],
        RULE, [mc(2, "l", "\\textit{Hybrid fusion}", first=True)], RULE,
        ["Token dimension; layers; heads", f"{hyb['fusion_dim']}; {hyb['fusion_layers']}; {hyb['fusion_heads']}"],
        ["Dropout; drop path", f"{hyb['drop_rate']}; {hyb['drop_path_rate']}"],
        RULE, [mc(2, "l", "\\textit{Mask-guided fine-tuning (exploratory)}", first=True)], RULE,
        ["Epochs; learning rates", f"{ft['epochs']}; {sci(ft['backbone_lr'])} / {sci(ft['head_lr'])}, no warm-up"],
    ]
    w.table("t7_training", "XX", [["Setting", "Value"]], rows, [rel(p) for p in paths.values()]
            + [rel(CONFIGS / "model" / "hybrid.yaml"), rel(CONFIGS / "data" / "ham10000.yaml")], width="\\columnwidth")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="report/tables")
    args = p.parse_args()
    out = Path(args.out)
    out = out if out.is_absolute() else PROJECT_DIR / out
    out.mkdir(parents=True, exist_ok=True)
    w = Writer(out)
    for table in (t0_splits, t1_ham, t2_ham_paired, t3_isic, t4_localisation, t5_trust, t6_part_b, t7_training):
        table(w)
    print(f"wrote {len(w.written)} tables to {out}: {', '.join(w.written)}")


if __name__ == "__main__":
    main()
