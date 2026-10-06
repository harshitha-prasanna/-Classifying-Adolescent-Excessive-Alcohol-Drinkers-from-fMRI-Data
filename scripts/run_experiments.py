"""Run every experiment, then write result tables, figures and the demo model.

    python scripts/run_experiments.py                 # full run (default config)
    python scripts/run_experiments.py --quick         # ~2 min smoke run
    python scripts/run_experiments.py --skip-deep     # classical models only
"""
import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, permutation_test_score, train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from fmri_alcohol import plots  # noqa: E402
from fmri_alcohol.data import SimulationConfig, load_cohort, save_cohort, simulate_cohort  # noqa: E402
from fmri_alcohol.evaluate import (ProtocolA, run_deep_model, run_feature_model,  # noqa: E402
                                   run_full_cohort, summarise)
from fmri_alcohol.features import build_features, connectivity_matrices  # noqa: E402
from fmri_alcohol.models.classical import LateFusion, l2_logistic, newton_lr, svm  # noqa: E402
from fmri_alcohol.models.deep import DeepClassifier  # noqa: E402
from fmri_alcohol.preprocess import balance_undersample, zscore_timeseries  # noqa: E402

N_REGIONS = {"ica25": 25, "craddock100": 100}
DL_NAMES = {"rnn": "R", "rnn_nn": "R+N", "cnn_nn": "C+N", "cnn_rnn": "C+R+N",
            "cnn_rnn_demo": "C+R+N+demo"}
NO_AGE = ("sex", "scanner")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_cohort(data_dir: Path, name: str, seed: int):
    path = data_dir / f"{name}.npz"
    if not path.exists():
        log(f"{path} not found - simulating it")
        save_cohort(simulate_cohort(SimulationConfig(n_regions=N_REGIONS[name], seed=seed), name=name), path)
    return load_cohort(path)


def feature_experiments(parc, Xz, demo, y, protocol):
    """(label, feature-matrix, model factory, group) for every feature model."""
    F = lambda brain, d=("age", "sex", "scanner"): build_features(Xz, demo, brain, d)  # noqa: E731
    exps = [
        # --- reference-study replication (derived dynamic-range features) ---
        ("LR", F("range"), newton_lr, "replication"),
        ("LR - age", F("range", NO_AGE), newton_lr, "replication"),
        ("LR fMRI only", F("range", ()), newton_lr, "replication"),
    ]
    for k in ("L", "P", "S", "RB"):
        exps.append((f"SVM-{k}", F("range"), (lambda k=k: svm(k)), "replication"))
    # --- extensions ---
    exps += [
        ("Demo only", F(None), newton_lr, "extension"),
        ("Age only", F(None, ("age",)), newton_lr, "extension"),
        ("FC-LR", F("conn"), l2_logistic, "extension"),
        ("FC-LR - age", F("conn", NO_AGE), l2_logistic, "extension"),
        ("FC-LR fMRI only", F("conn", ()), l2_logistic, "extension"),
        ("FC-SVM-L", F("conn"), (lambda: svm("L")), "extension"),
        ("FC + Demo fusion", F("conn"), LateFusion, "extension"),
    ]
    return exps


def cached(cache_dir: Path, key: str, fn):
    """Run ``fn`` once and keep its output on disk, so an interrupted run
    resumes where it stopped (clear with --fresh)."""
    path = cache_dir / (re.sub(r"[^A-Za-z0-9_.-]+", "_", key) + ".joblib")
    if path.exists():
        return joblib.load(path)
    out = fn()
    joblib.dump(out, path)
    return out


def train_demo_model(Xz, demo, y, res_dir: Path, seed: int = 0):
    """Final model for scripts/demo.py: hold out a stratified 20% of the whole
    cohort, then train the fusion model on a balanced subset of the rest."""
    rest, holdout = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    train = rest[balance_undersample(y[rest], seed=seed)]
    model = LateFusion().fit(build_features(Xz, demo, "conn")[train], y[train])
    joblib.dump({"model": model, "brain": "conn", "demographics": ["age", "sex", "scanner"],
                 "parcellation": "ica25", "train_subjects": np.sort(train),
                 "holdout_subjects": np.sort(holdout)}, res_dir / "demo_model.joblib")
    log(f"demo model: trained on {len(train)} subjects, {len(holdout)} held out "
        f"({int(y[holdout].sum())} heavy drinkers)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--quick", action="store_true", help="1 repeat, ICA-25 only, 30 DL epochs")
    ap.add_argument("--skip-deep", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore cached results from earlier runs")
    ap.add_argument("--demo-model-only", action="store_true", help="only (re)train results/demo_model.joblib")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.quick:
        cfg["parcellations"] = ["ica25"]
        cfg["protocol"]["repeats"] = 1
        cfg["deep"]["epochs"] = 30
        cfg["deep"]["no_stop_epochs"] = 20
        cfg["full_cohort"]["repeats"] = 1
        cfg["permutation_test"]["n_permutations"] = 30
    if args.skip_deep:
        cfg["deep"]["enabled"] = False

    data_dir = ROOT / cfg["data_dir"]
    res_dir = ROOT / cfg["results_dir"]
    fig_dir = res_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if args.demo_model_only:
        cohort = get_cohort(data_dir, "ica25", cfg["seed"])
        res_dir.mkdir(parents=True, exist_ok=True)
        train_demo_model(zscore_timeseries(cohort.X), cohort.demographics, cohort.y, res_dir)
        return
    cache_dir = res_dir / "cache" / ("quick" if args.quick else "full")
    if args.fresh:
        shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    protocol = ProtocolA(**cfg["protocol"])
    dl_protocol = ProtocolA(k=protocol.k, dev_frac=protocol.dev_frac, repeats=cfg["deep"]["repeats"])

    all_folds, all_preds, histories, full_rows = [], [], [], []
    extra = {}
    for parc in cfg["parcellations"]:
        cohort = get_cohort(data_dir, parc, cfg["seed"])
        log(cohort.summary())
        Xz = zscore_timeseries(cohort.X)
        y, demo = cohort.y, cohort.demographics

        if parc == cfg["parcellations"][0]:
            plots.plot_example_timeseries(Xz, y, demo, fig_dir / "example_timeseries.png")
            plots.plot_histograms(cohort.X, fig_dir / "bold_histograms.png")
            plots.plot_class_balance(y, fig_dir / "class_balance.png")
        C = connectivity_matrices(Xz)
        plots.plot_connectivity_difference(C[y == 1].mean(0), C[y == 0].mean(0),
                                           cohort.region_networks, fig_dir / f"fc_difference_{parc}.png")

        # ---------- feature-based models ----------
        for name, F, factory, group in feature_experiments(parc, Xz, demo, y, protocol):
            df, pr = cached(cache_dir, f"{parc}__{name}", lambda: run_feature_model(
                name, factory, F, y, protocol, parcellation=parc, group=group, n_features=F.shape[1]))
            all_folds.append(df)
            all_preds.append(pr)
            log(f"{parc:12s} {name:18s} acc={df.accuracy.mean():.3f} f1={df.f1.mean():.3f} "
                f"auc={df.auc.mean():.3f}  ({df.seconds.iloc[0]:.0f}s)")

        # ---------- deep models on raw z-scored time series ----------
        if cfg["deep"]["enabled"]:
            D = demo[["age", "sex", "scanner"]].to_numpy(float)
            dc = cfg["deep"]
            runs = [(a, DL_NAMES[a], dict(epochs=dc["epochs"], lr=dc["lr"], use_rules=True))
                    for a in dc["architectures"]]
            runs += [(a, DL_NAMES[a] + " (no stop)",
                      dict(epochs=dc["no_stop_epochs"], lr=dc["no_stop_lr"], use_rules=False))
                     for a in dc.get("no_stop_architectures", [])]
            for arch, label, kw in runs:
                factory = lambda seed, arch=arch, kw=kw: DeepClassifier(  # noqa: E731
                    arch, batch_size=dc["batch_size"], dropout=dc["dropout"], seed=seed, **kw)
                df, pr, hist = cached(cache_dir, f"{parc}__deep__{label}", lambda: run_deep_model(
                    label, factory, Xz, D if arch.endswith("demo") else None,
                    y, dl_protocol, parcellation=parc, group="deep"))
                all_folds.append(df)
                all_preds.append(pr)
                histories.append(hist.assign(parcellation=parc))
                log(f"{parc:12s} {label:18s} acc={df.accuracy.mean():.3f} f1={df.f1.mean():.3f} "
                    f"auc={df.auc.mean():.3f} epochs={df.epochs.mean():.0f} ({df.seconds.iloc[0]:.0f}s)")

        # ---------- protocol B: full imbalanced cohort ----------
        if cfg["full_cohort"]["enabled"]:
            for name, brain, d in [("Demo only", None, ("age", "sex", "scanner")),
                                   ("LR (range+demo)", "range", ("age", "sex", "scanner")),
                                   ("FC-LR", "conn", ("age", "sex", "scanner")),
                                   ("FC-LR fMRI only", "conn", ())]:
                F = build_features(Xz, demo, brain, d)
                df = cached(cache_dir, f"{parc}__full__{name}", lambda: run_full_cohort(
                    name, lambda: l2_logistic(class_weight="balanced"), F, y,
                    repeats=cfg["full_cohort"]["repeats"], parcellation=parc))
                full_rows.append(df)
                log(f"{parc:12s} [full cohort] {name:18s} bal_acc={df.balanced_accuracy.mean():.3f} "
                    f"auc={df.auc.mean():.3f}")

        # ---------- permutation test for the fMRI-only connectivity model ----------
        bal = balance_undersample(y, seed=0)
        F = build_features(Xz, demo, "conn", ())
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1e-3, max_iter=5000))
        score, null, p = cached(cache_dir, f"{parc}__permutation", lambda: permutation_test_score(
            pipe, F[bal], y[bal], cv=StratifiedKFold(10, shuffle=True, random_state=0),
            n_permutations=cfg["permutation_test"]["n_permutations"], random_state=0, n_jobs=-1))
        extra[f"permutation_{parc}"] = {"score": float(score), "p_value": float(p),
                                        "null_mean": float(null.mean()), "null_sd": float(null.std())}
        plots.plot_permutation(null, score, p, fig_dir / f"permutation_{parc}.png")
        log(f"{parc:12s} permutation test FC fMRI-only: acc={score:.3f} p={p:.4f}")

        if parc == "ica25":
            train_demo_model(Xz, demo, y, res_dir)

    folds = pd.concat(all_folds, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)
    folds.to_csv(res_dir / "fold_metrics.csv", index=False)
    preds.to_csv(res_dir / "predictions.csv", index=False)
    summary = summarise(folds, by=("parcellation", "group", "model"))
    summary.to_csv(res_dir / "summary.csv", index=False)
    if histories:
        pd.concat(histories).to_csv(res_dir / "deep_training_history.csv", index=False)
    if full_rows:
        full = pd.concat(full_rows, ignore_index=True)
        full_sum = summarise(full, by=("parcellation", "model"))
        full_sum.to_csv(res_dir / "summary_full_cohort.csv", index=False)
    (res_dir / "extra.json").write_text(json.dumps(extra, indent=2))

    # ---------- figures ----------
    for parc in cfg["parcellations"]:
        s = summary[summary.parcellation == parc]
        rep = s[s.group.isin(["replication", "deep"])]
        plots.plot_model_comparison(rep, fig_dir / f"models_{parc}.png",
                                    title=f"Reference-study models ({parc}), 10-fold CV")
        plots.plot_model_comparison(s, fig_dir / f"all_models_{parc}.png",
                                    title=f"All models ({parc}), 10-fold CV")
        abl = s[s.model.isin(["Age only", "Demo only", "LR", "LR - age", "LR fMRI only",
                              "FC-LR", "FC-LR - age", "FC-LR fMRI only", "FC + Demo fusion"])]
        plots.plot_ablation(abl, fig_dir / f"ablation_{parc}.png",
                            title=f"Demographics vs. fMRI features ({parc})")
        pp = preds[preds.parcellation == parc]
        roc_models = [m for m in ["Age only", "LR", "LR - age", "C+R+N", "FC-LR fMRI only",
                                  "FC + Demo fusion"] if m in set(pp.model)]
        plots.plot_roc(pp[pp.repeat == 0], roc_models, fig_dir / f"roc_{parc}.png")
        best = s.sort_values("accuracy_mean", ascending=False).iloc[0]
        f = folds[(folds.parcellation == parc) & (folds.model == best.model)]
        plots.plot_confusion(f.tp.sum(), f.fp.sum(), f.tn.sum(), f.fn.sum(),
                             fig_dir / f"confusion_best_{parc}.png",
                             title=f"{best.model}: pooled test folds")
        lr = folds[(folds.parcellation == parc) & (folds.model == "LR")]
        plots.plot_confusion(lr.tp.sum(), lr.fp.sum(), lr.tn.sum(), lr.fn.sum(),
                             fig_dir / f"confusion_LR_{parc}.png", title="LR baseline: pooled test folds")
    if histories:
        h = pd.concat(histories)
        for parc in h.parcellation.unique():
            hp = h[(h.parcellation == parc) & (h.model == "C+R+N")]
            if len(hp):
                plots.plot_training_curves(hp, fig_dir / f"training_curve_{parc}.png",
                                           title=f"C+R+N train vs dev accuracy ({parc})")

    cols = ["parcellation", "group", "model", "accuracy_mean", "accuracy_sd", "f1_mean",
            "auc_mean", "recall_mean", "specificity_mean", "train_accuracy_mean", "dev_accuracy_mean"]
    table = summary[cols].round(3)
    md = ["# Results (10-fold CV, mean over folds x repeats)\n", table.to_markdown(index=False)]
    if full_rows:
        md += ["\n\n# Full imbalanced cohort (class-weighted, stratified 10-fold)\n",
               full_sum[["parcellation", "model", "balanced_accuracy_mean", "auc_mean", "f1_mean",
                         "recall_mean", "specificity_mean"]].round(3).to_markdown(index=False)]
    md += ["\n\n# Permutation tests\n", "```json\n" + json.dumps(extra, indent=2) + "\n```\n"]
    (res_dir / "RESULTS.md").write_text("\n".join(md))
    log(f"done - see {res_dir / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
