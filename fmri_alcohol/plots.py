"""Figures for the report and slides."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import auc, roc_curve  # noqa: E402

BLUE, ORANGE, GREY, GREEN, RED = "#2a6fdb", "#e8743b", "#9aa3ad", "#2e9e6a", "#d64545"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "savefig.bbox": "tight",
})


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_example_timeseries(X, y, demo, path, n_regions=6):
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.6), sharey=True)
    t = np.arange(X.shape[1]) * 2.2
    for ax, label, title in zip(axes, (0, 1), ("Non-heavy drinker", "Heavy drinker")):
        s = int(np.flatnonzero(y == label)[0])
        for r in range(n_regions):
            ax.plot(t, X[s, :, r] + 4 * r, lw=0.7)
        d = demo.iloc[s]
        ax.set_title(f"{title} ({'M' if d.sex else 'F'}, age {d.age:.1f})")
        ax.set_xlabel("time (s)")
        ax.set_yticks([])
    axes[0].set_ylabel("z-scored BOLD (offset per region)")
    return _save(fig, path)


def plot_histograms(X_raw, path):
    fig, axes = plt.subplots(1, 2, figsize=(6, 2.3))
    for ax, s in zip(axes, (0, 1)):
        ax.hist(X_raw[s].ravel(), bins=60, color=BLUE, alpha=0.8)
        ax.set_title(f"Subject {s}: BOLD signal histogram")
        ax.set_xlabel("raw BOLD")
    return _save(fig, path)


def plot_class_balance(y, path):
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    counts = [int((y == 0).sum()), int((y == 1).sum())]
    ax.bar(["Non-heavy", "Heavy"], counts, color=[GREY, ORANGE])
    for i, c in enumerate(counts):
        ax.text(i, c + 8, f"{c} ({100 * c / len(y):.0f}%)", ha="center")
    ax.set_ylabel("subjects")
    ax.set_title("Class distribution (m = %d)" % len(y))
    ax.set_ylim(0, max(counts) * 1.15)
    return _save(fig, path)


def plot_model_comparison(summary: pd.DataFrame, path, title="", metric="accuracy"):
    s = summary.copy()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    x = np.arange(len(s))
    w = 0.4
    ax.bar(x - w / 2, s[f"{metric}_mean"], w, yerr=s[f"{metric}_sd"], color=BLUE,
           label=metric.capitalize(), capsize=2, error_kw={"lw": 0.7})
    ax.bar(x + w / 2, s["f1_mean"], w, yerr=s["f1_sd"], color=ORANGE, label="F1",
           capsize=2, error_kw={"lw": 0.7})
    ax.axhline(0.5, color="k", lw=0.8, ls="--", label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(s["model"], rotation=40, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(ncol=3, loc="upper right", frameon=False)
    return _save(fig, path)


def plot_confusion(tp, fp, tn, fn, path, title="Confusion matrix"):
    M = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(2.8, 2.5))
    ax.imshow(M, cmap="Blues")
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(M[i, j])), ha="center", va="center",
                    color="white" if M[i, j] > M.max() / 2 else "black", fontsize=11)
    ax.set_xticks([0, 1], ["Non-heavy", "Heavy"])
    ax.set_yticks([0, 1], ["Non-heavy", "Heavy"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    return _save(fig, path)


def plot_roc(preds: pd.DataFrame, models, path, title="ROC (pooled test folds)"):
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    for m in models:
        p = preds[preds.model == m]
        fpr, tpr, _ = roc_curve(p.y, p.p)
        ax.plot(fpr, tpr, lw=1.3, label=f"{m} (AUC {auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    return _save(fig, path)


def plot_training_curves(history: pd.DataFrame, path, title="CNN+RNN training"):
    fig, ax = plt.subplots(figsize=(4, 2.6))
    h = history[(history.repeat == 0) & (history.fold == 0)]
    ax.plot(h.epoch, h.train_acc, color=BLUE, label="train")
    ax.plot(h.epoch, h.dev_acc, color=ORANGE, label="dev")
    ax.axhline(0.6, color=GREY, ls=":", lw=0.8, label="threshold 0.6")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.3, 1.0)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=7)
    return _save(fig, path)


def plot_connectivity_difference(C_hd, C_ctrl, networks, path):
    order = np.argsort(networks, kind="stable")
    diff = (C_hd - C_ctrl)[np.ix_(order, order)]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), layout="constrained")
    im0 = axes[0].imshow(C_ctrl[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_title("Mean FC: non-heavy drinkers")
    v = np.abs(diff).max()
    im1 = axes[1].imshow(diff, cmap="PuOr_r", vmin=-v, vmax=v)
    axes[1].set_title("FC difference (heavy - non-heavy)")
    nets = np.asarray(networks)[order]
    ticks = [np.flatnonzero(nets == n).mean() for n in dict.fromkeys(nets)]
    for ax in axes:
        ax.grid(False)
        ax.set_xticks(ticks, list(dict.fromkeys(nets)), fontsize=7, rotation=90)
        ax.set_yticks(ticks, list(dict.fromkeys(nets)), fontsize=7)
    fig.colorbar(im0, ax=axes[0], shrink=0.85, label="Pearson r")
    fig.colorbar(im1, ax=axes[1], shrink=0.85, label="Δr")
    return _save(fig, path)


def plot_ablation(summary: pd.DataFrame, path, title="Effect of demographics"):
    fig, ax = plt.subplots(figsize=(5, 3.0))
    s = summary

    def family(m):
        if m.startswith("FC"):
            return ORANGE
        if m.startswith("LR"):
            return BLUE
        return GREY

    ax.barh(s.model, s.auc_mean, xerr=s.auc_sd, color=[family(m) for m in s.model], capsize=2,
            error_kw={"lw": 0.7})
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GREY, label="demographics only"),
                       Patch(color=BLUE, label="range features (paper)"),
                       Patch(color=ORANGE, label="connectivity (ours)")],
              fontsize=6.5, frameon=False, loc="upper right")
    ax.axvline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("ROC-AUC (mean ± sd over folds)")
    ax.invert_yaxis()
    ax.set_title(title)
    return _save(fig, path)


def plot_permutation(null_scores, score, pvalue, path):
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    ax.hist(null_scores, bins=25, color=GREY)
    ax.axvline(score, color=RED, lw=1.5, label=f"observed {score:.2f} (p={pvalue:.3f})")
    ax.set_xlabel("CV accuracy with shuffled labels")
    ax.set_ylabel("count")
    ax.set_title("Permutation test")
    ax.legend(frameon=False, fontsize=7)
    return _save(fig, path)
