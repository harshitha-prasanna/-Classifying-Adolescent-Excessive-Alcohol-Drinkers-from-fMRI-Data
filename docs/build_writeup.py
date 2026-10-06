"""Build the two-page project write-up (docs/writeup.pdf) from results/.

    python docs/build_writeup.py

All numbers are read from results/summary.csv, results/summary_full_cohort.csv
and results/extra.json, so the PDF always matches the latest experiment run.
"""
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.fonts import addMapping
from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"
TEAM = "Harshitha P (PES1UG24CS186) &nbsp;·&nbsp; Gautam Prasanth Nair (PES1UG24CS169)"

summary = pd.read_csv(RES / "summary.csv")
full = pd.read_csv(RES / "summary_full_cohort.csv")
extra = json.loads((RES / "extra.json").read_text())


def get(model, parc="ica25", col="accuracy_mean", df=summary):
    r = df[(df.parcellation == parc) & (df.model == model)]
    return float(r[col].iloc[0]) if len(r) else float("nan")


def fmt(model, parc="ica25"):
    return f"{get(model, parc):.2f}"


# make <b>/<i> inside paragraphs map to the real Times bold/italic faces
addMapping("Times-Roman", 0, 0, "Times-Roman")
addMapping("Times-Roman", 1, 0, "Times-Bold")
addMapping("Times-Roman", 0, 1, "Times-Italic")
addMapping("Times-Roman", 1, 1, "Times-BoldItalic")
ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["BodyText"], fontName="Times-Roman", fontSize=9.2,
                      leading=11.3, alignment=TA_JUSTIFY, spaceAfter=3)
h = ParagraphStyle("h", parent=body, fontName="Times-Bold", fontSize=10.5, leading=12,
                   spaceBefore=4, spaceAfter=2, alignment=0, textColor=colors.HexColor("#1b3a6b"))
title = ParagraphStyle("title", parent=body, fontName="Times-Bold", fontSize=15, leading=18,
                       alignment=TA_CENTER, spaceAfter=2)
sub = ParagraphStyle("sub", parent=body, alignment=TA_CENTER, fontSize=9, spaceAfter=6)
cap = ParagraphStyle("cap", parent=body, fontSize=7.8, leading=9, alignment=TA_CENTER)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=9, bulletIndent=1, spaceAfter=1.5)


def P(text, style=body):
    return Paragraph(text, style)


def B(text):
    return Paragraph(text, bullet, bulletText="•")


def fig(name, width_mm):
    img = Image(str(FIG / name))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = width_mm * mm, width_mm * mm * ratio
    return img


perm = extra.get("permutation_ica25", {})
perm_c = extra.get("permutation_craddock100", {})
deep_models = ["R", "R+N", "C+N", "C+R+N", "C+R+N+demo"]
deep_best = max(deep_models, key=lambda m: get(m))
fc_full = get("FC-LR", col="auc_mean", df=full)
demo_full = get("Demo only", col="auc_mean", df=full)

story = [
    P("Classifying Adolescent Excessive Alcohol Drinkers from fMRI Data", title),
    P(f"UE24CS352A Machine Learning · Mini-Project #35 &nbsp;|&nbsp; {TEAM}", sub),

    P("1. Problem Statement", h),
    P("Heavy drinking during adolescence is known to alter brain <i>structure</i>, but whether it leaves a "
      "detectable signature in brain <i>function</i> is still open. The task is a binary classifier that "
      "predicts whether a 16–19-year-old is a <b>heavy drinker</b> from their resting-state fMRI "
      "(BOLD signal) and basic demographics. We replicate the models of the reference study (Kim, Liu &amp; "
      "Noh, Stanford CS229: logistic regression, SVMs, CNN/RNN). We then test the ideas it lists as future "
      "work: richer fMRI features, demographics inside the models and more rigorous validation."),

    P("2. Dataset", h),
    P("The reference study uses 715 NCANDA subjects. Each subject has T = 269 volumes (TR = 2.2 s), "
      "parcellated into N = 25 ICA components or N = 100 Craddock regions, plus age, sex and scanner type. "
      "122 subjects (17%) are heavy drinkers. NCANDA is released only under a data-use agreement, so we "
      "wrote a <b>documented simulator</b> with exactly this structure (m × T × N tensors, same class ratio, "
      "demographics). Each region follows one of five networks (DMN, salience, executive, visual, "
      "somatomotor) driven by slow AR(1) latent signals. On top of that the simulator adds a global signal, "
      "scanner-dependent noise, motion spikes and a raw scanner baseline. Heavy drinkers differ by small, "
      "explicit effects: weaker DMN/ECN coupling, stronger salience–DMN coupling, and higher prevalence with "
      "age. The code loads real data in the same <font face='Courier'>.npz</font> format without changes. "
      "<b>Our numbers describe how the methods behave on NCANDA-shaped data. They are not clinical "
      "findings.</b>"),

    P("3. Approach", h),
    B("<b>Pre-processing (as in the paper):</b> z-score each region's time series; undersample the majority "
      "class to a 50/50 set of 244 subjects; hold out a stratified 5% <b>dev set</b> (13 subjects) that never "
      "enters cross-validation; run 10-fold stratified CV on the remaining 231. We repeat the whole procedure "
      "over 5 random undersamplings and report mean ± sd."),
    B("<b>Features:</b> the paper's derived dynamic range <i>x(N) = max(x<sub>N</sub>) − min(x<sub>N</sub>)</i>; "
      "our <b>functional connectivity</b> (Fisher-z Pearson correlation of every region pair: 300 features "
      "for ICA-25, 4,950 for Craddock-100); demographics."),
    B("<b>Models:</b> logistic regression with <b>Newton's method</b> (from scratch, θ := θ − H<sup>−1</sup>∇<i>l</i>(θ), "
      "tol 1e-5); SVMs (linear, poly-2, sigmoid, RBF) with C/γ tuned by inner CV; PyTorch RNN, RNN+NN, CNN+NN "
      "and CNN→LSTM→Dense (window 5, stride 1, 5 filters, RNN dim 5), trained with batch BCE and the paper's "
      "three dev-set stopping rules. Extensions: L2-regularised LR on connectivity, a <b>late-fusion</b> model "
      "(averaged log-odds of a demographics LR and a connectivity LR), and a CNN+RNN that takes demographics "
      "as a second input."),
    B("<b>Validation beyond the paper:</b> ROC-AUC alongside accuracy/F1; age ablations; a full-cohort protocol "
      "(all 715 subjects, class-weighted, stratified 10-fold, balanced accuracy); a 200-permutation label "
      "test."),

    P("4. Implementation", h),
    P("Python package <font face='Courier'>fmri_alcohol</font>: <font face='Courier'>data</font> (simulator + "
      "loader), <font face='Courier'>preprocess</font>, <font face='Courier'>features</font>, "
      "<font face='Courier'>models/</font> (Newton LR, SVM/LR factories, PyTorch networks), "
      "<font face='Courier'>evaluate</font> (protocols, metrics) and <font face='Courier'>plots</font>. "
      "Scripts: <font face='Courier'>generate_data.py</font>, <font face='Courier'>run_experiments.py</font> "
      "(writes every table and figure to <font face='Courier'>results/</font>) and "
      "<font face='Courier'>demo.py</font> (live prediction on a held-out subject, with a what-if age option). "
      "15 pytest unit tests cover the pipeline. They check, among other things, that no subject is shared "
      "between train/test/dev splits and that our Newton LR matches scikit-learn's coefficients to 1e-4. All "
      "tuning happens inside training folds, so test folds are never used to choose hyper-parameters."),
]

# ---------- results table ----------
rows = [["Model", "Features", "ICA-25 Acc", "F1", "AUC", "Craddock Acc", "AUC"]]
table_models = [
    ("LR (paper baseline)", "LR", "range + demo"),
    ("LR − age", "LR - age", "range + sex, scanner"),
    ("LR fMRI only", "LR fMRI only", "range"),
    ("SVM-Linear", "SVM-L", "range + demo"),
    ("SVM-RBF", "SVM-RB", "range + demo"),
    ("RNN only", "R", "raw BOLD"),
    ("CNN+RNN+Dense", "C+R+N", "raw BOLD"),
    ("CNN+RNN+demo", "C+R+N+demo", "raw BOLD + demo"),
    ("CNN+RNN, no stop rules", "C+R+N (no stop)", "raw BOLD"),
    ("Age only", "Age only", "age"),
    ("FC-LR", "FC-LR", "conn + demo"),
    ("FC-LR − age", "FC-LR - age", "conn + sex, scanner"),
    ("FC-LR fMRI only", "FC-LR fMRI only", "conn"),
    ("FC + Demo fusion", "FC + Demo fusion", "conn ⊕ demo"),
]
for label, key, feats in table_models:
    rows.append([label, feats,
                 f"{get(key):.2f}±{get(key, col='accuracy_sd'):.2f}", f"{get(key, col='f1_mean'):.2f}",
                 f"{get(key, col='auc_mean'):.2f}", f"{get(key, 'craddock100'):.2f}",
                 f"{get(key, 'craddock100', 'auc_mean'):.2f}"])
best_row = max(range(1, len(rows)), key=lambda i: get(table_models[i - 1][1], "craddock100", "auc_mean"))
tbl = Table(rows, colWidths=[31 * mm, 30 * mm, 20 * mm, 11 * mm, 11 * mm, 20 * mm, 11 * mm])
style = [
    ("FONT", (0, 0), (-1, -1), "Times-Roman", 7.8),
    ("FONT", (0, 0), (-1, 0), "Times-Bold", 7.8),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe7f3")),
    ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
    ("LINEABOVE", (0, 6), (-1, 6), 0.4, colors.grey),
    ("LINEABOVE", (0, 10), (-1, 10), 0.4, colors.grey),
    ("BACKGROUND", (0, best_row), (-1, best_row), colors.HexColor("#fff3c4")),
    ("FONT", (0, best_row), (-1, best_row), "Times-Bold", 7.8),
    ("ALIGN", (2, 0), (-1, -1), "CENTER"),
    ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
]
tbl.setStyle(TableStyle(style))

story += [
    P("5. Results", h),
    tbl,
    P("Table 1. 10-fold CV on the balanced set (mean over folds × 5 undersamplings; deep models 1 × 10 "
      "folds). Rows: paper models / deep models / our extensions. Highlighted: best model on both parcellations.", cap),
    Spacer(1, 3),
    Table([[fig("ablation_ica25.png", 96), fig("roc_ica25.png", 70)]],
          colWidths=[100 * mm, 76 * mm], style=[("VALIGN", (0, 0), (-1, -1), "MIDDLE")]),
    P("Figure 1. Left: demographics vs. fMRI features (mean per-fold AUC). Right: ROC curves pooled over the "
      "test folds of one repeat, so AUCs differ slightly from Table 1 (ICA-25).", cap),
    Spacer(1, 2),
    B(f"<b>Replication.</b> As in the paper, LR on derived range features + demographics is a reasonable "
      f"baseline (acc {fmt('LR')}), but removing age drops it to {fmt('LR - age')}, and range features alone "
      f"give {fmt('LR fMRI only')} (chance). Age alone reaches {fmt('Age only')}, so the baseline's "
      f"performance comes almost entirely from demographics. SVMs perform about the same "
      f"(RBF {fmt('SVM-RB')}, linear {fmt('SVM-L')})."),
    B(f"<b>Deep learning.</b> All deep models stay at chance (best: {deep_best}, {fmt(deep_best)}). With "
      f"the paper's stopping rules, the 13-subject dev set ends training after about 10 epochs on noise, and "
      f"the models often predict one class (C+R+N recall {get('C+R+N', col='recall_mean'):.2f}). With the "
      f"rules turned off, C+R+N reaches train accuracy {get('C+R+N (no stop)', col='train_accuracy_mean'):.2f} "
      f"but test {fmt('C+R+N (no stop)')}: it overfits. Adding demographics ({fmt('C+R+N+demo')}) does not "
      f"help. About 200 subjects are too few for these networks, which matches the reference study."),
    B(f"<b>Connectivity carries signal that range features miss.</b> Without age, FC-LR keeps "
      f"{fmt('FC-LR - age')} (range features: {fmt('LR - age')}). fMRI-only FC gives {fmt('FC-LR fMRI only')} "
      f"on ICA-25 and {fmt('FC-LR fMRI only', 'craddock100')} on Craddock-100 (permutation p = "
      f"{perm.get('p_value', float('nan')):.3f} and {perm_c.get('p_value', float('nan')):.3f}). Late fusion is "
      f"the best model: acc {fmt('FC + Demo fusion')} / AUC {get('FC + Demo fusion', col='auc_mean'):.2f} "
      f"(ICA-25) and <b>{fmt('FC + Demo fusion', 'craddock100')} / "
      f"{get('FC + Demo fusion', 'craddock100', 'auc_mean'):.2f}</b> (Craddock-100). Regularisation decides "
      f"the parcellation result. Unregularised Newton LR on Craddock's 103 range features overfits (train "
      f"{get('LR', 'craddock100', 'train_accuracy_mean'):.2f}, test {fmt('LR', 'craddock100')}). L2-penalised "
      f"FC copes with 4,950 features, but its shared penalty then ignores the demographic columns "
      f"(FC-LR {fmt('FC-LR', 'craddock100')} ≈ FC-LR − age {fmt('FC-LR - age', 'craddock100')}). Fusion "
      f"fixes this by giving demographics their own model."),
    B(f"<b>Full cohort.</b> Using all 715 subjects with class weights instead of discarding 471 controls, "
      f"FC-LR gets AUC {fc_full:.2f} vs. {demo_full:.2f} for demographics only (balanced accuracy "
      f"{get('FC-LR', col='balanced_accuracy_mean', df=full):.2f}). Undersampling throws away useful data."),

    P("6. Conclusions", h),
    P("We rebuilt the reference study end to end and reproduced its main result: complex deep models "
      "underperform a simple logistic regression, and that regression depends on age rather than on brain "
      "activity. The fix is better <i>features</i>, not deeper models. Between-region functional connectivity, "
      "a strong L2 penalty, and a separate model for demographics recover an fMRI signal that stays "
      "significant after removing age and passes a permutation test. Methodologically, a 13-subject dev set "
      "is too small to drive early stopping; repeated CV with AUC and all subjects is more reliable than a "
      "single undersampled split. Next steps: run the same pipeline on real NCANDA data (it loads as-is), "
      "regress out age/site confounds explicitly, try graph neural networks on the connectivity matrix, and "
      "use transfer learning from large resting-state cohorts to overcome the small sample size."),
    P("<b>Code:</b> private GitHub repository <font face='Courier'>fmri-alcohol-classifier</font> "
      "(README has setup and run instructions). <b>Ref.:</b> Kim, Liu, Noh, <i>Classifying Adolescent "
      "Excessive Alcohol Drinkers from fMRI Data</i>, CS229 Stanford; NCANDA (ncanda.org); Craddock et al., "
      "Hum. Brain Mapp. 2012.", ParagraphStyle("small", parent=body, fontSize=8, leading=9.5)),
]

out = ROOT / "docs/writeup.pdf"
doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                        topMargin=12 * mm, bottomMargin=12 * mm,
                        title="Classifying Adolescent Excessive Alcohol Drinkers from fMRI Data",
                        author="Harshitha P, Gautam Prasanth Nair")
doc.build(story)
print(f"wrote {out}")
