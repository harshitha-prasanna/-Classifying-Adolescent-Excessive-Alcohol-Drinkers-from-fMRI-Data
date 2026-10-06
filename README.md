# Classifying Adolescent Excessive Alcohol Drinkers from fMRI Data

**UE24CS352A – Machine Learning · Mini-Project (Problem #35)**
Team:

| Name | SRN |
|---|---|
| Harshitha P | PES1UG24CS186 |
| Gautam Prasanth Nair | PES1UG24CS169 |

This project predicts whether an adolescent (16–19 y) is a **heavy drinker** from
their **resting-state fMRI** (BOLD signal, `T = 269` volumes, TR = 2.2 s,
parcellated into `N = 25` ICA components or `N = 100` Craddock regions) plus
demographics (age, sex, scanner). It re-implements the models from the
reference study (Kim, Liu & Noh, *CS229, Stanford*) and extends them:

| Reference study (replicated) | Our extensions |
|---|---|
| Logistic regression with **Newton's method** (from scratch) on derived *dynamic-range* features | **Functional-connectivity** (Fisher-z correlation) features with L2-regularised LR |
| SVMs – linear, poly-2, sigmoid, RBF | **Late fusion** of a demographics model + an fMRI model |
| RNN, RNN+NN, CNN+NN, CNN+RNN+Dense (PyTorch) with the paper's 3 dev-set stopping rules | **CNN+RNN+demographics** network (the paper's proposed future work) |
| Balanced undersampling, 5 % dev set, 10-fold CV | Repeated CV over 5 undersamplings, **ROC-AUC**, full-cohort class-weighted protocol, **permutation test**, age ablations |

> **About the data.** The NCANDA fMRI data used by the reference study is
> only released under a data-use agreement, so it is not public. We therefore
> built a **documented simulator** (`fmri_alcohol/data.py`) that reproduces the
> cohort's structure: 715 subjects, 122 (17 %) heavy drinkers, T = 269,
> ICA-25 / Craddock-100 parcellations, age/sex/scanner, scanner noise and
> motion spikes. Heavy drinkers differ by a *small, explicit* amount (weaker
> DMN/ECN coupling, stronger salience–DMN coupling) and drinking is more
> likely with age. **Results therefore show how the methods behave on
> NCANDA-shaped data. They are not clinical findings.** Real parcellated data
> can be dropped in with no code changes (see [Using real data](#using-real-data)).

---

## 1. Setup

Requires Python 3.10–3.12 (PyTorch wheels). If your system Python is newer
(e.g. 3.14) or you are on an Intel Mac, let `uv` fetch a matching interpreter:

```bash
pip install uv            # or: brew install uv
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Otherwise:

```bash
git clone <your-private-repo-url> fmri-alcohol-classifier
cd fmri-alcohol-classifier
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Run

```bash
python scripts/generate_data.py        # 1. create data/ica25.npz and data/craddock100.npz (~10 s)
python -m pytest -q tests              # 2. unit tests (15 tests)
python scripts/run_experiments.py      # 3. all experiments (~30 min on a laptop CPU; resumable)
python scripts/run_experiments.py --quick   #    or a ~3 min smoke run
python scripts/demo.py                 # 4. live demo on a held-out subject
python docs/build_writeup.py           # 5. rebuild the 2-page write-up PDF
```

`make data test experiments demo writeup` does the same if `make` is available.

### Live demo

```bash
python scripts/demo.py                          # random held-out subject
python scripts/demo.py --subject SIM-0042       # a specific subject
python scripts/demo.py --subject SIM-0042 --age 16.2   # what-if: change age
python scripts/demo.py --all                    # score every held-out subject
```

It prints the subject's demographics, `P(heavy drinker)`, the prediction and
the ground truth. It also saves `results/demo_subject.png` (BOLD traces and the
connectivity matrix).

## 3. Repository layout

```
fmri_alcohol/
  data.py            simulator (SimulationConfig) + .npz loader/saver
  preprocess.py      z-scoring, class balancing, dev split, stratified k-fold
  features.py        dynamic range (paper), temporal std, functional connectivity
  evaluate.py        protocols A (paper) and B (full cohort), metrics
  plots.py           all figures
  models/
    logistic_newton.py   Newton's-method logistic regression (from scratch)
    classical.py         SVMs, L2-LR, late fusion
    deep.py              RNN / CNN / CNN+RNN (+demo) in PyTorch + stopping rules
scripts/
  generate_data.py   build the datasets
  run_experiments.py run everything -> results/
  demo.py            live demonstration
configs/default.yaml experiment settings
tests/               pytest unit tests
results/             RESULTS.md, summary*.csv, fold_metrics.csv, figures/
docs/                write-up (PDF + generator)
```

## 4. Method

1. **Pre-processing.** Z-score each region's time series per subject. Balance
   classes by undersampling the majority class (244 subjects). Set aside a
   stratified 5 % **dev set** (13 subjects) that is never used in CV. Run
   **10-fold stratified CV** on the remaining 231. The whole procedure is
   repeated for 5 different undersamplings.
2. **Features.**
   * *Derived (paper):* `x_derived(N) = max(x_N) − min(x_N)` per region.
   * *Connectivity (ours):* Pearson correlation between every pair of regions,
     Fisher-z transformed: 300 features (ICA-25) or 4,950 (Craddock-100).
   * *Demographics:* age, sex, scanner type.
3. **Models.** See the table above. The SVM `C`/`gamma` and the LR L2
   strength are tuned by inner 3-fold CV inside each training fold, so the
   test fold is never used for tuning. Deep models train with Adam and batch
   BCE. They stop on the paper's rules: (1) train and dev accuracy within
   (train+dev)/20 and both > 0.6; (2) the gap is growing, train > 0.65 and
   train > dev; (3) train > 0.65 while dev < 0.55. When they stop on
   overfitting, they fall back to the best-dev epoch.
4. **Metrics.** Accuracy, F1, ROC-AUC, sensitivity, specificity, and balanced
   accuracy for the imbalanced full-cohort protocol.

## 5. Results

See [`results/RESULTS.md`](results/RESULTS.md) for every number and
`results/figures/` for the plots. A summary is in the write-up
(`docs/writeup.pdf`).

### Headline numbers (10-fold CV, balanced set, mean over 5 undersamplings)

| Model | Features | ICA-25 acc / AUC | Craddock-100 acc / AUC |
|---|---|---|---|
| LR, Newton (paper baseline) | range + demographics | 0.62 / 0.66 | 0.52 / 0.52 |
| LR − age | range + sex, scanner | 0.55 / 0.54 | 0.48 / 0.47 |
| SVM-RBF | range + demographics | 0.63 / 0.68 | 0.60 / 0.63 |
| CNN+RNN+Dense (paper's main model) | raw BOLD | 0.48 / 0.45 | 0.49 / 0.54 |
| Age only | age | 0.66 / 0.70 | 0.66 / 0.70 |
| FC-LR, fMRI only (ours) | connectivity | 0.61 / 0.67 | 0.64 / 0.70 |
| **FC + Demo late fusion (ours)** | connectivity ⊕ demographics | **0.67 / 0.75** | **0.70 / 0.76** |

* **The paper's finding reproduces.** The derived-feature LR is an age model:
  without age it drops to chance, and age alone does better.
* **Deep models stay at chance**, both with the paper's dev-set stopping
  rules and without them. Without the rules they overfit (train 0.84,
  test 0.49).
* **Connectivity carries real signal.** It survives removing age, and the
  permutation test gives p = 0.005 (ICA-25) and p = 0.010 (Craddock-100).
  Fusing it with a separate demographics model gives the best result.
* On the **full imbalanced cohort** (715 subjects, class weights), FC-LR
  reaches AUC 0.73 vs. 0.68 for demographics alone.


## Using real data

Save your parcellated data as `.npz` with keys `X` (m × T × N float),
`y` (m, 0/1), `age`, `sex` (0/1), `scanner` (0/1), and optionally
`subject_id` and `region_networks`. Then use it as `data/ica25.npz` (or add a
new name to `configs/default.yaml` → `parcellations` and to `N_REGIONS` in
`scripts/run_experiments.py`).

## References

1. Y. Kim, C. Liu, J. Noh. *Classifying Adolescent Excessive Alcohol Drinkers from fMRI Data.* CS229 project, Stanford University.
2. NCANDA – National Consortium on Alcohol & Neurodevelopment in Adolescence. http://ncanda.org
3. R. C. Craddock et al. *A whole brain fMRI atlas generated via spatially constrained spectral clustering.* Human Brain Mapping 33(8), 2012.
4. L. M. Squeglia, J. Jacobus, S. F. Tapert. *The effect of alcohol use on human adolescent brain structures and systems.* Handb. Clin. Neurol. 125, 2014.
5. S. H. Park et al. *Alcohol use effects on adolescent brain development revealed by simultaneously removing confounding factors…* Scientific Reports 8, 2018.
