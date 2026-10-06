import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.linear_model import LogisticRegression  # noqa: E402

from fmri_alcohol.data import SimulationConfig, load_cohort, save_cohort, simulate_cohort  # noqa: E402
from fmri_alcohol.evaluate import ProtocolA, classification_metrics  # noqa: E402
from fmri_alcohol.features import build_features, connectivity, dynamic_range  # noqa: E402
from fmri_alcohol.models.deep import DeepClassifier, should_stop  # noqa: E402
from fmri_alcohol.models.logistic_newton import NewtonLogisticRegression  # noqa: E402
from fmri_alcohol.preprocess import balance_undersample, dev_split, zscore_timeseries  # noqa: E402


@pytest.fixture(scope="module")
def cohort():
    return simulate_cohort(SimulationConfig(n_subjects=120, n_timepoints=60, n_regions=10, seed=1))


def test_cohort_shape_and_prevalence(cohort):
    assert cohort.X.shape == (120, 60, 10)
    assert cohort.y.sum() == round(0.17 * 120)
    assert set(cohort.demographics.columns) >= {"age", "sex", "scanner"}


def test_same_subjects_across_parcellations():
    a = simulate_cohort(SimulationConfig(n_subjects=50, n_timepoints=20, n_regions=5, seed=3))
    b = simulate_cohort(SimulationConfig(n_subjects=50, n_timepoints=20, n_regions=12, seed=3))
    assert (a.y == b.y).all() and a.demographics.equals(b.demographics)


def test_save_load_roundtrip(cohort, tmp_path):
    path = save_cohort(cohort, tmp_path / "c.npz")
    c2 = load_cohort(path)
    np.testing.assert_allclose(c2.X, cohort.X)
    assert (c2.y == cohort.y).all()


def test_zscore(cohort):
    Z = zscore_timeseries(cohort.X)
    np.testing.assert_allclose(Z.mean(1), 0, atol=1e-4)
    np.testing.assert_allclose(Z.std(1), 1, atol=1e-3)


def test_features(cohort):
    Z = zscore_timeseries(cohort.X)
    np.testing.assert_allclose(dynamic_range(Z), Z.max(1) - Z.min(1))
    assert connectivity(Z).shape == (120, 45)
    F = build_features(Z, cohort.demographics, "range+conn", ("age",))
    assert F.shape == (120, 10 + 45 + 1)


def test_balancing_and_dev_split(cohort):
    idx = balance_undersample(cohort.y, seed=0)
    assert cohort.y[idx].mean() == 0.5
    cv, dev = dev_split(idx, cohort.y, 0.1, seed=0)
    assert not set(cv) & set(dev) and len(cv) + len(dev) == len(idx)


def test_protocol_no_leakage(cohort):
    for _, _, tr, te, dev in ProtocolA(k=5, repeats=2).splits(cohort.y):
        assert not (set(tr) & set(te)) and not (set(dev) & (set(tr) | set(te)))


def test_newton_matches_sklearn():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = (X @ [1.0, -2.0, 0.5, 0.0] + rng.normal(size=300) > 0).astype(int)
    ours = NewtonLogisticRegression(l2=1.0).fit(X, y)
    ref = LogisticRegression(C=1.0, tol=1e-10, max_iter=1000).fit(X, y)
    np.testing.assert_allclose(ours.coef_, ref.coef_[0], atol=1e-4)
    assert ours.n_iter_ < 15


def test_metrics():
    m = classification_metrics([0, 0, 1, 1], [0.1, 0.6, 0.8, 0.9])
    assert m["accuracy"] == 0.75 and m["recall"] == 1.0 and m["specificity"] == 0.5


def test_stopping_rules():
    assert should_stop(0.62, 0.61)[1] == "converged"
    assert should_stop(0.90, 0.60)[1] == "diverging"
    assert should_stop(0.50, 0.50)[0] is False


@pytest.mark.parametrize("arch", ["rnn", "rnn_nn", "cnn_nn", "cnn_rnn", "cnn_rnn_demo"])
def test_deep_models_run(cohort, arch):
    Z = zscore_timeseries(cohort.X)
    D = cohort.demographics[["age", "sex", "scanner"]].to_numpy(float)
    use_d = arch.endswith("demo")
    clf = DeepClassifier(arch, epochs=2).fit(Z[:80], cohort.y[:80], Z[80:], cohort.y[80:],
                                             D[:80] if use_d else None, D[80:] if use_d else None)
    p = clf.predict_proba(Z[80:], D[80:] if use_d else None)
    assert p.shape == (40, 2) and np.all((p >= 0) & (p <= 1))
