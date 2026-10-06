"""Dataset creation and loading.

The NCANDA resting-state fMRI data used in the reference study is only
available under a data-use agreement, so this module provides:

1. ``simulate_cohort`` -- a generator of NCANDA-like parcellated BOLD data
   (m subjects x T timesteps x N regions) with demographics, whose group
   differences are explicit, documented parameters (see ``SimulationConfig``).
2. ``load_cohort`` / ``save_cohort`` -- a simple ``.npz`` format so real
   parcellated data (e.g. NCANDA, once access is granted) can be dropped in
   without changing any other code.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

NETWORKS = ("DMN", "SAL", "ECN", "VIS", "SMN")


@dataclass
class SimulationConfig:
    """Parameters of the synthetic NCANDA-like cohort.

    Defaults mirror the cohort described in the reference study
    (715 subjects aged 16-19, 17% heavy drinkers, T = 269 volumes, TR = 2.2 s).
    Effect sizes are deliberately small so that the task stays hard.
    """

    n_subjects: int = 715
    n_timepoints: int = 269
    n_regions: int = 25
    heavy_drinker_rate: float = 0.17
    tr_seconds: float = 2.2
    seed: int = 229

    # --- demographics -> drinking status (log-odds per unit) ---
    age_logodds: float = 1.6          # per year above 17.5 (drinking rises with age)
    male_logodds: float = 0.35
    scanner_logodds: float = 0.15     # site / scanner confound

    # --- brain signal model ---
    network_ar: float = 0.85          # slow latent network fluctuations
    noise_ar: float = 0.5             # regional noise autocorrelation
    base_coupling: float = 1.0
    coupling_subject_sd: float = 0.18
    hd_coupling_shift: dict = field(default_factory=lambda: {"DMN": -0.15, "ECN": -0.12})
    hd_cross_coupling: float = 0.12   # extra SAL<->DMN coupling in heavy drinkers
    age_coupling_slope: float = 0.06  # developmental increase of DMN coupling / year
    hd_amplitude_boost: float = 0.05  # extra variance in SAL regions for heavy drinkers
    scanner_noise: tuple = (1.0, 1.25)
    global_signal_weight: float = 0.3
    motion_spike_rate: float = 0.01   # fraction of volumes with a motion spike

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Cohort:
    X: np.ndarray            # (m, T, N) float32 BOLD signals
    y: np.ndarray            # (m,) int, 1 = heavy drinker
    demographics: pd.DataFrame  # columns: subject_id, age, sex, scanner
    region_networks: np.ndarray  # (N,) network name per region
    name: str = "cohort"

    @property
    def shape(self):
        return self.X.shape

    def summary(self) -> str:
        m, T, N = self.X.shape
        n_hd = int(self.y.sum())
        return (f"{self.name}: m={m} subjects, T={T} timesteps, N={N} regions, "
                f"heavy drinkers={n_hd} ({100 * n_hd / m:.1f}%)")


def _ar1(rng: np.random.Generator, n_series: int, T: int, phi: float) -> np.ndarray:
    """Unit-variance AR(1) processes, shape (T, n_series)."""
    eps = rng.standard_normal((T, n_series)) * np.sqrt(1 - phi ** 2)
    out = np.empty((T, n_series))
    out[0] = rng.standard_normal(n_series)
    for t in range(1, T):
        out[t] = phi * out[t - 1] + eps[t]
    return out


def simulate_demographics(cfg: SimulationConfig) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    m = cfg.n_subjects
    age = rng.uniform(16.0, 20.0, m)
    sex = rng.integers(0, 2, m)            # 1 = male
    scanner = (rng.random(m) < 0.45).astype(int)   # 0 = GE, 1 = Siemens
    logit = (cfg.age_logodds * (age - 17.5) + cfg.male_logodds * sex
             + cfg.scanner_logodds * scanner)
    p = 1 / (1 + np.exp(-logit))
    n_hd = int(round(cfg.heavy_drinker_rate * m))
    hd_idx = rng.choice(m, size=n_hd, replace=False, p=p / p.sum())
    y = np.zeros(m, dtype=int)
    y[hd_idx] = 1
    demo = pd.DataFrame({
        "subject_id": [f"SIM-{i:04d}" for i in range(m)],
        "age": np.round(age, 2),
        "sex": sex,
        "scanner": scanner,
    })
    return demo, y


def simulate_cohort(cfg: SimulationConfig | None = None, name: str | None = None) -> Cohort:
    """Generate a synthetic resting-state fMRI cohort.

    Each region belongs to one of five canonical networks. Regional signals are
    a subject-specific coupling times a latent AR(1) network time course, plus a
    shared global signal, scanner-dependent AR(1) noise and sporadic motion
    spikes, on top of an arbitrary raw BOLD baseline (so z-scoring matters).
    Heavy drinkers get weaker DMN/ECN coupling, stronger SAL-DMN coupling and
    slightly larger SAL amplitude. Demographics and subject labels depend only
    on ``cfg.seed``, so different parcellations share the same subjects.
    """
    cfg = cfg or SimulationConfig()
    demo, y = simulate_demographics(cfg)
    m, T, N = cfg.n_subjects, cfg.n_timepoints, cfg.n_regions
    K = len(NETWORKS)

    # Fixed "atlas": region -> network assignment and loadings.
    atlas_rng = np.random.default_rng(1000 + N)
    region_net = np.arange(N) % K
    atlas_rng.shuffle(region_net)
    loadings = atlas_rng.uniform(0.6, 1.0, N)
    baseline = atlas_rng.uniform(400, 900, N)   # raw scanner units

    rng = np.random.default_rng(cfg.seed * 7919 + N)
    X = np.empty((m, T, N), dtype=np.float32)
    age_c = demo["age"].to_numpy() - 17.5
    scanner = demo["scanner"].to_numpy()
    dmn, sal = NETWORKS.index("DMN"), NETWORKS.index("SAL")

    for s in range(m):
        coupling = cfg.base_coupling + cfg.coupling_subject_sd * rng.standard_normal(K)
        coupling[dmn] += cfg.age_coupling_slope * age_c[s]
        if y[s]:
            for net, shift in cfg.hd_coupling_shift.items():
                coupling[NETWORKS.index(net)] += shift
        coupling = np.clip(coupling, 0.05, None)

        Z = _ar1(rng, K, T, cfg.network_ar)                     # (T, K)
        if y[s]:
            Z[:, sal] = ((1 - cfg.hd_cross_coupling) * Z[:, sal]
                         + cfg.hd_cross_coupling * Z[:, dmn])
        g = _ar1(rng, 1, T, cfg.network_ar)[:, 0]
        noise_sd = cfg.scanner_noise[scanner[s]] * rng.uniform(0.9, 1.1)
        E = _ar1(rng, N, T, cfg.noise_ar) * noise_sd

        signal = Z[:, region_net] * (coupling[region_net] * loadings)
        signal += cfg.global_signal_weight * g[:, None]
        if y[s]:
            signal[:, region_net == sal] *= 1 + cfg.hd_amplitude_boost
        bold = signal + E

        spikes = rng.random(T) < cfg.motion_spike_rate
        if spikes.any():
            bold[spikes] += rng.normal(0, 3.0, (spikes.sum(), 1))

        # Express as percent-signal fluctuations around a raw baseline.
        X[s] = (baseline * (1 + 0.01 * bold)).astype(np.float32)

    parcellation = {25: "ica25", 100: "craddock100"}.get(N, f"n{N}")
    return Cohort(X=X, y=y, demographics=demo,
                  region_networks=np.array([NETWORKS[k] for k in region_net]),
                  name=name or f"simulated_{parcellation}")


def save_cohort(cohort: Cohort, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = cohort.demographics
    np.savez_compressed(
        path, X=cohort.X, y=cohort.y,
        subject_id=d["subject_id"].to_numpy().astype(str), age=d["age"].to_numpy(),
        sex=d["sex"].to_numpy(), scanner=d["scanner"].to_numpy(),
        region_networks=np.asarray(cohort.region_networks).astype(str), name=cohort.name,
    )
    return path


def load_cohort(path: str | Path) -> Cohort:
    """Load a cohort saved in the project ``.npz`` format.

    Required keys: ``X`` (m, T, N), ``y`` (m,), ``age``, ``sex``, ``scanner``.
    Optional: ``subject_id``, ``region_networks``, ``name``.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as f:
        m = f["X"].shape[0]
        demo = pd.DataFrame({
            "subject_id": f["subject_id"] if "subject_id" in f else np.arange(m).astype(str),
            "age": f["age"], "sex": f["sex"], "scanner": f["scanner"],
        })
        nets = f["region_networks"] if "region_networks" in f else np.array(["NA"] * f["X"].shape[2])
        name = str(f["name"]) if "name" in f else path.stem
        return Cohort(X=f["X"].astype(np.float32), y=f["y"].astype(int),
                      demographics=demo, region_networks=nets, name=name)
