"""Generate the simulated NCANDA-like cohorts (ICA-25 and Craddock-100).

    python scripts/generate_data.py            # writes data/ica25.npz, data/craddock100.npz
    python scripts/generate_data.py --subjects 200 --out data/small
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fmri_alcohol.data import SimulationConfig, save_cohort, simulate_cohort  # noqa: E402

PARCELLATIONS = {"ica25": 25, "craddock100": 100}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data")
    ap.add_argument("--subjects", type=int, default=715)
    ap.add_argument("--seed", type=int, default=229)
    ap.add_argument("--parcellations", nargs="+", default=list(PARCELLATIONS))
    args = ap.parse_args()

    out = Path(args.out)
    for name in args.parcellations:
        cfg = SimulationConfig(n_subjects=args.subjects, n_regions=PARCELLATIONS[name], seed=args.seed)
        cohort = simulate_cohort(cfg, name=name)
        path = save_cohort(cohort, out / f"{name}.npz")
        cohort.demographics.assign(heavy_drinker=cohort.y).to_csv(out / f"{name}_demographics.csv", index=False)
        (out / f"{name}_config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
        print(f"[saved] {path}  ->  {cohort.summary()}")


if __name__ == "__main__":
    main()
