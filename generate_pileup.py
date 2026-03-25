"""Generate synthetic pileup waveforms from preprocessed single-pulse data.

Usage:
    python generate_pileup.py [processed_waveforms.npz] [output.npz]

Defaults:
    input:  processed_waveforms.npz
    output: pileup_waveforms.npz

Creates pileup events by superimposing two waveforms with a random time delay
(between 4 ns and 50 ns, i.e. 2–25 samples at 500 MS/s). All four pair types
(n+n, n+g, g+n, g+g) are generated in equal proportions.
"""

import sys
from pathlib import Path

import numpy as np

# ── CAEN V1730 constants ─────────────────────────────────────────────────────
NS_PER_SAMPLE = 2.0       # 500 MS/s
SAMPLES_PER_PULSE = 104

# ── Pileup generation parameters ─────────────────────────────────────────────
DELAY_MIN_NS = 4.0
DELAY_MAX_NS = 50.0
DELAY_MIN_SAMPLES = int(np.ceil(DELAY_MIN_NS / NS_PER_SAMPLE))   # 2
DELAY_MAX_SAMPLES = int(np.floor(DELAY_MAX_NS / NS_PER_SAMPLE))  # 25
N_PILEUP_TOTAL = 400000  # 100k per pair type × 4 types
PAIR_TYPES = ["nn", "ng", "gn", "gg"]  # (primary, secondary)


def generate_pileup(
    X: np.ndarray,
    y: np.ndarray,
    n_total: int = N_PILEUP_TOTAL,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic pileup waveforms.

    Parameters
    ----------
    X : (N, 104) voltage waveforms
    y : (N,) labels — 0=photon, 1=neutron
    n_total : total number of pileup events to generate
    seed : random seed

    Returns
    -------
    pileup_wf : (n_total, 104) pileup waveforms in volts
    primary_label : (n_total,) particle type of the first pulse (0 or 1)
    secondary_label : (n_total,) particle type of the second pulse (0 or 1)
    delays : (n_total,) delay in samples applied to the second pulse
    """
    rng = np.random.default_rng(seed)

    neutron_idx = np.where(y == 1)[0]
    photon_idx = np.where(y == 0)[0]

    # Map pair type to (primary_pool, secondary_pool)
    pool_map = {
        "nn": (neutron_idx, neutron_idx),
        "ng": (neutron_idx, photon_idx),
        "gn": (photon_idx, neutron_idx),
        "gg": (photon_idx, photon_idx),
    }
    label_map = {
        "nn": (1, 1),
        "ng": (1, 0),
        "gn": (0, 1),
        "gg": (0, 0),
    }

    n_per_type = n_total // len(PAIR_TYPES)
    remainder = n_total - n_per_type * len(PAIR_TYPES)

    pileup_wf = np.zeros((n_total, SAMPLES_PER_PULSE), dtype=np.float32)
    primary_label = np.zeros(n_total, dtype=np.int8)
    secondary_label = np.zeros(n_total, dtype=np.int8)
    delays = np.zeros(n_total, dtype=np.int16)

    offset = 0
    for i, pair_type in enumerate(PAIR_TYPES):
        # Give any remainder to the last type
        count = n_per_type + (remainder if i == len(PAIR_TYPES) - 1 else 0)
        pool_pri, pool_sec = pool_map[pair_type]
        lab_pri, lab_sec = label_map[pair_type]

        # Pick random waveform indices
        idx_pri = rng.choice(pool_pri, size=count, replace=True)
        idx_sec = rng.choice(pool_sec, size=count, replace=True)

        # Random delays in samples
        d = rng.integers(DELAY_MIN_SAMPLES, DELAY_MAX_SAMPLES + 1, size=count)

        for j in range(count):
            wf1 = X[idx_pri[j]]
            wf2 = X[idx_sec[j]]
            delay = d[j]

            # Shift wf2 right by `delay` samples and add to wf1
            combined = wf1.copy()
            combined[delay:] += wf2[:SAMPLES_PER_PULSE - delay]

            pileup_wf[offset] = combined
            primary_label[offset] = lab_pri
            secondary_label[offset] = lab_sec
            delays[offset] = delay
            offset += 1

    # Shuffle everything together
    shuffle = rng.permutation(n_total)
    pileup_wf = pileup_wf[shuffle]
    primary_label = primary_label[shuffle]
    secondary_label = secondary_label[shuffle]
    delays = delays[shuffle]

    return pileup_wf, primary_label, secondary_label, delays


def main(input_path: Path, output_path: Path) -> None:
    print(f"Loading single-pulse data from {input_path} ...")
    data = np.load(input_path)
    X = data["X_voltage"]
    y = data["y"]

    print(f"Generating {N_PILEUP_TOTAL:,} synthetic pileup events ...")
    print(f"  Delay range: {DELAY_MIN_NS}–{DELAY_MAX_NS} ns "
          f"({DELAY_MIN_SAMPLES}–{DELAY_MAX_SAMPLES} samples)")
    print(f"  Pair types: {', '.join(PAIR_TYPES)} "
          f"(~{N_PILEUP_TOTAL // len(PAIR_TYPES):,} each)")

    pileup_wf, primary_label, secondary_label, delays = generate_pileup(X, y)

    print(f"Saving to {output_path} ...")
    np.savez_compressed(
        output_path,
        pileup_wf=pileup_wf,
        primary_label=primary_label,
        secondary_label=secondary_label,
        delays_samples=delays,
        time_ns=data["time_ns"],
    )

    print(f"  pileup_wf:       {pileup_wf.shape}")
    print(f"  primary_label:   {primary_label.shape}  (0=photon, 1=neutron)")
    print(f"  secondary_label: {secondary_label.shape}")
    print(f"  delays_samples:  {delays.shape}  (range {delays.min()}–{delays.max()} samples)")

    # Summary of pair types
    for pt in PAIR_TYPES:
        p, s = (1 if pt[0] == "n" else 0), (1 if pt[1] == "n" else 0)
        count = ((primary_label == p) & (secondary_label == s)).sum()
        print(f"  {pt}: {count:,}")


if __name__ == "__main__":
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("processed_waveforms.npz")
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("pileup_waveforms.npz")
    main(input_path, output_path)
