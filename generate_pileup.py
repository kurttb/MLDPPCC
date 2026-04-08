"""Generate synthetic pileup waveforms from preprocessed single-pulse data.

Usage:
    python generate_pileup.py [processed_waveforms.npz] [output.npz]

Defaults:
    input:  processed_waveforms.npz
    output: pileup_waveforms.npz

Partitions the singles into two disjoint pools:
  - Pool A (2/3): source material consumed by pileup generation
  - Pool B (1/3): clean singles never touched by pileup generation

Each pileup uses exactly 2 unique singles (no waveform reused anywhere).
Source waveforms are randomly paired, so the pair type (nn, ng, gn, gg)
follows the natural neutron/photon ratio in the data.
The number of pileups equals len(Pool B) for a 50/50 balance downstream.
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


def generate_pileup(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic pileup waveforms using a strict no-reuse partition.

    Parameters
    ----------
    X : (N, 104) voltage waveforms
    y : (N,) labels — 0=photon, 1=neutron
    seed : random seed

    Returns
    -------
    pileup_wf : (n_pileup, 104) pileup waveforms in volts
    primary_label : (n_pileup,) particle type of the first pulse (0 or 1)
    secondary_label : (n_pileup,) particle type of the second pulse (0 or 1)
    delays : (n_pileup,) delay in samples applied to the second pulse
    clean_indices : sorted array of indices into X that were NOT used (clean singles)
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]

    # ── Partition into clean (1/3) and pileup-source (2/3) ──────────────────
    all_idx = rng.permutation(N)
    n_clean = N // 3
    clean_indices = np.sort(all_idx[:n_clean])
    source_indices = all_idx[n_clean:]

    # ── Pair source waveforms ───────────────────────────────────────────────
    # Shuffle source and take consecutive pairs: (0,1), (2,3), ...
    rng.shuffle(source_indices)
    n_pileup = len(source_indices) // 2  # use all available pairs
    pri_indices = source_indices[0 : 2 * n_pileup : 2]   # even positions
    sec_indices = source_indices[1 : 2 * n_pileup : 2]   # odd positions

    # ── Build pileup waveforms ──────────────────────────────────────────────
    delays = rng.integers(
        DELAY_MIN_SAMPLES, DELAY_MAX_SAMPLES + 1, size=n_pileup
    ).astype(np.int16)

    pileup_wf = X[pri_indices].copy().astype(np.float32)
    for j in range(n_pileup):
        d = delays[j]
        pileup_wf[j, d:] += X[sec_indices[j], :SAMPLES_PER_PULSE - d]

    primary_label = y[pri_indices].astype(np.int8)
    secondary_label = y[sec_indices].astype(np.int8)

    # Shuffle the output order
    shuffle = rng.permutation(n_pileup)
    pileup_wf = pileup_wf[shuffle]
    primary_label = primary_label[shuffle]
    secondary_label = secondary_label[shuffle]
    delays = delays[shuffle]

    return pileup_wf, primary_label, secondary_label, delays, clean_indices


def main(input_path: Path, output_path: Path) -> None:
    print(f"Loading single-pulse data from {input_path} ...")
    data = np.load(input_path)
    X = data["X_voltage"]
    y = data["y"]
    N = X.shape[0]

    n_clean = N // 3
    n_source = N - n_clean
    n_pileup = n_source // 2

    print(f"Total singles: {N:,}")
    print(f"  Clean singles (1/3): {n_clean:,}")
    print(f"  Pileup source (2/3): {n_source:,}")
    print(f"  Pileups to generate: {n_pileup:,}")
    print(f"  Delay range: {DELAY_MIN_NS}–{DELAY_MAX_NS} ns "
          f"({DELAY_MIN_SAMPLES}–{DELAY_MAX_SAMPLES} samples)")

    pileup_wf, primary_label, secondary_label, delays, clean_indices = \
        generate_pileup(X, y)

    print(f"\nGenerated {pileup_wf.shape[0]:,} pileup events")
    print(f"Saving to {output_path} ...")
    np.savez_compressed(
        output_path,
        pileup_wf=pileup_wf,
        primary_label=primary_label,
        secondary_label=secondary_label,
        delays_samples=delays,
        clean_indices=clean_indices,
        time_ns=data["time_ns"],
    )

    print(f"  pileup_wf:       {pileup_wf.shape}")
    print(f"  primary_label:   {primary_label.shape}  (0=photon, 1=neutron)")
    print(f"  secondary_label: {secondary_label.shape}")
    print(f"  delays_samples:  {delays.shape}  (range {delays.min()}–{delays.max()} samples)")
    print(f"  clean_indices:   {clean_indices.shape}  (singles safe to use)")

    # Summary of pair types
    pair_counts = {}
    for pt in ["nn", "ng", "gn", "gg"]:
        p, s = (1 if pt[0] == "n" else 0), (1 if pt[1] == "n" else 0)
        count = ((primary_label == p) & (secondary_label == s)).sum()
        pair_counts[pt] = count
        print(f"  {pt}: {count:,}")


if __name__ == "__main__":
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("processed_waveforms.npz")
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("pileup_waveforms.npz")
    main(input_path, output_path)
