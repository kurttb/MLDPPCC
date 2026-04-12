"""Preprocess raw CAEN V1730 waveform CSVs into a labeled .npz dataset.

Usage:
    python preprocess.py path/to/Raw_Labelled_Waveforms

Expects the directory to contain ``photon.csv`` and ``neutron.csv``.
Outputs ``processed_waveforms.npz`` in the current directory.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── CAEN V1730 digitizer constants (Jinia et al., IEEE Access 2021) ──────────

SAMPLE_RATE_HZ = 500e6          # 500 MS/s
ADC_BITS = 14
ADC_MAX = 2**ADC_BITS           # 16384
NS_PER_SAMPLE = 1e9 / SAMPLE_RATE_HZ  # 2 ns
SAMPLES_PER_PULSE = 104
V_PER_ADC = 2.0 / ADC_MAX      # V1730: 2 Vpp full-scale
BASELINE_SAMPLES = 8            # first 16 ns used for baseline
TAIL_START_SAMPLE = 25          # ~50 ns — start of tail integration window


# ── I/O ──────────────────────────────────────────────────────────────────────

def load_waveforms(path: Path) -> np.ndarray:
    """Load a raw CSV and return array of shape (n_waveforms, 104).

    Rows in the CSV are time samples; columns are waveforms — so we transpose.
    """
    raw = pd.read_csv(path, header=None).to_numpy(dtype=np.int16)
    return raw.T


def time_axis() -> np.ndarray:
    """Return the time axis in nanoseconds for one pulse window."""
    return np.arange(SAMPLES_PER_PULSE) * NS_PER_SAMPLE


# ── Preprocessing ────────────────────────────────────────────────────────────

def baseline_subtract(waveforms: np.ndarray) -> np.ndarray:
    """Baseline-subtract, flip polarity, and convert to volts.

    1. Per-waveform baseline = mean of first BASELINE_SAMPLES samples.
    2. Flip polarity (negative-going → positive-going).
    3. Scale ADC counts to volts.
    """
    wf = waveforms.astype(np.float32)
    baselines = wf[:, :BASELINE_SAMPLES].mean(axis=1, keepdims=True)
    return (baselines - wf) * V_PER_ADC


def euclidean_normalize(waveforms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Euclidean-normalize each waveform and return (normalized, norms).

    The returned norms array lets you invert the normalization later:
        waveforms_raw = waveforms_normalized * norms[:, None]
    This is useful for converting reconstructions back to voltage scale.
    """
    norms = np.linalg.norm(waveforms, axis=1)
    safe = np.where(norms == 0, 1.0, norms)
    return waveforms / safe[:, None], norms


def extract_features(wf_v: np.ndarray) -> pd.DataFrame:
    """Extract PSD features: pulse height, total/tail integral, PSD ratio."""
    pulse_height = wf_v.max(axis=1)
    total_integral = wf_v.sum(axis=1)
    tail_integral = wf_v[:, TAIL_START_SAMPLE:].sum(axis=1)
    return pd.DataFrame({
        "pulse_height_V": pulse_height,
        "total_integral": total_integral,
        "tail_integral": tail_integral,
        "psd_ratio": tail_integral / np.where(total_integral != 0, total_integral, 1.0),
    })


# ── Main ─────────────────────────────────────────────────────────────────────

def main(data_dir: Path, out_path: Path) -> None:
    photon_path = data_dir / "photon.csv"
    neutron_path = data_dir / "neutron.csv"

    for p in (photon_path, neutron_path):
        if not p.exists():
            print(f"Error: {p} not found", file=sys.stderr)
            sys.exit(1)

    print(f"Loading photon waveforms from {photon_path} ...")
    photon_wf = load_waveforms(photon_path)

    print(f"Loading neutron waveforms from {neutron_path} ...")
    neutron_wf = load_waveforms(neutron_path)

    print("Preprocessing ...")
    photon_v = baseline_subtract(photon_wf)
    neutron_v = baseline_subtract(neutron_wf)

    X = np.concatenate([photon_v, neutron_v], axis=0)
    y = np.concatenate([
        np.zeros(photon_v.shape[0], dtype=np.int8),   # 0 = photon
        np.ones(neutron_v.shape[0], dtype=np.int8),    # 1 = neutron
    ])

    rng = np.random.default_rng(42)
    shuffle_idx = rng.permutation(len(y))
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    X_euclidean, X_norms = euclidean_normalize(X)

    print(f"Saving to {out_path} ...")
    np.savez_compressed(
        out_path,
        X_voltage=X,
        X_euclidean=X_euclidean,
        X_norms=X_norms,
        y=y,
        time_ns=time_axis(),
    )

    print(f"  X_voltage:   {X.shape}  (baseline-subtracted, positive-going, volts)")
    print(f"  X_euclidean: {X_euclidean.shape}  (Euclidean-normalized)")
    print(f"  X_norms:     {X_norms.shape}  (original L2 norms, for recovery to volts)")
    print(f"  y:           {y.shape}  (0=photon, 1=neutron)")
    print(f"  Photon: {(y == 0).sum():,}  |  Neutron: {(y == 1).sum():,}  |  Total: {len(y):,}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <data_dir> [output.npz]", file=sys.stderr)
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("processed_waveforms.npz")
    main(data_dir, out_path)
