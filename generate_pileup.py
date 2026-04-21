"""Generate synthetic pileup waveforms from single-pulse data

Usage:
    python generate_pileup.py [processed_waveforms.npz] [output.npz]
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern"],
#     "font.size": 16,
#     "axes.labelsize": 12,
#     "axes.titlesize": 14,
#     "xtick.labelsize": 12,
#     "ytick.labelsize": 12,
#     "legend.fontsize": 14,
#     "figure.figsize": (8, 6),
#     "savefig.dpi": 400,
# })


# CAEN V1730 constants
NS_PER_SAMPLE = 2.0       # 500 MS/s
SAMPLES_PER_PULSE = 104

# Pileup generation parameters
DELAY_MIN_NS = 4.0
DELAY_MAX_NS = 50.0
DELAY_MIN_SAMPLES = int(np.ceil(DELAY_MIN_NS / NS_PER_SAMPLE))   # 2
DELAY_MAX_SAMPLES = int(np.floor(DELAY_MAX_NS / NS_PER_SAMPLE))  # 25


def generate_pileup(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate pileups with no waveform reuse"""
    rng = np.random.default_rng(seed)
    N = X.shape[0]

    # Split into clean singles and source singles
    all_idx = rng.permutation(N)
    n_clean = N // 3
    clean_indices = np.sort(all_idx[:n_clean])
    source_indices = all_idx[n_clean:]

    # Pair source waveforms
    rng.shuffle(source_indices)
    n_pileup = len(source_indices) // 2  # use all available pairs
    pri_indices = source_indices[0 : 2 * n_pileup : 2]   # even positions
    sec_indices = source_indices[1 : 2 * n_pileup : 2]   # odd positions

    # Build pileup waveforms
    delays = rng.integers(
        DELAY_MIN_SAMPLES, DELAY_MAX_SAMPLES + 1, size=n_pileup
    ).astype(np.int16)

    # Keep the two components as reconstruction targets
    pileup_wf = X[pri_indices].copy().astype(np.float32)
    primary_component = X[pri_indices].copy().astype(np.float32)
    secondary_component = np.zeros((n_pileup, SAMPLES_PER_PULSE), dtype=np.float32)
    for j in range(n_pileup):
        d = delays[j]
        secondary_component[j, d:] = X[sec_indices[j], :SAMPLES_PER_PULSE - d]
        pileup_wf[j, d:] += X[sec_indices[j], :SAMPLES_PER_PULSE - d]

    primary_label = y[pri_indices].astype(np.int8)
    secondary_label = y[sec_indices].astype(np.int8)

    # Shuffle the output order
    shuffle = rng.permutation(n_pileup)
    pileup_wf = pileup_wf[shuffle]
    primary_component = primary_component[shuffle]
    secondary_component = secondary_component[shuffle]
    primary_label = primary_label[shuffle]
    secondary_label = secondary_label[shuffle]
    delays = delays[shuffle]

    return (pileup_wf, primary_component, secondary_component,
            primary_label, secondary_label, delays, clean_indices)


def save_combined_example_plot(
    delay_ns_list: list,
    pileup_wf: np.ndarray,
    primary_component: np.ndarray,
    secondary_component: np.ndarray,
    primary_label: np.ndarray,
    secondary_label: np.ndarray,
    delays: np.ndarray,
    time_ns: np.ndarray,
    out_path: Path,
) -> None:
    """Plot example pileups for selected delays"""
    label_map = {0: "photon", 1: "neutron"}
    n = len(delay_ns_list)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, delay_ns in zip(axes, delay_ns_list):
        target_samples = int(round(delay_ns / NS_PER_SAMPLE))
        matches = np.where(delays == target_samples)[0]
        if len(matches) == 0:
            ax.set_title(f"{delay_ns:g} ns — no match")
            continue
        j = int(matches[0])
        ax.plot(time_ns, primary_component[j],
                label=f"primary ({label_map[int(primary_label[j])]})", alpha=0.7)
        ax.plot(time_ns, secondary_component[j],
                label=f"secondary ({label_map[int(secondary_label[j])]})", alpha=0.7)
        ax.plot(time_ns, pileup_wf[j], label="pileup (sum)", color="black", linewidth=1.5)
        ax.set_title(f"{delay_ns:g} ns delay ({target_samples} samples)")
        ax.set_xlabel("Time (ns)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Voltage")
    fig.suptitle("Example Pileup Waveforms")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def main(input_path: Path, output_path: Path) -> None:
    print(f"Loading single-pulse data from {input_path} ...")
    data = np.load(input_path)

    if output_path.exists():
        print(f"\n{output_path} already exists — skipping generation, loading existing file for plots.")
        existing = np.load(output_path)
        pileup_wf = existing["pileup_wf"]
        primary_component = existing["primary_component"]
        secondary_component = existing["secondary_component"]
        primary_label = existing["primary_label"]
        secondary_label = existing["secondary_label"]
        delays = existing["delays_samples"]
        clean_indices = existing["clean_indices"]
    else:
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

        (pileup_wf, primary_component, secondary_component,
         primary_label, secondary_label, delays, clean_indices) = generate_pileup(X, y)

        print(f"\nGenerated {pileup_wf.shape[0]:,} pileup events")
        print(f"Saving to {output_path} ...")
        np.savez_compressed(
            output_path,
            pileup_wf=pileup_wf,
            primary_component=primary_component,
            secondary_component=secondary_component,
            primary_label=primary_label,
            secondary_label=secondary_label,
            delays_samples=delays,
            clean_indices=clean_indices,
            time_ns=data["time_ns"],
        )

    print(f"  pileup_wf:            {pileup_wf.shape}")
    print(f"  primary_component:    {primary_component.shape}  (target for separator decoder 1)")
    print(f"  secondary_component:  {secondary_component.shape}  (target for separator decoder 2)")
    print(f"  primary_label:        {primary_label.shape}  (0=photon, 1=neutron)")
    print(f"  secondary_label:      {secondary_label.shape}")
    print(f"  delays_samples:       {delays.shape}  (range {delays.min()}–{delays.max()} samples)")
    print(f"  clean_indices:        {clean_indices.shape}  (singles safe to use)")

    # Save example pileups at three delays
    figures_dir = output_path.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    combined_path = figures_dir / "example_pileups.png"
    print(f"\nSaving combined example pileup plot to {combined_path} ...")
    save_combined_example_plot(
        [4, 16, 40], pileup_wf, primary_component, secondary_component,
        primary_label, secondary_label, delays, data["time_ns"], combined_path,
    )

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
