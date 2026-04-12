# MLDPPCC
Machine-Learning Dual-Particle Pulse Cleaner and Classifier

A three-stage ML pipeline for neutron/photon pulse shape discrimination in organic scintillator data, including classification and separation of pile-up events.

## Pipeline Overview

```
Raw waveform
     │
     ▼
C1 (BiLSTM) ── single vs pile-up?
     │
  ┌──┴──┐
  │     │
single  pile-up
  │     │
  ▼     ▼
 A1    A2 Separator
 AE     splits into 2 components
  │     │
  ▼     ├─► component 1 ──► A1 ──► neutron/gamma
 C2     └─► component 2 ──► A1 ──► neutron/gamma
  │
  ▼
neutron/gamma
```

## Setup

### 1. Preprocessing

Place `photon.csv` and `neutron.csv` in a directory, then:

```bash
python preprocess.py path/to/Raw_Labelled_Waveforms
```

Produces `processed_waveforms.npz` containing baseline-subtracted voltage waveforms (`X_voltage`), L2-normalized waveforms (`X_euclidean`), per-waveform L2 norms (`X_norms`), and labels.

### 2. Synthetic pileup generation

```bash
python generate_pileup.py
```

Produces `pileup_waveforms.npz` from `processed_waveforms.npz`. Partitions singles into a source pool (2/3) and clean pool (1/3). Each pileup uses two unique source waveforms with a random delay (4-50 ns). Saves the combined pileup, individual component waveforms (for separation training), labels, and delays.

### 3. Training

Run the notebooks in order. All use the `tf-blackwell` conda environment.

## File Guide

### Preprocessing scripts
| File | Description |
|------|-------------|
| `preprocess.py` | Raw CSVs → `processed_waveforms.npz` (baseline subtract, L2 normalize) |
| `generate_pileup.py` | Synthetic pileup generation with ground-truth component waveforms |

### C1 — Single vs Pileup Classifier (BiLSTM)
| File | Description |
|------|-------------|
| `c1_classifier.ipynb` | Train and evaluate the C1 BiLSTM classifier |
| `c1_tune.ipynb` | Broad hyperparameter search (50 trials) with overfitting protections |
| `c1_tune_focused.ipynb` | Narrowed search based on broad results (25 trials) |

### A1 — Singles Autoencoder + Classifier
| File | Description |
|------|-------------|
| `a1_initial.ipynb` | Initial dense autoencoder exploration |
| `a1_multitask.ipynb` | Multi-task CNN autoencoder (reconstruction + n/g classification), old vs tuned comparison |
| `a1_tune.ipynb` | Hyperparameter search for A1 (100 trials, dense + CNN architectures) |

### A2 — Pileup Separator + Classifier
| File | Description |
|------|-------------|
| `a2_separator.ipynb` | Deltoro-style 1D-CAE that separates pileups into individual components, then classifies each with the trained A1 model |
| `a2_multitask.ipynb` | Multi-task pileup autoencoder (reconstruction + composition classification), comparison of architectures |
| `a2_tune.ipynb` | Hyperparameter search for A2 (100 trials, dense + CNN + U-Net) |

### Results CSVs
| File | Description |
|------|-------------|
| `c1_tune_results.csv` | C1 broad search results |
| `c1_tune_focused_results.csv` | C1 focused search results |
| `c1_tune_finalists.csv` | C1 top-5 honest test-set evaluation |
| `c1_tune_focused_finalists.csv` | C1 focused top-5 test evaluation |
| `a1_tune_results.csv` | A1 search results (dense + CNN, latent=8, pure L2) |
| `a2_tune_results.csv` | A2 search results (dense + CNN + U-Net) |

### Normalization

All classification models use **L2 (Euclidean) normalization** — each waveform is divided by its L2 norm, stripping amplitude and forcing the network to classify on shape only (following Jinia et al. Eq. 1). The separator (A2) uses **raw voltage** because separation requires amplitude information.

### Key references

- Jinia et al., "An Artificial Neural Network System for Photon-Based Active Interrogation Applications," IEEE Access, 2021 — dataset source, L2 normalization approach
- Deltoro et al., "Reconstruction of pile-up events using a one-dimensional convolutional autoencoder for the NEDA detector array," NUCL SCI TECH, 2025 — 1D-CAE separation architecture
