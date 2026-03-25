# MLDPPCC
Machine-Learning Dual-Particle Pulse Cleaner and Classifier

## Preprocessing

Place `photon.csv` and `neutron.csv` in a directory (e.g. `Raw_Labelled_Waveforms/`), then run:

```bash
python preprocess.py path/to/Raw_Labelled_Waveforms
```

This produces `processed_waveforms.npz` in the current directory containing baseline-subtracted voltage waveforms, Euclidean-normalized waveforms, and labels (0 = photon, 1 = neutron). The raw CSVs and `.npz` files are gitignored.
