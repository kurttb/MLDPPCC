"""C1 BiLSTM Hyperparameter Tuning — random search over key hyperparameters.

Usage:
    python hyperparam_tuning.py

Results are saved to hyperparam_results.csv.
"""

import itertools
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

print(f"TensorFlow {tf.__version__}")
print(f"GPUs: {tf.config.list_physical_devices('GPU')}")

# ── Load data ────────────────────────────────────────────────────────────────
print("\nLoading data...")
singles = np.load("processed_waveforms.npz")
X_singles_all = singles["X_voltage"].astype(np.float32)

pileups = np.load("pileup_waveforms.npz")
X_pileup = pileups["pileup_wf"].astype(np.float32)
n_pileup = X_pileup.shape[0]

clean_idx = pileups["clean_indices"]
X_singles = X_singles_all[clean_idx]
n_singles = X_singles.shape[0]

# Balance
rng = np.random.default_rng(42)
n_balanced = min(n_singles, n_pileup)
single_idx = rng.choice(n_singles, size=n_balanced, replace=False)
pileup_idx = rng.choice(n_pileup, size=n_balanced, replace=False)

X_all = np.concatenate([X_singles[single_idx], X_pileup[pileup_idx]], axis=0)
y_all = np.concatenate([
    np.zeros(n_balanced, dtype=np.int64),
    np.ones(n_balanced, dtype=np.int64),
])

# Euclidean normalize
norms = np.linalg.norm(X_all, axis=1, keepdims=True)
norms[norms == 0] = 1.0
X_all = X_all / norms

# Shuffle
shuffle = rng.permutation(len(y_all))
X_all = X_all[shuffle]
y_all = y_all[shuffle]

# Train / val / test split (60/20/20)
X_tv, X_test, y_tv, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.25, random_state=42, stratify=y_tv
)

# Add feature dim
X_train = X_train[..., np.newaxis]
X_val   = X_val[..., np.newaxis]
X_test  = X_test[..., np.newaxis]

print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

# ── Search space ─────────────────────────────────────────────────────────────
SEARCH_SPACE = {
    "lstm_units":    [32, 64, 128],
    "n_lstm_layers": [1, 2],
    "dense_units":   [16, 32, 64],
    "dropout":       [0.1, 0.2, 0.3, 0.4, 0.5],
    "learning_rate": [1e-4, 5e-4, 1e-3, 2e-3],
    "batch_size":    [256, 512, 1024],
}

N_TRIALS = 30
MAX_EPOCHS = 25

total_combos = 1
for v in SEARCH_SPACE.values():
    total_combos *= len(v)
print(f"\nTotal possible combos: {total_combos}")
print(f"Sampling {N_TRIALS} random configurations\n")


# ── Model builder ────────────────────────────────────────────────────────────
def build_model(lstm_units, n_lstm_layers, dense_units, dropout, learning_rate):
    layers = [keras.layers.Input(shape=(104, 1))]
    for i in range(n_lstm_layers):
        return_seq = (i < n_lstm_layers - 1)
        layers.append(
            keras.layers.Bidirectional(
                keras.layers.LSTM(lstm_units, return_sequences=return_seq, dropout=dropout)
            )
        )
    layers.append(keras.layers.Dense(dense_units, activation="relu"))
    layers.append(keras.layers.Dropout(dropout))
    layers.append(keras.layers.Dense(1, activation="sigmoid"))

    model = keras.Sequential(layers)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Random search ────────────────────────────────────────────────────────────
random.seed(42)
all_combos = list(itertools.product(*SEARCH_SPACE.values()))
keys = list(SEARCH_SPACE.keys())
sampled = random.sample(all_combos, min(N_TRIALS, len(all_combos)))

results = []

for trial_num, combo in enumerate(sampled):
    hp = dict(zip(keys, combo))
    print(f"{'='*70}")
    print(f"Trial {trial_num+1}/{len(sampled)}: {hp}")
    print(f"{'='*70}")

    keras.backend.clear_session()

    model = build_model(
        lstm_units=hp["lstm_units"],
        n_lstm_layers=hp["n_lstm_layers"],
        dense_units=hp["dense_units"],
        dropout=hp["dropout"],
        learning_rate=hp["learning_rate"],
    )

    n_params = model.count_params()

    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True, verbose=1,
            monitor="val_accuracy",
        ),
        keras.callbacks.ReduceLROnPlateau(
            patience=2, factor=0.5, verbose=1,
        ),
    ]

    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=hp["batch_size"],
        callbacks=callbacks,
        verbose=2,  # one line per epoch
    )
    train_time = time.time() - t0

    test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=1024, verbose=0)

    best_epoch = np.argmax(history.history["val_accuracy"]) + 1
    best_val_acc = max(history.history["val_accuracy"])
    best_val_loss = history.history["val_loss"][best_epoch - 1]

    result = {
        **hp,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "train_time_s": round(train_time, 1),
    }
    results.append(result)

    print(f"  Params: {n_params:,}  |  Best epoch: {best_epoch}")
    print(f"  Val acc: {best_val_acc:.4f}  |  Test acc: {test_acc:.4f}  |  Time: {train_time:.0f}s")
    print()

    # Save incrementally so we don't lose progress if interrupted
    df = pd.DataFrame(results)
    df.to_csv("hyperparam_results.csv", index=False)

# ── Final summary ────────────────────────────────────────────────────────────
df = pd.DataFrame(results).sort_values("test_acc", ascending=False).reset_index(drop=True)
df.index += 1
df.index.name = "Rank"

print("\n" + "="*70)
print("TOP 10 CONFIGURATIONS BY TEST ACCURACY")
print("="*70)
display_cols = ["lstm_units", "n_lstm_layers", "dense_units", "dropout",
                "learning_rate", "batch_size", "n_params",
                "best_val_acc", "test_acc", "best_epoch", "train_time_s"]
print(df[display_cols].head(10).to_string())

best = df.iloc[0]
print(f"\nBest configuration:")
print(f"  LSTM units:    {int(best['lstm_units'])}")
print(f"  BiLSTM layers: {int(best['n_lstm_layers'])}")
print(f"  Dense units:   {int(best['dense_units'])}")
print(f"  Dropout:       {best['dropout']}")
print(f"  Learning rate: {best['learning_rate']}")
print(f"  Batch size:    {int(best['batch_size'])}")
print(f"  Parameters:    {int(best['n_params']):,}")
print(f"  Val accuracy:  {best['best_val_acc']:.4f}")
print(f"  Test accuracy: {best['test_acc']:.4f}")

df.to_csv("hyperparam_results.csv")
print("\nResults saved to hyperparam_results.csv")
