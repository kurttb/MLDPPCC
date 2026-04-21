from __future__ import annotations

import itertools
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers


SEARCH_SPACE = {
    "lstm_units": [32, 64],
    "n_lstm_layers": [2],
    "dense_units": [32, 64],
    "dropout": [0.1, 0.2],
    "learning_rate": [1e-3, 2e-3],
    "batch_size": [256, 512],
    "optimizer": ["adam", "adamw"],
}

N_TRIALS = 64
MAX_EPOCHS = 100
PATIENCE = 6
OVERFIT_GAP_THRESHOLD = 0.05

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = SCRIPT_DIR / "C1_Tune_Results_greatlakes_last32.csv"


for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    singles = np.load(SCRIPT_DIR / "processed_waveforms.npz")
    X_singles_all = singles["X_voltage"].astype(np.float32)

    pileups = np.load(SCRIPT_DIR / "pileup_waveforms.npz")
    X_pileup = pileups["pileup_wf"].astype(np.float32)
    clean_idx = pileups["clean_indices"]
    X_singles = X_singles_all[clean_idx]

    rng = np.random.default_rng(42)
    n_balanced = min(X_singles.shape[0], X_pileup.shape[0])
    single_idx = rng.choice(X_singles.shape[0], size=n_balanced, replace=False)
    pileup_idx = rng.choice(X_pileup.shape[0], size=n_balanced, replace=False)

    X_all = np.concatenate([X_singles[single_idx], X_pileup[pileup_idx]], axis=0)
    y_all = np.concatenate(
        [
            np.zeros(n_balanced, dtype=np.int64),
            np.ones(n_balanced, dtype=np.int64),
        ]
    )

    # Normalize each waveform independently so each row has min 0 and max 1.
    mins = X_all.min(axis=1, keepdims=True)
    maxs = X_all.max(axis=1, keepdims=True)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    X_all = (X_all - mins) / ranges

    shuffle = rng.permutation(len(y_all))
    X_all = X_all[shuffle]
    y_all = y_all[shuffle]

    X_tv, X_test, y_tv, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, random_state=42, stratify=y_tv
    )

    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_optimizer(name: str, learning_rate: float):
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=learning_rate)
    if name == "adamw":
        return keras.optimizers.AdamW(learning_rate=learning_rate)
    if name == "nadam":
        return keras.optimizers.Nadam(learning_rate=learning_rate)
    raise ValueError(f"Unknown optimizer: {name}")


def build_bilstm(hp: dict[str, object], n_samples: int = 104) -> keras.Model:
    model_layers = [keras.layers.Input(shape=(n_samples, 1))]
    for layer_index in range(int(hp["n_lstm_layers"])):
        return_seq = layer_index < int(hp["n_lstm_layers"]) - 1
        model_layers.append(
            layers.Bidirectional(
                layers.LSTM(
                    int(hp["lstm_units"]),
                    return_sequences=return_seq,
                    dropout=float(hp["dropout"]),
                )
            )
        )
    model_layers.append(layers.Dense(int(hp["dense_units"]), activation="relu"))
    model_layers.append(layers.Dropout(float(hp["dropout"])))
    model_layers.append(layers.Dense(1, activation="sigmoid"))

    model = keras.Sequential(model_layers)
    model.compile(
        optimizer=get_optimizer(str(hp["optimizer"]), float(hp["learning_rate"])),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def composite_score(val_loss: float, val_acc: float, alpha: float = 0.5) -> float:
    return alpha * val_loss + (1 - alpha) * (1 - val_acc)


def run_trial(
    hp: dict[str, object],
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
) -> dict[str, object]:
    keras.backend.clear_session()
    model = build_bilstm(hp, n_samples=X_train.shape[1])
    n_params = model.count_params()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True, verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=3, factor=0.5, verbose=0
        ),
    ]

    start_time = time.time()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=int(hp["batch_size"]),
        callbacks=callbacks,
        verbose=1,
    )
    train_time = time.time() - start_time

    train_loss, train_acc = model.evaluate(
        X_train, y_train, batch_size=int(hp["batch_size"]), verbose=0
    )
    val_loss, val_acc = model.evaluate(
        X_val, y_val, batch_size=int(hp["batch_size"]), verbose=0
    )

    overfit_gap = float(train_acc - val_acc)
    is_overfit = overfit_gap > OVERFIT_GAP_THRESHOLD
    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)

    return {
        **hp,
        "n_params": int(n_params),
        "best_epoch": best_epoch,
        "train_loss": float(train_loss),
        "train_acc": float(train_acc),
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "overfit_gap": overfit_gap,
        "is_overfit": bool(is_overfit),
        "composite_score": float(composite_score(float(val_loss), float(val_acc))),
        "train_time_s": round(train_time, 1),
    }


def all_configs() -> list[tuple[object, ...]]:
    configs = list(itertools.product(*SEARCH_SPACE.values()))
    if len(configs) > N_TRIALS:
        raise ValueError(
            "This script expects the focused grid to be fully enumerated; found more than N_TRIALS configs."
        )
    return configs


def key_from_hp(hp: dict[str, object]) -> tuple[object, ...]:
    return (
        int(hp["lstm_units"]),
        int(hp["n_lstm_layers"]),
        int(hp["dense_units"]),
        float(hp["dropout"]),
        float(hp["learning_rate"]),
        int(hp["batch_size"]),
        str(hp["optimizer"]),
    )


def main() -> None:
    print(f"TensorFlow {tf.__version__}")
    print(f"GPUs: {tf.config.list_physical_devices('GPU')}")

    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    print(
        f"Class balance (train): single={(y_train == 0).sum():,}  pileup={(y_train == 1).sum():,}"
    )

    configs = all_configs()
    split_index = len(configs) // 2
    target_configs = configs[split_index:]
    keys = list(SEARCH_SPACE.keys())

    print(f"Total possible combos: {len(configs):,}")
    print(f"Running last {len(target_configs)} configurations: trials {split_index + 1} to {len(configs)}")
    print(f"Results file: {RESULTS_PATH.name}")

    if RESULTS_PATH.exists():
        existing_df = pd.read_csv(RESULTS_PATH)
        results = existing_df.to_dict("records")
        completed_keys = {key_from_hp(row) for row in results}
        print(f"Resuming from {RESULTS_PATH.name}: {len(results)} completed trials")
    else:
        results = []
        completed_keys = set()
        print(f"Starting fresh: {RESULTS_PATH.name}")

    for global_trial_num, combo in enumerate(target_configs, start=split_index + 1):
        hp = dict(zip(keys, combo))
        key = key_from_hp(hp)

        if key in completed_keys:
            print(f"\nTrial {global_trial_num}/{len(configs)}: SKIP (already done) {hp}")
            continue

        print(f"\n{'=' * 78}")
        print(f"Trial {global_trial_num}/{len(configs)}")
        print(
            f"  arch: lstm={hp['lstm_units']} layers={hp['n_lstm_layers']} "
            f"dense={hp['dense_units']} drop={hp['dropout']}"
        )
        print(
            f"  opt:  {hp['optimizer']} lr={hp['learning_rate']} bs={hp['batch_size']}"
        )

        try:
            result = run_trial(hp, X_train, X_val, y_train, y_val)
            results.append(result)
            completed_keys.add(key)
            flag = " [OVERFIT]" if result["is_overfit"] else ""
            print(
                f"  -> train_acc={result['train_acc']:.4f}  val_acc={result['val_acc']:.4f}  "
                f"gap={result['overfit_gap']:+.4f}{flag}  "
                f"score={result['composite_score']:.5f}  "
                f"params={result['n_params']:,}  time={result['train_time_s']}s"
            )
        except Exception as exc:
            print(f"  -> FAILED: {exc}")
            continue

        pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)

    n_overfit = sum(1 for row in results if row["is_overfit"])
    print(f"\n{'=' * 78}")
    print(
        f"Complete: {len(results)}/{len(target_configs)} successful in the Great Lakes slice, "
        f"{n_overfit} flagged as overfit"
    )


if __name__ == "__main__":
    main()