#!/usr/bin/env python3
"""Step 9: Train hidden-to-output SNN weights with a simple supervised rule."""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

try:
    from brian2 import (
        Hz,
        Network,
        NeuronGroup,
        PoissonGroup,
        SpikeMonitor,
        Synapses,
        ms,
        prefs,
        start_scope,
    )
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Brian2 is required for this script. Install it with: pip install brian2"
    ) from exc

INPUT_PATH = "../data/processed/spike_rates.npz"
OUTPUT_CM_PATH = "../plots/snn_trained_confusion_matrix_v2.png"
OUTPUT_WEIGHTS_PATH = "../data/processed/snn_trained_weights.npz"
SIM_DURATION_MS = 200
RNG_SEED = 42
TEST_SIZE = 0.2
N_EPOCHS = 15
ETA = 0.001
N_HIDDEN = 50
R_MAX_HZ = 200.0


def _first_available(data: np.lib.npyio.NpzFile, keys: tuple[str, ...]) -> np.ndarray:
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"None of these keys were found: {keys}. Available keys: {list(data.files)}")


def build_snn(w_in_hidden: np.ndarray, w_hidden_out: np.ndarray):
    """Build the SNN once and return reusable objects."""
    start_scope()
    prefs.codegen.target = "numpy"

    input_group = PoissonGroup(8, rates=np.zeros(8) * Hz)

    lif_model = """
    dv/dt = (v_rest - v)/tau : 1
    v_rest : 1 (constant)
    tau : second (constant)
    """

    hidden_group = NeuronGroup(
        N_HIDDEN,
        model=lif_model,
        threshold="v > 1",
        reset="v = 0",
        method="exact",
    )
    hidden_group.v_rest = 0
    hidden_group.tau = 10 * ms
    hidden_group.v = 0

    output_group = NeuronGroup(
        2,
        model=lif_model,
        threshold="v > 1",
        reset="v = 0",
        method="exact",
    )
    output_group.v_rest = 0
    output_group.tau = 10 * ms
    output_group.v = 0

    syn_in_hidden = Synapses(input_group, hidden_group, model="w : 1", on_pre="v_post += w")
    syn_in_hidden.connect()
    syn_in_hidden.w = w_in_hidden.flatten()

    syn_hidden_out = Synapses(hidden_group, output_group, model="w : 1", on_pre="v_post += w")
    syn_hidden_out.connect()
    syn_hidden_out.w = w_hidden_out.flatten()

    hidden_spike_monitor = SpikeMonitor(hidden_group)
    output_spike_monitor = SpikeMonitor(output_group)

    net = Network(
        input_group,
        hidden_group,
        output_group,
        syn_in_hidden,
        syn_hidden_out,
        hidden_spike_monitor,
        output_spike_monitor,
    )
    return net, input_group, syn_hidden_out, hidden_spike_monitor, output_spike_monitor


def run_window(
    net: Network,
    input_group: PoissonGroup,
    syn_hidden_out: Synapses,
    hidden_spike_monitor: SpikeMonitor,
    output_spike_monitor: SpikeMonitor,
    rates_window: np.ndarray,
    w_hidden_out: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run one window and return hidden counts, output counts, and predicted label."""
    net.restore("blank")
    rate_values = np.clip(rates_window.astype(np.float64), 0.0, None)
    max_rate = np.max(rate_values)
    if max_rate > 0:
        normalized_rates = (rate_values / max_rate) * R_MAX_HZ
    else:
        normalized_rates = rate_values
    input_group.rates = normalized_rates * Hz
    syn_hidden_out.w = w_hidden_out.flatten()
    net.run(SIM_DURATION_MS * ms)

    hidden_counts = np.asarray(hidden_spike_monitor.count[:], dtype=np.float64)
    output_counts = np.asarray(np.bincount(output_spike_monitor.i, minlength=2), dtype=np.int32)
    winner = int(np.argmax(output_counts))
    predicted_label = 1 if winner == 0 else 2
    return hidden_counts, output_counts, predicted_label


def plot_confusion_matrix(cm: np.ndarray, out_path: str) -> None:
    """Plot and save confusion matrix image."""
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Trained SNN Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["Rest (1)", "Fist (2)"])
    plt.yticks([0, 1], ["Rest (1)", "Fist (2)"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2.0 else "black"
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.normpath(os.path.join(script_dir, INPUT_PATH))
    output_cm_path = os.path.normpath(os.path.join(script_dir, OUTPUT_CM_PATH))
    output_weights_path = os.path.normpath(os.path.join(script_dir, OUTPUT_WEIGHTS_PATH))

    data = np.load(input_path)
    spike_rates = _first_available(data, ("spike_rates",))
    labels = _first_available(data, ("y_labels", "y"))

    if spike_rates.ndim != 2 or spike_rates.shape[1] != 8:
        raise ValueError(f"Expected spike_rates shape (N, 8), got {spike_rates.shape}")
    if labels.shape[0] != spike_rates.shape[0]:
        raise ValueError(
            f"Mismatched sample counts: spike_rates N={spike_rates.shape[0]}, labels N={labels.shape[0]}"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        spike_rates,
        labels,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=RNG_SEED,
    )

    rng = np.random.default_rng(RNG_SEED)
    w_in_hidden = rng.uniform(0.0, 0.30, size=(8, N_HIDDEN))
    w_hidden_out = rng.uniform(0.0, 0.60, size=(N_HIDDEN, 2))

    net, input_group, syn_hidden_out, hidden_spike_monitor, output_spike_monitor = build_snn(
        w_in_hidden, w_hidden_out
    )
    net.store("blank")

    for epoch in range(N_EPOCHS):
        correct = 0
        total_output_spikes = 0.0
        permutation = rng.permutation(X_train.shape[0])

        for idx in permutation:
            rates_window = X_train[idx]
            true_label = int(y_train[idx])

            hidden_counts, output_counts, pred_label = run_window(
                net,
                input_group,
                syn_hidden_out,
                hidden_spike_monitor,
                output_spike_monitor,
                rates_window,
                w_hidden_out,
            )

            if pred_label == true_label:
                correct += 1

            total_output_spikes += float(np.sum(output_counts))

            correct_idx = true_label - 1
            pred_idx = pred_label - 1

            w_hidden_out[:, correct_idx] += ETA * hidden_counts
            if pred_label != true_label:
                w_hidden_out[:, pred_idx] -= ETA * hidden_counts

            w_hidden_out = np.clip(w_hidden_out, 0.0, 1.0)

        epoch_acc = correct / X_train.shape[0]
        avg_output_spikes = total_output_spikes / X_train.shape[0]
        print(
            f"Epoch {epoch + 1}/{N_EPOCHS} training accuracy: {epoch_acc:.4f} | "
            f"avg output spike count: {avg_output_spikes:.4f}"
        )

    predictions = np.zeros(X_test.shape[0], dtype=np.int32)
    for i, rates_window in enumerate(X_test):
        _, _, pred_label = run_window(
            net,
            input_group,
            syn_hidden_out,
            hidden_spike_monitor,
            output_spike_monitor,
            rates_window,
            w_hidden_out,
        )
        predictions[i] = pred_label

    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="macro")
    cm = confusion_matrix(y_test, predictions, labels=[1, 2])

    print(f"Test accuracy: {acc:.4f}")
    print(f"Test macro F1-score: {f1:.4f}")
    print("Confusion matrix (rows=true [1,2], cols=pred [1,2]):")
    print(cm)

    os.makedirs(os.path.dirname(output_weights_path), exist_ok=True)
    np.savez(
        output_weights_path,
        w_in_hidden=w_in_hidden,
        w_hidden_out=w_hidden_out,
        sim_duration_ms=SIM_DURATION_MS,
        r_max_hz=R_MAX_HZ,
        n_hidden=N_HIDDEN,
        rng_seed=RNG_SEED,
    )

    plot_confusion_matrix(cm, output_cm_path)
    print(f"Saved trained weights to: {output_weights_path}")
    print(f"Saved confusion matrix plot to: {output_cm_path}")


if __name__ == "__main__":
    main()
