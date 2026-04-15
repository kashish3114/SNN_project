#!/usr/bin/env python3
"""Step 8: Run full-dataset SNN inference (no training) and report metrics."""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

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
OUTPUT_CM_PATH = "../plots/snn_confusion_matrix.png"
SIM_DURATION_MS = 100
RNG_SEED = 42


def _first_available(data: np.lib.npyio.NpzFile, keys: tuple[str, ...]) -> np.ndarray:
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"None of these keys were found: {keys}. Available keys: {list(data.files)}")


def build_snn(w_in_hidden: np.ndarray, w_hidden_out: np.ndarray):
    """Build step7 architecture once and return reusable network objects."""
    start_scope()

    # Use NumPy backend to avoid per-run compilation overhead.
    prefs.codegen.target = "numpy"
    input_group = PoissonGroup(8, rates=np.zeros(8) * Hz)

    lif_model = """
    dv/dt = (v_rest - v)/tau : 1
    v_rest : 1 (constant)
    tau : second (constant)
    """

    hidden_group = NeuronGroup(
        30,
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
    return net, input_group, output_spike_monitor


def plot_confusion_matrix(cm: np.ndarray, out_path: str) -> None:
    """Plot and save confusion matrix image."""
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("SNN Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["Rest (1)", "Fist (2)"])
    plt.yticks([0, 1], ["Rest (1)", "Fist (2)"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    # Annotate cells for readability.
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

    data = np.load(input_path)
    spike_rates = _first_available(data, ("spike_rates",))
    y_true = _first_available(data, ("y_labels", "y"))

    if spike_rates.ndim != 2 or spike_rates.shape[1] != 8:
        raise ValueError(f"Expected spike_rates shape (N, 8), got {spike_rates.shape}")

    if y_true.shape[0] != spike_rates.shape[0]:
        raise ValueError(
            f"Mismatched sample counts: spike_rates N={spike_rates.shape[0]}, y N={y_true.shape[0]}"
        )

    # Fixed random weights for all windows (no training).
    rng = np.random.default_rng(RNG_SEED)
    w_in_hidden = rng.uniform(0.0, 0.30, size=(8, 30))
    w_hidden_out = rng.uniform(0.0, 0.60, size=(30, 2))

    predictions = np.zeros(spike_rates.shape[0], dtype=np.int32)

    net, input_group, output_spike_monitor = build_snn(w_in_hidden, w_hidden_out)
    net.store("initial")

    for i in range(spike_rates.shape[0]):
        rates_window = spike_rates[i].astype(np.float64)
        net.restore("initial")
        input_group.rates = rates_window * Hz
        net.run(SIM_DURATION_MS * ms)

        counts = np.bincount(output_spike_monitor.i, minlength=2)
        winner = int(np.argmax(counts))  # 0 -> Rest, 1 -> Fist
        predictions[i] = 1 if winner == 0 else 2

    acc = accuracy_score(y_true, predictions)
    f1 = f1_score(y_true, predictions, average="macro")
    cm = confusion_matrix(y_true, predictions, labels=[1, 2])

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1-score: {f1:.4f}")
    print("Confusion matrix (rows=true [1,2], cols=pred [1,2]):")
    print(cm)

    plot_confusion_matrix(cm, output_cm_path)
    print(f"Saved confusion matrix plot to: {output_cm_path}")


if __name__ == "__main__":
    main()
