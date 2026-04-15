#!/usr/bin/env python3
"""Step 7: Build a simple Brian2 SNN for EMG classification."""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

try:
    from brian2 import Hz, NeuronGroup, PoissonGroup, SpikeMonitor, Synapses, ms, run, start_scope
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Brian2 is required for this script. Install it with: pip install brian2"
    ) from exc

INPUT_PATH = "../data/processed/spike_rates.npz"
OUTPUT_PLOT_PATH = "../plots/output_spikes.png"
WINDOW_INDEX = 0
SIM_DURATION_MS = 100
RNG_SEED = 42


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.normpath(os.path.join(script_dir, INPUT_PATH))
    output_plot_path = os.path.normpath(os.path.join(script_dir, OUTPUT_PLOT_PATH))

    data = np.load(input_path)
    if "spike_rates" not in data:
        raise KeyError(
            f"'spike_rates' key not found in {input_path}. Available keys: {list(data.files)}"
        )

    spike_rates = data["spike_rates"]
    if spike_rates.ndim != 2 or spike_rates.shape[1] != 8:
        raise ValueError(f"Expected spike_rates shape (N, 8), got {spike_rates.shape}")
    if WINDOW_INDEX < 0 or WINDOW_INDEX >= spike_rates.shape[0]:
        raise IndexError(
            f"WINDOW_INDEX={WINDOW_INDEX} out of bounds for N={spike_rates.shape[0]}"
        )

    rates_window = spike_rates[WINDOW_INDEX].astype(np.float64)  # Hz for 8 input neurons

    np.random.seed(RNG_SEED)
    start_scope()

    # Input layer: Poisson spike generators from rate-coded EMG features.
    input_group = PoissonGroup(8, rates=rates_window * Hz)

    # LIF dynamics requested by specification.
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

    # Dense synapses with random weights.
    syn_in_hidden = Synapses(input_group, hidden_group, model="w : 1", on_pre="v_post += w")
    syn_in_hidden.connect()
    syn_in_hidden.w = "0.0 + 0.30 * rand()"

    syn_hidden_out = Synapses(hidden_group, output_group, model="w : 1", on_pre="v_post += w")
    syn_hidden_out.connect()
    syn_hidden_out.w = "0.0 + 0.60 * rand()"

    hidden_spike_monitor = SpikeMonitor(hidden_group)
    output_spike_monitor = SpikeMonitor(output_group)

    run(SIM_DURATION_MS * ms)

    hidden_spike_count = int(hidden_spike_monitor.num_spikes)
    output_spike_count = int(output_spike_monitor.num_spikes)

    # Winner-takes-most prediction from output neuron spike counts.
    output_counts = np.bincount(output_spike_monitor.i, minlength=2)
    pred_idx = int(np.argmax(output_counts))
    pred_label = "Rest" if pred_idx == 0 else "Fist"

    # Raster for output neurons only.
    plt.figure(figsize=(7, 3.5))
    plt.scatter(output_spike_monitor.t / ms, output_spike_monitor.i, s=22, color="black")
    plt.xlabel("Time (ms)")
    plt.ylabel("Output neuron index")
    plt.yticks([0, 1])
    plt.title("Output Spike Raster")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(output_plot_path, dpi=150)
    plt.close()

    print(f"Selected window index: {WINDOW_INDEX}")
    print(f"Hidden spike count: {hidden_spike_count}")
    print(f"Output spike count: {output_spike_count}")
    print(f"Output neuron spike counts [0,1]: {output_counts.tolist()}")
    print(f"Predicted class (winner-takes-most): {pred_idx} ({pred_label})")
    print(f"Saved output raster: {output_plot_path}")


if __name__ == "__main__":
    main()
