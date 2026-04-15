#!/usr/bin/env python3
"""Step 6: Generate Poisson spikes from rate-coded EMG features using Brian2."""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

try:
    from brian2 import Hz, PoissonGroup, SpikeMonitor, ms, run, start_scope
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Brian2 is required for this script. Install it with: pip install brian2"
    ) from exc

INPUT_PATH = "../data/processed/spike_rates.npz"
OUTPUT_FIG_PATH = "../plots/spike_raster.png"
WINDOW_INDEX = 0
SIM_DURATION_MS = 100


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.normpath(os.path.join(script_dir, INPUT_PATH))
    output_fig_path = os.path.normpath(os.path.join(script_dir, OUTPUT_FIG_PATH))

    data = np.load(input_path)
    if "spike_rates" not in data:
        raise KeyError(
            f"'spike_rates' key not found in {input_path}. Available keys: {list(data.files)}"
        )

    spike_rates = data["spike_rates"]
    if spike_rates.ndim != 2 or spike_rates.shape[1] != 8:
        raise ValueError(
            f"Expected spike_rates shape (N, 8), got {spike_rates.shape}"
        )

    rates_window = spike_rates[WINDOW_INDEX].astype(np.float64)  # shape: (8,)

    # Reset Brian2 state and build an 8-neuron Poisson population.
    start_scope()
    poisson_group = PoissonGroup(8, rates=rates_window * Hz)
    spike_monitor = SpikeMonitor(poisson_group)

    run(SIM_DURATION_MS * ms)

    # Raster plot: x-axis time (ms), y-axis neuron index.
    plt.figure(figsize=(8, 4))
    plt.scatter(spike_monitor.t / ms, spike_monitor.i, s=12, color="black")
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron index")
    plt.yticks(range(8))
    plt.title("Spike Raster (Window 0)")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)
    plt.savefig(output_fig_path, dpi=150)
    plt.close()

    print(f"Total spikes generated: {spike_monitor.num_spikes}")
    print(f"Saved raster plot to: {output_fig_path}")


if __name__ == "__main__":
    main()
