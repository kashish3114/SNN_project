#!/usr/bin/env python3
"""Step 11: Manual EMG inference using trained SNN weights from step 9."""

from __future__ import annotations

import os

import numpy as np

WEIGHTS_PATH = "../data/processed/snn_trained_weights.npz"
WINDOW_LEN = 200
N_CHANNELS = 8
SIM_DURATION_MS = 200
DEFAULT_R_MAX_HZ = 200.0
RNG_SEED = 42


def create_sample_input() -> np.ndarray:
    """
    Create a sample EMG window with shape (200, 8).

    Modify this function to test manual EMG values. You can replace the random
    values below with a hand-crafted NumPy array if preferred.
    """
    rng = np.random.default_rng(RNG_SEED)
    sample = rng.normal(loc=0.0, scale=0.25, size=(WINDOW_LEN, N_CHANNELS))

    # Example pattern: boost a few channels to resemble stronger activation.
    sample[:, 2] += 0.35
    sample[:, 5] += 0.20
    return sample


def compute_rms_per_channel(emg_window: np.ndarray) -> np.ndarray:
    """Compute RMS across time for each channel."""
    if emg_window.shape != (WINDOW_LEN, N_CHANNELS):
        raise ValueError(
            f"Expected EMG window shape ({WINDOW_LEN}, {N_CHANNELS}), got {emg_window.shape}"
        )
    return np.sqrt(np.mean(np.square(emg_window), axis=0))


def rms_to_spike_rates(rms_values: np.ndarray, r_max_hz: float) -> np.ndarray:
    """Scale RMS values to spike rates in the range [0, r_max_hz]."""
    rms_values = np.clip(rms_values.astype(np.float64), 0.0, None)
    max_rms = np.max(rms_values)
    if max_rms == 0:
        return np.zeros_like(rms_values)
    return (rms_values / max_rms) * r_max_hz


def build_snn(w_in_hidden: np.ndarray, w_hidden_out: np.ndarray):
    """Build the trained SNN architecture for one-window inference."""
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

    start_scope()
    prefs.codegen.target = "numpy"

    input_group = PoissonGroup(N_CHANNELS, rates=np.zeros(N_CHANNELS) * Hz)

    lif_model = """
    dv/dt = (v_rest - v)/tau : 1
    v_rest : 1 (constant)
    tau : second (constant)
    """

    hidden_group = NeuronGroup(
        w_in_hidden.shape[1],
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

    output_spike_monitor = SpikeMonitor(output_group)

    net = Network(
        input_group,
        hidden_group,
        output_group,
        syn_in_hidden,
        syn_hidden_out,
        output_spike_monitor,
    )
    return net, input_group, output_spike_monitor


def load_trained_weights(weights_path: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Load trained weights and rate-scaling metadata from step 9."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            "Trained weights not found. Run scripts/step9_train_snn.py first to create "
            f"{weights_path}."
        )

    data = np.load(weights_path)
    required_keys = {"w_in_hidden", "w_hidden_out"}
    missing = required_keys.difference(data.files)
    if missing:
        raise KeyError(f"Missing keys in weights file: {sorted(missing)}")

    w_in_hidden = data["w_in_hidden"]
    w_hidden_out = data["w_hidden_out"]
    r_max_hz = float(data["r_max_hz"]) if "r_max_hz" in data else DEFAULT_R_MAX_HZ
    return w_in_hidden, w_hidden_out, r_max_hz


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.normpath(os.path.join(script_dir, WEIGHTS_PATH))

    w_in_hidden, w_hidden_out, r_max_hz = load_trained_weights(weights_path)
    emg_window = create_sample_input()
    rms_values = compute_rms_per_channel(emg_window)
    spike_rates = rms_to_spike_rates(rms_values, r_max_hz)

    try:
        from brian2 import Hz, ms
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Brian2 is required for this script. Install it with: pip install brian2"
        ) from exc

    net, input_group, output_spike_monitor = build_snn(w_in_hidden, w_hidden_out)
    input_group.rates = spike_rates * Hz
    net.run(SIM_DURATION_MS * ms)

    output_counts = np.asarray(np.bincount(output_spike_monitor.i, minlength=2), dtype=np.int32)
    winner = int(np.argmax(output_counts))
    predicted_label = "Rest" if winner == 0 else "Fist"

    print("RMS per channel:")
    print(np.round(rms_values, 4))
    print("\nSpike rates (Hz):")
    print(np.round(spike_rates, 2))
    print("\nOutput spike counts [Rest, Fist]:")
    print(output_counts.tolist())
    print(f"Predicted class: {predicted_label}")


if __name__ == "__main__":
    main()
