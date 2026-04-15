#!/usr/bin/env python3
"""Step 5: Convert EMG windows to spike-rate features for SNN experiments."""

from __future__ import annotations

import os
import numpy as np

R_MAX_HZ = 200.0
INPUT_PATH = "../data/processed/windows_rest_fist.npz"
OUTPUT_PATH = "../data/processed/spike_rates.npz"


def _first_available(data: np.lib.npyio.NpzFile, keys: tuple[str, ...]) -> np.ndarray:
    """Return the first key that exists in the npz file."""
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"None of these keys were found: {keys}. Available keys: {list(data.files)}")


def _to_n8t_shape(x_windows: np.ndarray) -> np.ndarray:
    """
    Convert window tensor to shape (N, 8, window_size).

    Supports both:
    - (N, 8, window_size)
    - (N, window_size, 8)
    """
    if x_windows.ndim != 3:
        raise ValueError(f"Expected 3D windows array, got shape {x_windows.shape}")

    if x_windows.shape[1] == 8:
        return x_windows
    if x_windows.shape[2] == 8:
        return np.transpose(x_windows, (0, 2, 1))

    raise ValueError(
        f"Could not identify channel axis (size 8) in input shape {x_windows.shape}."
    )


def compute_rms_per_channel(x_n8t: np.ndarray) -> np.ndarray:
    """Compute RMS over time for each channel in each window, output shape (N, 8)."""
    return np.sqrt(np.mean(np.square(x_n8t), axis=2))


def normalize_per_window(rms: np.ndarray) -> np.ndarray:
    """
    Normalize each window's 8-channel RMS vector to [0, 1] using min-max scaling.

    If all channels are equal in a window, output zeros for that window.
    """
    rms_min = np.min(rms, axis=1, keepdims=True)
    rms_max = np.max(rms, axis=1, keepdims=True)
    denom = rms_max - rms_min

    normalized = np.divide(
        rms - rms_min,
        denom,
        out=np.zeros_like(rms, dtype=np.float32),
        where=denom > 0,
    )
    return normalized.astype(np.float32)


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.normpath(os.path.join(script_dir, INPUT_PATH))
    output_path = os.path.normpath(os.path.join(script_dir, OUTPUT_PATH))

    data = np.load(input_path)

    # Support requested key names plus compatibility with existing dataset keys.
    x_windows = _first_available(data, ("X_windows", "X"))
    y_labels = _first_available(data, ("y_labels", "y"))
    subject_ids = _first_available(data, ("subject_ids", "subjects"))

    x_n8t = _to_n8t_shape(x_windows)
    rms = compute_rms_per_channel(x_n8t)  # (N, 8)
    rms_norm = normalize_per_window(rms)  # (N, 8)
    spike_rates = (rms_norm * R_MAX_HZ).astype(np.float32)  # (N, 8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(
        output_path,
        spike_rates=spike_rates,
        y_labels=y_labels,
        subject_ids=subject_ids,
        r_max_hz=np.array(R_MAX_HZ, dtype=np.float32),
    )

    # Print examples for quick sanity check.
    print(f"Loaded windows shape: {x_windows.shape}")
    print(f"Spike rates shape: {spike_rates.shape}")
    print("Example RMS values (window 0):")
    print(np.round(rms[0], 6))
    print("Example spike rates in Hz (window 0):")
    print(np.round(spike_rates[0], 6))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
