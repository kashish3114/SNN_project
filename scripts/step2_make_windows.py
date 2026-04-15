"""Step 2: Window raw EMG data into fixed-length segments.

Requirements implemented:
- Load EMG files from data/raw/EMG_data_for_gestures-master/
- Skip header and malformed rows
- Remove samples with label 0 (ignored during windowing)
- Sliding windows: length=200, stride=100
- Majority-vote label; drop windows with mixed labels or any label 0
- Track subject ID for each window
- Save X, y, subjects to data/processed/windows_rest_fist.npz
"""

import os
from collections import Counter
from typing import List, Tuple

import numpy as np

ROOT_DIR = "../data/raw/EMG_data_for_gestures-master"
OUT_PATH = "../data/processed/windows_rest_fist.npz"
WINDOW_LEN = 200
STRIDE = 100  # 50% overlap

# Only keep Rest (1) and Fist (2) windows for the initial 2-class task.
VALID_LABELS = {1, 2}


def iter_subject_dirs(root_dir: str) -> List[str]:
    return sorted([d for d in os.listdir(root_dir) if d.isdigit()])


def load_file(path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load a single EMG file.

    Returns:
        emg: (N, 8) float32
        labels: (N,) int32
        bad_rows: count of malformed rows skipped
    """
    emg_rows = []
    label_rows = []
    bad_rows = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # Skip header row
            if parts and parts[0].lower() == "time":
                continue
            if len(parts) != 10:
                bad_rows += 1
                continue
            try:
                # Columns: time, ch1..ch8, label
                ch = [float(x) for x in parts[1:9]]
                label = int(float(parts[9]))
            except ValueError:
                bad_rows += 1
                continue

            emg_rows.append(ch)
            label_rows.append(label)

    if not emg_rows:
        return np.empty((0, 8), dtype=np.float32), np.empty((0,), dtype=np.int32), bad_rows

    emg = np.asarray(emg_rows, dtype=np.float32)
    labels = np.asarray(label_rows, dtype=np.int32)
    return emg, labels, bad_rows


def window_file(
    emg: np.ndarray, labels: np.ndarray, subject_id: int
) -> Tuple[List[np.ndarray], List[int], List[int]]:
    """Create windows for a single file.

    Windows are dropped if:
    - any label is 0 (transition), or
    - labels are mixed within the window, or
    - label is not in VALID_LABELS.
    """
    windows = []
    win_labels = []
    win_subjects = []

    n = emg.shape[0]
    if n < WINDOW_LEN:
        return windows, win_labels, win_subjects

    for start in range(0, n - WINDOW_LEN + 1, STRIDE):
        end = start + WINDOW_LEN
        w_emg = emg[start:end]
        w_labels = labels[start:end]

        # Drop any window containing label 0
        if np.any(w_labels == 0):
            continue
        # Enforce consistent label within window
        if not np.all(w_labels == w_labels[0]):
            continue
        label = int(w_labels[0])
        if label not in VALID_LABELS:
            continue

        windows.append(w_emg)
        win_labels.append(label)
        win_subjects.append(subject_id)

    return windows, win_labels, win_subjects


def main() -> None:
    subjects = iter_subject_dirs(ROOT_DIR)
    all_windows: List[np.ndarray] = []
    all_labels: List[int] = []
    all_subjects: List[int] = []

    total_bad_rows = 0

    for subj in subjects:
        subj_dir = os.path.join(ROOT_DIR, subj)
        files = sorted([f for f in os.listdir(subj_dir) if f.endswith(".txt")])
        subject_id = int(subj)

        for fname in files:
            path = os.path.join(subj_dir, fname)
            emg, labels, bad_rows = load_file(path)
            total_bad_rows += bad_rows

            windows, win_labels, win_subjects = window_file(emg, labels, subject_id)
            all_windows.extend(windows)
            all_labels.extend(win_labels)
            all_subjects.extend(win_subjects)

    if not all_windows:
        raise RuntimeError("No windows were produced. Check data paths and parameters.")

    X = np.stack(all_windows, axis=0)
    y = np.asarray(all_labels, dtype=np.int32)
    subjects_arr = np.asarray(all_subjects, dtype=np.int32)

    # Sanity checks
    class_counts = Counter(y.tolist())
    subject_counts = Counter(subjects_arr.tolist())

    print("Total windows:", X.shape[0])
    print("Windows per class:", dict(sorted(class_counts.items())))
    print("Windows per subject:", dict(sorted(subject_counts.items())))
    print("Total malformed rows skipped:", total_bad_rows)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(OUT_PATH, X=X, y=y, subjects=subjects_arr)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
