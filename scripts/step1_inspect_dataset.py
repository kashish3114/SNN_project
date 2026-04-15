"""Step 1: Dataset loading and inspection (no ML/SNN)."""

import os
from collections import Counter


def iter_subject_dirs(root_dir: str):
    return sorted([d for d in os.listdir(root_dir) if d.isdigit()])


def summarize_dataset(root_dir: str):
    subjects = iter_subject_dirs(root_dir)
    label_counts = Counter()
    file_counts = 0
    sample_counts = 0
    header_lines = 0
    col_mismatch = []

    for subj in subjects:
        subj_dir = os.path.join(root_dir, subj)
        files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.txt')])
        file_counts += len(files)
        for fname in files:
            path = os.path.join(subj_dir, fname)
            with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
                for line_num, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if parts and parts[0].lower() == 'time':
                        header_lines += 1
                        continue
                    if len(parts) != 10:
                        col_mismatch.append((path, line_num, len(parts)))
                        continue
                    sample_counts += 1
                    try:
                        label = int(float(parts[9]))
                    except ValueError:
                        col_mismatch.append((path, line_num, 'label_parse'))
                        continue
                    label_counts[label] += 1

    summary = {
        'subjects': len(subjects),
        'files': file_counts,
        'samples': sample_counts,
        'header_lines': header_lines,
        'label_counts': dict(sorted(label_counts.items())),
        'col_mismatch_count': len(col_mismatch),
        'col_mismatch_examples': col_mismatch[:5],
    }
    return summary


if __name__ == '__main__':
    root = '../data/raw/EMG_data_for_gestures-master'
    summary = summarize_dataset(root)
    for key, value in summary.items():
        print(f"{key}: {value}")
