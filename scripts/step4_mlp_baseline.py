#!/usr/bin/env python3
"""
Step 4: MLP baseline for EMG window classification.

Loads windowed EMG data, extracts RMS and MAV features per channel,
trains an MLPClassifier, and evaluates performance.
"""

from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def load_npz(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load X, y, and subjects arrays from a .npz file."""
    data = np.load(path)
    X = data["X"]
    y = data["y"]
    subjects = data["subjects"]
    return X, y, subjects


def extract_features(X: np.ndarray) -> np.ndarray:
    """
    Extract RMS and MAV features per window per channel.

    Input shape: (num_windows, window_len, 8)
    Output shape: (num_windows, 16)
    """
    # RMS: sqrt(mean(x^2)) across time axis
    rms = np.sqrt(np.mean(np.square(X), axis=1))
    # MAV: mean(abs(x)) across time axis
    mav = np.mean(np.abs(X), axis=1)

    # Concatenate [rms(8), mav(8)] -> 16 features
    features = np.concatenate([rms, mav], axis=1)
    return features


def evaluate_split(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Fit an MLP and print accuracy, macro F1, and confusion matrix."""
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=500,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1-score: {f1:.4f}")
    print("Confusion matrix:")
    print(cm)


def stratified_split_eval(X: np.ndarray, y: np.ndarray) -> None:
    """Stratified 80/20 train-test evaluation."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    print("Stratified 80/20 split results:")
    evaluate_split(X_train, y_train, X_test, y_test)


def subject_wise_eval(X: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> None:
    """Subject-wise evaluation: train on subjects 1-30, test on 31-36."""
    train_mask = np.isin(subjects, list(range(1, 31)))
    test_mask = np.isin(subjects, list(range(31, 37)))

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    if X_train.size == 0 or X_test.size == 0:
        print("\nSubject-wise split skipped: no samples for train or test subjects.")
        return

    print("\nSubject-wise evaluation (train subjects 1-30, test subjects 31-36):")
    evaluate_split(X_train, y_train, X_test, y_test)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 4 MLP baseline for EMG window classification."
    )
    parser.add_argument(
        "--data",
        default="../data/processed/windows_rest_fist.npz",
        help="Path to processed .npz data file.",
    )
    parser.add_argument(
        "--no-subject-eval",
        action="store_true",
        help="Disable subject-wise evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    X, y, subjects = load_npz(args.data)
    features = extract_features(X)

    stratified_split_eval(features, y)

    if not args.no_subject_eval:
        subject_wise_eval(features, y, subjects)


if __name__ == "__main__":
    main()
