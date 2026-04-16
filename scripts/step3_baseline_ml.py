#!/usr/bin/env python3
"""
Step 3: Baseline ML classifier for EMG gesture recognition.

Loads windowed EMG data, extracts RMS and MAV features per channel,
trains a Logistic Regression classifier, and evaluates performance.
"""

from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


def load_npz(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load X, y, subjects arrays from a .npz file."""
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
    # RMS: sqrt(mean(x^2)) across time axis (window_len)
    rms = np.sqrt(np.mean(np.square(X), axis=1))
    # MAV: mean(abs(x)) across time axis
    mav = np.mean(np.abs(X), axis=1)

    # Concatenate [rms(8), mav(8)] -> 16 features
    features = np.concatenate([rms, mav], axis=1)
    return features


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """Train/test split evaluation with stratification."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        multi_class="auto",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    print("Stratified 80/20 split results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1-score: {f1:.4f}")
    print("Confusion matrix:")
    print(cm)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Rest", "Fist"], yticklabels=["Rest", "Fist"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Logistic Regression Confusion Matrix (80/20 Split)")
    plt.tight_layout()
    plt.savefig("lr_confusion_matrix_split.png")
    plt.close()


def subject_wise_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    *,
    train_subjects: range = range(1, 31),
    test_subjects: range = range(31, 37),
) -> None:
    """Train on a subject range and test on a held-out subject range."""
    train_mask = np.isin(subjects, list(train_subjects))
    test_mask = np.isin(subjects, list(test_subjects))

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    if X_train.size == 0 or X_test.size == 0:
        print("\nSubject-wise split skipped: no samples for train or test subjects.")
        return

    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        multi_class="auto",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    print("\nSubject-wise evaluation (train subjects 1-30, test subjects 31-36):")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1-score: {f1:.4f}")
    print("Confusion matrix:")
    print(cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Rest", "Fist"], yticklabels=["Rest", "Fist"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Logistic Regression Confusion Matrix (Subject-wise)")
    plt.tight_layout()
    plt.savefig("lr_confusion_matrix_subject.png")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 3 baseline ML classifier for EMG gesture recognition."
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

    train_and_evaluate(features, y)

    if not args.no_subject_eval:
        subject_wise_evaluation(features, y, subjects)


if __name__ == "__main__":
    main()
