#!/usr/bin/env python3
"""Step 10: Compare model accuracies and save a summary bar chart."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt

OUTPUT_PLOT_PATH = "../plots/model_comparison.png"


def plot_accuracies(models: list[str], accuracies: list[float], out_path: str) -> None:
    """Plot model accuracies as a labeled bar chart and save it."""
    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        models,
        accuracies,
        color=["#4C78A8", "#F58518", "#54A24B", "#E45756"],
    )

    plt.title("Model Performance Comparison")
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)

    for bar, accuracy in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{accuracy:.4f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def print_table(models: list[str], accuracies: list[float]) -> None:
    """Print the model comparison table."""
    print("Model | Accuracy")
    for model, accuracy in zip(models, accuracies):
        print(f"{model} | {accuracy:.4f}")


def main() -> None:
    models = [
        "Logistic Regression",
        "SNN (untrained)",
        "SNN (trained)",
        "MLP",
    ]
    accuracies = [0.50, 0.56, 0.6735, 0.985]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_plot_path = os.path.normpath(os.path.join(script_dir, OUTPUT_PLOT_PATH))

    print_table(models, accuracies)
    plot_accuracies(models, accuracies, output_plot_path)
    print(f"\nSaved plot to: {output_plot_path}")


if __name__ == "__main__":
    main()
