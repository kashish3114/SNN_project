# EMG Gesture Classification using SNN (Brian2)

## Overview
This project implements a full pipeline for EMG-based gesture classification
using both conventional ML models and a Spiking Neural Network (SNN).

## Pipeline
- Data loading & preprocessing
- Sliding window segmentation
- Feature extraction (RMS, MAV)
- Baseline models (Logistic Regression, MLP)
- Spike encoding (rate coding)
- Brian2 SNN implementation
- Evaluation (subject-wise split)

## Results
- Logistic Regression: ~50%
- MLP: ~98.5%
- SNN (trained): ~67%

## How to Run
1. step1_preprocess.py
2. step3_baseline_ml.py
3. step9_train_snn.py
4. step11_manual_inference.py

## Dataset
UCI EMG Dataset

## Author
Kashish Pranav Shah
