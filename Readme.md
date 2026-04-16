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
3. step4_mlp_baseline.py  
4. step5_spike_encoding.py  
5. step9_train_snn.py  
6. step10_model_comparison.py  
7. step11_manual_inference.py  

## Requirements
- brian2
- numpy
- matplotlib
- scikit-learn

Install using:
pip install -r requirements.txt

## Dataset
This project uses the UCI EMG Dataset for gesture classification.

Download from:
https://archive.ics.uci.edu/ml/datasets/EMG+data+for+gestures
## Dataset Setup
Download from:
https://archive.ics.uci.edu/ml/datasets/EMG+data+for+gestures

Place in:
data/raw/

## Motivation
SNNs provide biologically inspired, event-driven computation and potential
energy efficiency advantages over traditional neural networks.

## Key Findings
- MLP achieves ~98.5% accuracy, indicating strong feature separability
- SNN improves from ~56% to ~67% after training
- SNN does not match MLP due to limitations in rate coding and training method
- Highlights trade-off between biological realism and performance
- Subject-wise split ensures no overlap between training and testing subjects
  
## Author
Kashish Pranav Shah
