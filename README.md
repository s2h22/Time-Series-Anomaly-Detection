# Time Series Anomaly Detection with LSTM

This project implements an LSTM-based encoder-decoder model for detecting anomalies in multivariate time series data, developed as part of a technical assessment.

## 📊 Overview

The model learns the normal pattern of sensor readings and identifies deviations as anomalies using reconstruction error and thresholding. It was trained and evaluated on a real-world pump sensor dataset from Kaggle.

- Dataset: Multivariate time series from 52 pump sensors
- Normal/Abnormal: Abnormal states include BROKEN and RECOVERING
- Model: LSTM encoder-decoder with MSE loss
- Detection: Threshold-based anomaly score using Mahalanobis distance
- Evaluation: Precision, Recall, F1-score
- Visualization: Score plots and anomaly markers

## 📁 Dataset

- Source: [Pump Sensor Data from Kaggle](https://www.kaggle.com/nphantawee/pump-sensor-data)
- Preprocessing:
  - Normalization
  - Removal of sensors with high missing values
  - Train/Val/Test split with normal and abnormal labels

## 🧠 Model Architecture

- Encoder: LSTM layers to compress time series into latent vector
- Decoder: LSTM to reconstruct sequence from latent vector
- Loss: Mean Squared Error (MSE)
- Anomaly Score: Mahalanobis distance from reconstruction error

## 📈 Evaluation

Metrics:
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: 2 * (Precision * Recall) / (Precision + Recall)

An ablation study was performed to assess the impact of:
- Window size
- LSTM layers
- Latent size
- Decoder FC layer

## 📦 Requirements

Install the necessary Python packages using:

```bash
pip install -r requirements.txt
