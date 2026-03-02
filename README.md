Predictive Maintenance using Time-Series Machine Learning
Project Overview

This project builds an end-to-end machine learning pipeline to predict the Remaining Useful Life (RUL) of aircraft turbofan engines using NASA’s CMAPSS sensor dataset.

Predictive maintenance is widely used in manufacturing, automotive, and energy industries to reduce downtime and maintenance costs.

Problem Statement

Given multivariate sensor data from aircraft engines, predict how many cycles remain before engine failure.

This is a time-series regression problem.

Pipeline Architecture

Data Pipeline → Feature Engineering → Model Training → Experiment Tracking → Evaluation

Project components:

Data processing pipeline (Python scripts)

Rolling-window time-series feature engineering

Classical ML model (XGBoost)

Deep learning model (LSTM)

Experiment tracking with MLflow

Evaluation and visualization

Dataset

NASA Turbofan Engine Degradation Simulation (CMAPSS)

Each engine contains:

21 sensor signals

3 operational settings

Full run-to-failure trajectories

Target variable:
Remaining Useful Life (RUL)

Feature Engineering

Time-series were converted into tabular features using rolling windows (30 cycles):

For each sensor:

Rolling mean

Rolling standard deviation

Rolling minimum / maximum

This simulates real industrial predictive-maintenance pipelines.

Models
1️⃣ XGBoost (Classical ML)

Trained on engineered tabular features.

Performance:

RMSE: 9.64 cycles

MAE: 6.82 cycles

2️⃣ LSTM (Deep Learning)

Trained on raw time-series sequences.

Performance:

Higher error than XGBoost (feature engineering proved more effective)

Key Insights

Feature engineering significantly improves classical ML performance.

XGBoost outperformed LSTM on this dataset.

Experiment tracking with MLflow ensures reproducibility.

Tech Stack

Python • Pandas • Scikit-learn • XGBoost • PyTorch • MLflow