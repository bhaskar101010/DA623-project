# Human Activity Recognition Using Mobile Sensors

## Overview

This repository contains a machine learning pipeline for predicting human activity using data from mobile sensors. The project leverages a multivariate classification approach on sensor data that has been pre-aggregated (averaged), making it a tabular classification problem rather than a time series analysis.

## Motivation

Human Activity Recognition (HAR) is foundational for applications in health monitoring, fitness tracking, and smart environments. With the proliferation of smartphones and wearables, sensor-based HAR enables unobtrusive, real-time monitoring of daily activities. This project aims to build a robust classifier to predict human activities from high-dimensional sensor data.

## Project Structure

- **DA623_project.ipynb** – Jupyter notebook containing the complete data processing and modeling pipeline
- **test.csv** – Input CSV file with 561 sensor features and an "Activity" label column

## Approach

1. **Data Preprocessing**
   - Handles missing values using mean imputation
   - Removes outliers using z-score filtering

2. **Feature Engineering**
   - Standardizes features using `StandardScaler`
   - Reduces dimensionality with PCA, retaining 95% of variance (reducing to 89 components)

3. **Modeling**
   - Splits data into training and test sets (80/20, stratified)
   - Trains a neural network (MLPClassifier)
   - Optimizes hyperparameters using RandomizedSearchCV

4. **Evaluation**
   - Model performance can be evaluated using accuracy and classification reports (see notebook for details)

## How to Run

1. Clone this repository.
2. Ensure you have the required Python libraries:  
   `pandas`, `numpy`, `scikit-learn`, `scipy`
3. Place your cleaned sensor data as `test.csv` in the repository directory.
4. Open and run `DA623_project.ipynb` in Jupyter Notebook or Google Colab.

## Key Learnings

- Effective data cleaning and preprocessing (imputation, outlier removal) are critical for robust model performance.
- Dimensionality reduction (PCA) is essential for high-dimensional sensor data.
- Hyperparameter tuning significantly impacts neural network classifier results.
- Averaged (non-time series) sensor data can still yield strong activity classification results, though temporal patterns are lost.

## Reflections

- **Surprises:** PCA reduced 561 features to 89 while preserving most information; preprocessing had a major impact on results.
- **Scope for Improvement:**  
  - Use raw time series data for temporal modeling (RNNs, LSTMs)
  - Explore multimodal data fusion (e.g., combining sensor and video data)
  - Deploy the model for real-time inference on mobile devices

## References

- UCI HAR Dataset

---

**Author:**  
Bhaskar Dev 
9/5/25


