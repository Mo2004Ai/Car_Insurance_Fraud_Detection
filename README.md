# Car Insurance Fraud Detection System

## Overview
This project is an end-to-end Machine Learning system designed to detect fraudulent car insurance claims. It covers the full pipeline including data analysis, preprocessing, model training, and deployment using FastAPI and Streamlit. The system is specifically designed to handle the class imbalance problem inherent in fraud detection datasets.

## Key Features
* Data cleaning and preprocessing pipeline.
* Handling imbalanced data using multiple techniques:
    * Random OverSampling
    * Random UnderSampling
    * SMOTE (Synthetic Minority Over-sampling Technique)
* Implementation of multiple machine learning models:
    * Decision Tree
    * XGBoost
    * Voting Classifier (Ensemble)
    * Stacking Classifier (Ensemble)
* Threshold optimization to improve fraud detection metrics (Recall and F1-Score).
* Model serialization using joblib for production use.
* REST API development using FastAPI.
* Interactive user interface using Streamlit.

## Models Performance (After Threshold Tuning)
| Model | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: |
| Decision Tree | 0.31 | 0.43 | 0.36 |
| XGBoost | 0.31 | 0.40 | 0.35 |
| Voting Classifier | 0.33 | 0.54 | 0.41 |
| Stacking | 0.30 | 0.52 | 0.38 |

## Best Thresholds
| Model | Threshold |
| :--- | :---: |
| Decision Tree | 0.74 |
| XGBoost | 0.30 |
| Voting Classifier | 0.73 |
| Stacking | 0.60 |

## Project Structure
```text
project/
├── Car_Insurance_Fraud_Detection.py  # Exploratory Data Analysis
├── preprocess.py                      # Data preprocessing logic
├── model.py                           # Model training script
├── save_model.py                      # Script to save trained models
├── main.py                            # FastAPI application
├── app.py                             # Streamlit web interface
├── preprocess.pkl                     # Saved preprocessor object
├── final_models.pkl                   # Saved trained models
└── README.md                          # Project documentation
