# Human Activity Recognition with HAR70+ Dataset

This repository contains the complete source code and training pipelines for a Human Activity Recognition (HAR) system using the HAR70+ dataset. The system implements both traditional machine learning and deep learning models, structured in a modular and MLOps-aligned architecture using Python and GitHub Actions.

## Project Overview

This project was completed as a **senior/final-year project** by a **4th-year Digital Engineering student** at the **Sirindhorn International Institute of Technology (SIIT), Thammasat University**.

The objective was to develop a robust and reproducible pipeline for recognizing human activities using wearable sensor data collected from older adults. The project reflects a capstone-level challenge integrating data preprocessing, model development, evaluation, and automation.

## Features

- Statistical feature extraction for machine learning models
- CNN, LSTM, and Multi-Resolution CNN (MRCNN) deep learning architectures
- Class imbalance handling with weighted loss functions
- Model evaluation using Accuracy and Macro F1-score
- Model serialization (.h5 weights and .json architectures)
- GitHub Actions workflows for CI/CD automation
- Structured directory for reproducible experimentation

## Technologies Used

- Python 3.10
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- GitHub Actions (CI/CD)
- Pandas, NumPy

## Folder Structure

```
har70plus-activity-recognition/
├── data/
│   ├── raw/                      # Raw HAR70+ CSV files
│   └── processed/                # Preprocessed .pkl files (excluded from repo)
├── models/                       # Exported model weights (.h5) and architecture (.json)
│   ├── cnn_model.h5/.json
│   ├── lstm_model.h5/.json
│   └── mrcnn_model.h5/.json
├── scripts/
│   ├── preprocess.py             # Segment and save data
│   ├── train_ml.py               # Train ML models
│   └── train_dl.py               # Train DL models
├── .github/
│   └── workflows/
│       ├── train_ml.yml          # ML GitHub Actions workflow
│       └── train_dl.yml          # DL GitHub Actions workflow
├── results_ml.json               # Accuracy and F1-score for ML models
├── results_dl.json               # Accuracy and F1-score for DL models
├── requirements.txt
├── .gitignore
└── README.md
```

## Results Summary

| Model        | Accuracy | Macro F1 |
|--------------|----------|-----------|
| KNN          | 0.69     | 0.33      |
| SVM          | 0.59     | 0.22      |
| RandomForest | 0.92     | 0.52      |
| XGBoost      | 0.92     | 0.53      |
| CNN          | 0.93     | 0.56      |
| LSTM         | 0.41     | 0.32      |
| MRCNN        | 0.90     | 0.54      |

## How to Run

```bash
# Clone the repository
git clone https://github.com/PavineePattanapornchai/har70plus-activity-recognition.git
cd har70plus-activity-recognition

# Install dependencies
pip install -r requirements.txt

# Preprocess the data
python scripts/preprocess.py --window 500

# Train ML models
python scripts/train_ml.py --input data/processed/har70plus_500s.pkl

# Train DL models
python scripts/train_dl.py --input data/processed/har70plus_500s.pkl
```

## GitHub Actions Automation

This repo includes two GitHub Actions workflows for verifying pipelines:

- `.github/workflows/train_ml.yml`: ML training environment setup
- `.github/workflows/train_dl.yml`: DL training setup and simulation

These workflows validate that training environments are functional and follow automation best practices.

## Citations & Acknowledgements

**Dataset**:  
A. Logacjov and A. Ustad. *HAR70+*, UCI Machine Learning Repository.  
Available: https://doi.org/10.24432/C5CW3D

**Reference Implementation**:  
NTNU AI Lab — HARTH Experiments:  
https://github.com/ntnu-ai-lab/harth-ml-experiments

## License

This project is licensed under the MIT License.  
For academic and educational use only.
