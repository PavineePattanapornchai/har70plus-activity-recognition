# HAR70+ Activity Recognition

This project applies Machine Learning (ML) and Deep Learning (DL) techniques to the **HAR70+ dataset**, which contains dual accelerometer data from older adults performing daily activities. It showcases the development of an end-to-end human activity classification pipeline, including preprocessing, model training, evaluation, and reproducibility through GitHub Actions.

## Project Context

This repository is the **final senior project** of a 4th-year student in Digital Engineering at SIIT, Thammasat University. It demonstrates applied engineering, real-world data handling, and ML workflow automation for deployment-ready environments. The project was designed to reflect strong ownership and technical depth in data processing and model development.

## Dataset Overview

- **Dataset**: HAR70+ from the UCI Machine Learning Repository  
- **Subjects**: 18 older adults (aged 70–95)  
- **Sensors**: Two tri-axial accelerometers (lower back & thigh)  
- **Sampling Rate**: 50Hz  
- **Activities**:  
  - Walking  
  - Shuffling  
  - Stairs Ascending  
  - Stairs Descending  
  - Standing  
  - Sitting  
  - Lying  

## Technical Stack

- **Languages**: Python 3.10, PySpark
- **ML Libraries**: scikit-learn, XGBoost
- **DL Libraries**: TensorFlow, Keras
- **Distributed Processing**: Apache Spark (PySpark for scalable preprocessing pipelines)
- **Automation/DevOps**: GitHub Actions (CI/CD pipelines)

## Project Structure

```
├── data/
│   ├── raw/                   # Raw CSV sensor data (excluded from repo)
│   └── processed/             # Processed .pkl files (excluded from repo)
├── models/
│   ├── cnn_model.h5/.json     # CNN model weights & architecture
│   ├── lstm_model.h5/.json    # LSTM model weights & architecture
│   ├── mrcnn_model.h5/.json   # Multi-resolution CNN weights & architecture
├── scripts/
│   ├── preprocess.py          # Segments and processes raw data
│   ├── train_ml.py            # Trains ML models (KNN, SVM, RF, XGB)
│   ├── train_dl.py            # Trains DL models (CNN, LSTM, MRCNN)
├── .github/workflows/
│   ├── preprocess.yml         # GitHub Action for data preprocessing
│   ├── train_ml.yml           # GitHub Action for ML training
│   ├── train_dl.yml           # GitHub Action for DL training
├── results/
│   ├── results_ml.json        # Saved metrics from ML runs
│   └── results_dl.json        # Saved metrics from DL runs
└── README.md
```

## Key Features

- **Preprocessing**: Window segmentation (500 samples), statistical feature extraction for ML
- **ML Models**: KNN, SVM, Random Forest, XGBoost (feature-based)
- **DL Models**: CNN, LSTM, MRCNN (sequence-based with class weighting)
- **Model Persistence**: `.h5` for weights, `.json` for architecture
- **Results Logging**: JSON logs for reproducibility
- **CI/CD**: Automated testing of workflows using GitHub Actions

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/PavineePattanapornchai/har70plus-activity-recognition.git
   cd har70plus-activity-recognition
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate (Windows)
   pip install -r requirements.txt
   ```

3. Add the raw dataset to `data/raw/`, then run preprocessing:
   ```bash
   python scripts/preprocess.py --window 500
   ```

4. Train models:
   ```bash
   python scripts/train_ml.py --input data/processed/har70plus_500s.pkl
   python scripts/train_dl.py --input data/processed/har70plus_500s.pkl
   ```

## Citation

Dataset and concept from:

- A. Logacjov and A. Ustad. "HAR70+," *UCI Machine Learning Repository*, [https://doi.org/10.24432/C5CW3D](https://doi.org/10.24432/C5CW3D)
- Original implementation reference: [NTNU HARTH GitHub Repository](https://github.com/ntnu-ai-lab/harth-ml-experiments)

## Author

**Pavinee Pattanapornchai**  
Student ID: 6422782266  
Digital Engineering, Sirindhorn International Institute of Technology (SIIT), Thammasat University  
GitHub: [@PavineePattanapornchai](https://github.com/PavineePattanapornchai)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
