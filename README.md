# HAR70+ Activity Recognition

This project applies Machine Learning (ML) and Deep Learning (DL) techniques to the **HAR70+ dataset**, which contains dual accelerometer data from older adults performing daily activities. It showcases the development of an end-to-end human activity classification pipeline, including preprocessing, model training, evaluation, and reproducibility through GitHub Actions.

## Project Context

This repository is the **final senior project** of a 4th-year Digital Engineering student at SIIT, Thammasat University. It demonstrates applied engineering, real-world data handling, and ML workflow automation for deployment-ready environments. The project was designed to reflect strong ownership and technical depth in data processing and model development.

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

- **Programming**: Python 3.10, PySpark  
- **ML Libraries**: scikit-learn, XGBoost  
- **DL Libraries**: TensorFlow, Keras  
- **Distributed Processing**: Apache Spark (PySpark for scalable preprocessing pipelines)  
- **Automation**: GitHub Actions (CI/CD pipelines)  

## Project Structure

```
HAR70PLUS-ACTIVITY-RECOGNITION/
├── .github/workflows/
│   ├── preprocess.yml
│   ├── train_dl.yml
│   └── train_ml.yml
├── data/
│   └── raw/
│       └── har70plus/               # Raw dataset folder (might include original files)
├── models/
│   ├── cnn_model.json
│   ├── lstm_model.json
│   └── mrcnn_model.json
├── outputs/
│   ├── results_dl.json
│   └── results_ml.json
├── scripts/
│   ├── preprocess.py
│   ├── train_dl.py
│   └── train_ml.py
├── .gitignore
├── README.md
└── requirements.txt
````

## Key Features

- **Preprocessing**: Window segmentation (500 samples) and statistical feature extraction for ML  
- **ML Models**: KNN, SVM, Random Forest, XGBoost (feature-based)  
- **DL Models**: CNN, LSTM, MRCNN (sequence-based with class weighting)  
- **Model Persistence**: `.h5` for weights, `.json` for architecture  
- **Results Logging**: JSON logs for reproducibility  
- **CI/CD**: Automated workflows using GitHub Actions

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

3. Add the raw dataset to `data/raw/` (download from UCI ML Repository), then run preprocessing:

   ```bash
   python scripts/preprocess.py --window 500
   ```

4. Train models:

   ```bash
   python scripts/train_ml.py --input data/processed/har70plus_500s.pkl
   python scripts/train_dl.py --input data/processed/har70plus_500s.pkl
   ```

5. Monitor CI/CD:
   Review GitHub Actions workflows under the "Actions" tab to ensure reproducibility and workflow correctness.

## Citation

* A. Logacjov and A. Ustad. "HAR70+," *UCI Machine Learning Repository*, [https://doi.org/10.24432/C5CW3D](https://doi.org/10.24432/C5CW3D)
* Original implementation reference: [NTNU HARTH GitHub Repository](https://github.com/ntnu-ai-lab/harth-ml-experiments)

## Author

**Pavinee Pattanapornchai**
Student ID: 6422782266
Digital Engineering, Sirindhorn International Institute of Technology (SIIT), Thammasat University
GitHub: [@PavineePattanapornchai](https://github.com/PavineePattanapornchai)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
