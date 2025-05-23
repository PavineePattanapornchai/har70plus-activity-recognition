#  Trigger GitHub Actions workflow for ML
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import argparse
import warnings

warnings.filterwarnings("ignore")

# 🔍 Extract statistical features from ax, ay, az
def extract_features(data):
    features = []
    for window in data:
        window = np.array(window)[:, 0:3]  # first 3 columns = ax, ay, az
        stats = []
        for axis in range(3):
            x = window[:, axis]
            stats.extend([
                np.mean(x),
                np.var(x),
                pd.Series(x).skew(),
                pd.Series(x).kurt()
            ])
        features.append(stats)
    return np.array(features)

# 🚀 Train models and print accuracy + macro F1
def train_models(X_train, X_test, y_train, y_test):
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(),
        "RandomForest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        results[name] = {"Accuracy": round(acc, 4), "F1": round(f1, 4)}
        print(f"{name}: Accuracy = {acc:.4f}, Macro F1 = {f1:.4f}")

    # ✅ Save results to JSON
    with open("results_ml.json", "w") as f:
        json.dump(results, f, indent=2)
    print("✅ Saved results to results_ml.json")

# 🧠 Main script entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML models on HAR70+")
    parser.add_argument("--input", type=str, default="data/processed/har70plus_500s.pkl", help="Path to .pkl file")
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    df = pd.read_pickle(args.input)

    X = extract_features(df["data"])

    # ✅ Label encoding to map activity labels to 0–6 (e.g., 'walking' → 0)
    le = LabelEncoder()
    y = le.fit_transform(df["label"])

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    print("Training models...")
    train_models(X_train, X_test, y_train, y_test)