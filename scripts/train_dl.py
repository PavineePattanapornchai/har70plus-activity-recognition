import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Conv1D, MaxPooling1D, Flatten, LSTM,
    Dropout, Input, concatenate
)
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# CNN model
def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

# LSTM model
def build_lstm(input_shape, num_classes):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dropout(0.3),
        Dense(100, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

# MRCNN model (multi-resolution CNN)
def build_mrcnn(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    conv_3 = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    conv_5 = Conv1D(64, kernel_size=5, activation='relu', padding='same')(inputs)
    conv_7 = Conv1D(64, kernel_size=7, activation='relu', padding='same')(inputs)

    merged = concatenate([conv_3, conv_5, conv_7])
    pooled = MaxPooling1D(pool_size=2)(merged)
    dropped = Dropout(0.3)(pooled)
    flat = Flatten()(dropped)
    dense = Dense(100, activation='relu')(flat)
    outputs = Dense(num_classes, activation='softmax')(dense)

    return Model(inputs=inputs, outputs=outputs)

# Unified training function
def train_model(name, model, X_train, X_test, y_train, y_test, class_weights):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, class_weight=class_weights)
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    acc = accuracy_score(y_true, y_pred_classes)
    f1 = f1_score(y_true, y_pred_classes, average="macro")
    print(f"{name}: Accuracy = {acc:.4f}, Macro F1 = {f1:.4f}")

# Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed/har70plus_500s.pkl", help="Path to preprocessed .pkl file")
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    df = pd.read_pickle(args.input)
    X = np.array(df["data"].tolist())  # Shape: (samples, 500, 6)

    # Label encoding
    le = LabelEncoder()
    y_int = le.fit_transform(df["label"])
    y = to_categorical(y_int)

    # Class weights
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_int), y=y_int)
    class_weights = dict(enumerate(class_weights))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    input_shape = X_train.shape[1:]
    num_classes = y.shape[1]

    print("Training CNN...")
    cnn = build_cnn(input_shape, num_classes)
    train_model("CNN", cnn, X_train, X_test, y_train, y_test, class_weights)

    print("Training LSTM...")
    lstm = build_lstm(input_shape, num_classes)
    train_model("LSTM", lstm, X_train, X_test, y_train, y_test, class_weights)

    print("Training MRCNN...")
    mrcnn = build_mrcnn(input_shape, num_classes)
    train_model("MRCNN", mrcnn, X_train, X_test, y_train, y_test, class_weights)
