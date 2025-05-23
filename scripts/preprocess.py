import os
import glob
import pandas as pd
from collections import Counter
import argparse

def segment_signal(df, window_size):
    segments = []
    labels = []
    subject_ids = []

    for subject_id, subject_df in df.groupby("subject_id"):
        for start in range(0, len(subject_df), window_size):
            end = start + window_size
            segment = subject_df.iloc[start:end]
            if len(segment) == window_size:
                segments.append(segment[["ax", "ay", "az", "gx", "gy", "gz"]].values)
                labels.append(Counter(segment["label"]).most_common(1)[0][0])
                subject_ids.append(subject_id)

    return segments, labels, subject_ids

def preprocess_and_save(window_size, input_dir, output_path):
    print(f"Reading data from {input_dir}...")
    all_files = glob.glob(os.path.join(input_dir, "*.csv"))

    if not all_files:
        print("⚠️ No CSV files found. Please add HAR70+ CSVs to 'data/raw/har70plus/'.")
        return

    data = []
    for file in all_files:
        df = pd.read_csv(file)

        # ✅ Rename columns to match Colab structure
        df = df.rename(columns={
            "back_x": "ax", "back_y": "ay", "back_z": "az",
            "thigh_x": "gx", "thigh_y": "gy", "thigh_z": "gz"
        })

        df["subject_id"] = os.path.basename(file).split(".")[0]
        data.append(df)

    df_all = pd.concat(data, ignore_index=True)

    print(f"Segmenting with window size {window_size} samples...")
    segments, labels, subject_ids = segment_signal(df_all, window_size)

    print(f"Saving segmented data to {output_path}...")
    output_df = pd.DataFrame({
        "subject_id": subject_ids,
        "label": labels,
        "data": [seg.tolist() for seg in segments]
    })

    output_df.to_pickle(output_path)
    print("✅ Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess HAR70+ dataset")
    parser.add_argument("--window", type=int, required=True, help="Window size in samples (e.g., 500 or 250)")
    args = parser.parse_args()

    input_dir = "data/raw/har70plus"
    output_path = f"data/processed/har70plus_{args.window}s.pkl"

    preprocess_and_save(args.window, input_dir, output_path)
