import argparse

import numpy as np
import pandas as pd
import torch
import joblib

from model import LSTMFallDetector


def load_sequences_for_inference(csv_path, scaler, feature_cols, seq_len):
    df = pd.read_csv(csv_path)
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV")

    X = df[feature_cols].values.astype(np.float32)
    X_scaled = scaler.transform(X)

    sequences = []
    for i in range(len(X_scaled) - seq_len + 1):
        seq_x = X_scaled[i : i + seq_len]
        sequences.append(seq_x)

    if not sequences:
        raise ValueError("Not enough data to form even one sequence. Increase data or reduce seq_len.")

    sequences = np.array(sequences)
    return sequences


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained LSTM fall detection model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved model .pt file")
    parser.add_argument("--data-path", type=str, required=True, help="Path to CSV file for inference")
    parser.add_argument("--scaler-path", type=str, default=None, help="Path to saved scaler.joblib (optional)")
    parser.add_argument("--meta-path", type=str, default=None, help="Path to saved meta.joblib (optional)")

    args = parser.parse_args()

    if args.scaler_path is None:
        # assume same folder as model
        base_dir = args.model_path.rsplit("/", 1)[0] if "/" in args.model_path else "."
        args.scaler_path = base_dir + "/scaler.joblib"
    if args.meta_path is None:
        base_dir = args.model_path.rsplit("/", 1)[0] if "/" in args.model_path else "."
        args.meta_path = base_dir + "/meta.joblib"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = joblib.load(args.scaler_path)
    meta = joblib.load(args.meta_path)
    feature_cols = meta["feature_cols"]
    seq_len = meta["seq_len"]

    sequences = load_sequences_for_inference(args.data_path, scaler, feature_cols, seq_len)
    input_size = sequences.shape[2]

    model = LSTMFallDetector(input_size=input_size)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        x = torch.tensor(sequences, dtype=torch.float32).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1]  # probability of 'fall' class
        preds = (probs >= 0.5).long().cpu().numpy()

    for i, (p, prob) in enumerate(zip(preds, probs.cpu().numpy())):
        label = "FALL" if p == 1 else "NO_FALL"
        print(f"Sequence {i}: {label} (fall probability = {prob:.3f})")


if __name__ == "__main__":
    main()
