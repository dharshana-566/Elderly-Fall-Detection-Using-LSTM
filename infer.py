import argparse

import numpy as np
import pandas as pd
import torch
import joblib

from model import LSTMFallDetector


from data_utils import load_sequences_for_inference

def main():
    parser = argparse.ArgumentParser(description="Run inference with trained LSTM fall detection model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved model .pt file")
    parser.add_argument("--data-path", type=str, required=True, help="Path to CSV file for inference")
    parser.add_argument("--scaler-path", type=str, default=None, help="Path to saved scaler.joblib (optional)")
    parser.add_argument("--meta-path", type=str, default=None, help="Path to saved meta.joblib (optional)")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional path to save predictions to a CSV file.")

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

    results = []
    for i, (p, prob) in enumerate(zip(preds, probs.cpu().numpy())):
        label = "FALL" if p == 1 else "NO_FALL"
        print(f"Sequence {i}: {label} (fall probability = {prob:.3f})")
        results.append({
            "sequence_idx": i,
            "prediction": label,
            "fall_probability": prob
        })

    if args.output_csv:
        out_df = pd.DataFrame(results)
        out_df.to_csv(args.output_csv, index=False)
        print(f"\nSaved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
