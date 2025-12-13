import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
import torch
import joblib

from model import LSTMFallDetector


def load_sequences_for_inference(df, scaler, feature_cols, seq_len):
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

    return np.array(sequences)


class FallDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Elderly Fall Detection (LSTM)")
        self.root.geometry("700x500")

        self.model = None
        self.scaler = None
        self.meta = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._build_widgets()

    def _build_widgets(self):
        frame_top = tk.Frame(self.root)
        frame_top.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(frame_top, text="Model directory:").pack(side=tk.LEFT)
        self.model_dir_var = tk.StringVar(value="models")
        self.model_dir_entry = tk.Entry(frame_top, textvariable=self.model_dir_var, width=40)
        self.model_dir_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(frame_top, text="Browse", command=self.browse_model_dir).pack(side=tk.LEFT)
        tk.Button(frame_top, text="Load Model", command=self.load_model).pack(side=tk.LEFT, padx=5)

        frame_mid = tk.Frame(self.root)
        frame_mid.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(frame_mid, text="CSV file:").pack(side=tk.LEFT)
        self.csv_path_var = tk.StringVar(value="data/sample_falls.csv")
        self.csv_entry = tk.Entry(frame_mid, textvariable=self.csv_path_var, width=40)
        self.csv_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(frame_mid, text="Browse", command=self.browse_csv).pack(side=tk.LEFT)
        tk.Button(frame_mid, text="Run Detection", command=self.run_detection).pack(side=tk.LEFT, padx=5)

        frame_stats = tk.Frame(self.root)
        frame_stats.pack(fill=tk.X, padx=10, pady=5)

        self.status_var = tk.StringVar(value="Load model and select CSV to start.")
        tk.Label(frame_stats, textvariable=self.status_var, fg="blue").pack(side=tk.LEFT)

        frame_table = tk.Frame(self.root)
        frame_table.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        columns = ("index", "label", "prob")
        self.tree = ttk.Treeview(frame_table, columns=columns, show="headings")
        self.tree.heading("index", text="Seq #")
        self.tree.heading("label", text="Prediction")
        self.tree.heading("prob", text="Fall Probability")
        self.tree.column("index", width=60, anchor=tk.CENTER)
        self.tree.column("label", width=120, anchor=tk.CENTER)
        self.tree.column("prob", width=120, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.tree.tag_configure("fall", background="#ffcccc")
        self.tree.tag_configure("no_fall", background="#e6ffe6")

        scrollbar = ttk.Scrollbar(frame_table, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def browse_model_dir(self):
        dirname = filedialog.askdirectory(initialdir=".")
        if dirname:
            self.model_dir_var.set(dirname)

    def browse_csv(self):
        filename = filedialog.askopenfilename(
            initialdir=".",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        )
        if filename:
            self.csv_path_var.set(filename)

    def load_model(self):
        model_dir = self.model_dir_var.get()
        model_path = os.path.join(model_dir, "best_model.pt")
        scaler_path = os.path.join(model_dir, "scaler.joblib")
        meta_path = os.path.join(model_dir, "meta.joblib")

        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Model file not found: {model_path}")
            return
        if not os.path.exists(scaler_path) or not os.path.exists(meta_path):
            messagebox.showerror("Error", "Scaler or meta file not found in model directory.")
            return

        try:
            self.scaler = joblib.load(scaler_path)
            self.meta = joblib.load(meta_path)
            feature_cols = self.meta["feature_cols"]
            seq_len = self.meta["seq_len"]

            # Dummy input size derived from feature_cols length
            input_size = len(feature_cols)
            self.model = LSTMFallDetector(input_size=input_size)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()

            self.status_var.set("Model loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")

    def run_detection(self):
        if self.model is None or self.scaler is None or self.meta is None:
            messagebox.showwarning("Warning", "Please load the model first.")
            return

        csv_path = self.csv_path_var.get()
        if not os.path.exists(csv_path):
            messagebox.showerror("Error", f"CSV file not found: {csv_path}")
            return

        try:
            df = pd.read_csv(csv_path)
            feature_cols = self.meta["feature_cols"]
            seq_len = self.meta["seq_len"]

            sequences = load_sequences_for_inference(df, self.scaler, feature_cols, seq_len)

            with torch.no_grad():
                x = torch.tensor(sequences, dtype=torch.float32).to(self.device)
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = (probs >= 0.5).long().cpu().numpy()

            # Clear table
            for row in self.tree.get_children():
                self.tree.delete(row)

            fall_count = 0
            for i, (p, prob) in enumerate(zip(preds, probs.cpu().numpy())):
                label_str = "FALL" if p == 1 else "NO_FALL"
                if p == 1:
                    fall_count += 1
                tag = "fall" if p == 1 else "no_fall"
                self.tree.insert("", tk.END, values=(i, label_str, f"{prob:.3f}"), tags=(tag,))

            self.status_var.set(f"Detection complete. Sequences: {len(preds)}, FALLs: {fall_count}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to run detection: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = FallDetectionGUI(root)
    root.mainloop()