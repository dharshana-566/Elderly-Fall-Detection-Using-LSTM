import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


class SensorDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def load_sensor_csv(path, feature_cols=None, label_col="label"):
    """
    Load a sensor dataset from a CSV file.

    Parameters:
    - path: Path to the CSV file.
    - feature_cols: List of sensor columns to use. If None, uses all except label_col.
    - label_col: Name of the label column.

    Returns:
    - X: numpy array of features (float32).
    - y: numpy array of labels (int64).
    - feature_cols: The list of feature column names used.
    """
    df = pd.read_csv(path)

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != label_col]

    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.int64)

    return X, y, feature_cols


def create_sequences(X, y, seq_len):
    """
    Slice the full dataset into sliding window sequences.

    Parameters:
    - X: Scaled features array of shape (N, num_features).
    - y: Label array of shape (N,).
    - seq_len: Length of the sliding window.

    Returns:
    - sequences: numpy array of shape (num_sequences, seq_len, num_features).
    - labels: numpy array of shape (num_sequences,) containing the majority label in each window.
    """
    if len(X) < seq_len:
        raise ValueError(f"Dataset length {len(X)} is smaller than seq_len {seq_len}.")

    # Optimize sequence generation using numpy sliding windows instead of a python for-loop
    sequences = np.lib.stride_tricks.sliding_window_view(X, (seq_len, X.shape[1])).squeeze(1)
    
    # Same sliding window optimization for labels to calculate the majority vote quickly
    y_windows = np.lib.stride_tricks.sliding_window_view(y, (seq_len,))
    labels = (y_windows.mean(axis=1) >= 0.5).astype(np.int64)
    
    return sequences, labels


def load_sequences_for_inference(df_or_path, scaler, feature_cols, seq_len):
    """
    Load sequences specifically formatted for running inference via CLI or GUI.

    Parameters:
    - df_or_path: Path to CSV file or a loaded Pandas DataFrame.
    - scaler: Fitted standard scaler from training.
    - feature_cols: List of features expected by the model.
    - seq_len: Sequence length the model was trained on.

    Returns:
    - Numpy array of sequences ready for inference.
    """
    if isinstance(df_or_path, str):
        df = pd.read_csv(df_or_path)
    else:
        df = df_or_path

    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV")

    X = df[feature_cols].values.astype(np.float32)
    X_scaled = scaler.transform(X)

    if len(X_scaled) < seq_len:
        raise ValueError("Not enough data to form even one sequence. Increase data or reduce seq_len.")

    sequences = np.lib.stride_tricks.sliding_window_view(X_scaled, (seq_len, X.shape[1])).squeeze(1)
    return sequences


def prepare_datasets(csv_path, seq_len=100, test_size=0.2, val_size=0.1, random_state=42):
    """
    Full pipeline to load, scale, sequence, and split the data for training.
    """
    X, y, feature_cols = load_sensor_csv(csv_path)

    if len(X) < seq_len:
        raise ValueError(f"Dataset length {len(X)} is too small for seq_len {seq_len}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sequences, labels = create_sequences(X_scaled, y, seq_len)

    # For very small demo datasets, avoid stratified splitting to prevent
    # errors like "test_size = 1 should be >= number of classes".
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        sequences, labels, test_size=test_size, random_state=random_state
    )

    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, random_state=random_state
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, feature_cols


def make_dataloaders(train, val, test, batch_size=64):
    (X_train, y_train) = train
    (X_val, y_val) = val
    (X_test, y_test) = test

    train_ds = SensorDataset(X_train, y_train)
    val_ds = SensorDataset(X_val, y_val)
    test_ds = SensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
