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
    df = pd.read_csv(path)

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != label_col]

    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.int64)

    return X, y, feature_cols


def create_sequences(X, y, seq_len):
    sequences = []
    labels = []
    for i in range(len(X) - seq_len + 1):
        seq_x = X[i : i + seq_len]
        # majority label in the window (for robustness)
        window_labels = y[i : i + seq_len]
        label = 1 if window_labels.mean() >= 0.5 else 0
        sequences.append(seq_x)
        labels.append(label)
    return np.array(sequences), np.array(labels)


def prepare_datasets(csv_path, seq_len=100, test_size=0.2, val_size=0.1, random_state=42):
    X, y, feature_cols = load_sensor_csv(csv_path)

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
