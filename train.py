import os
import argparse

import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

from data_utils import prepare_datasets, make_dataloaders
from model import LSTMFallDetector


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            running_loss += loss.item() * x.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            all_targets.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc, all_targets, all_preds


def main():
    parser = argparse.ArgumentParser(description="Train LSTM-based fall detection model")
    parser.add_argument("--data-path", type=str, required=True, help="Path to sensor CSV file")
    parser.add_argument("--seq-len", type=int, default=5, help="Sequence length (number of time steps)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--hidden-size", type=int, default=64, help="LSTM hidden size")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save model")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, feature_cols = prepare_datasets(
        args.data_path, seq_len=args.seq_len
    )

    input_size = X_train.shape[2]

    train_loader, val_loader, test_loader = make_dataloaders(
        (X_train, y_train), (X_val, y_val), (X_test, y_test), batch_size=args.batch_size
    )

    model = LSTMFallDetector(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=2,
        dropout=0.3,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    epochs_no_improve = 0
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(args.model_dir, timestamp)
    os.makedirs(experiment_dir, exist_ok=True)
    
    model_path = os.path.join(experiment_dir, "best_model.pt")
    scaler_path = os.path.join(experiment_dir, "scaler.joblib")
    meta_path = os.path.join(experiment_dir, "meta.joblib")
    plot_path = os.path.join(experiment_dir, "training_history.png")

    train_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, "
            f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
            joblib.dump(scaler, scaler_path)
            joblib.dump({"feature_cols": feature_cols, "seq_len": args.seq_len}, meta_path)
            print(f"Saved new best model with val acc = {best_val_acc:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    # Save training history plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Training history plot saved to {plot_path}")

    # Load best model for final evaluation
    model.load_state_dict(torch.load(model_path, map_location=device))

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    print("\nTest results:")
    print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["no_fall", "fall"]))

    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
