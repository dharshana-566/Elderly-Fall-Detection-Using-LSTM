# Elderly Fall Detection using LSTM

This is a simple mini-project demonstrating fall detection using an LSTM-based deep neural network on time-series sensor data (e.g., accelerometer and gyroscope).

## Project Structure

- `data/` – put your CSV files here
- `models/` – saved PyTorch models (`.pt`)
- `data_utils.py` – data loading and preprocessing helpers
- `model.py` – LSTM model definition
- `train.py` – training script
- `infer.py` – inference / demo script

## Expected Data Format

Assume a CSV with columns like:

- `acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, label`

Where:

- Each **row** is one time step
- `label` is 0 = no fall, 1 = fall
- Sequences are built by sliding windows of fixed length (e.g., 100 time steps)

You can adapt `data_utils.py` to your exact column names and format.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py --data-path data/your_data.csv --seq-len 100 --batch-size 64 --epochs 20
```

The trained model will be stored in `models/best_model.pt`.

## Inference

```bash
python infer.py --model-path models/best_model.pt --data-path data/your_test_data.csv --seq-len 100
```

This will print predicted probabilities and labels for each sequence.
