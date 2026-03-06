# 🧓 Elderly Fall Detection Using LSTM

## Description
This project implements a **Long Short-Term Memory (LSTM)** based deep learning system designed to detect falls in elderly people using time-series sensor data. By analyzing sequential data, the Bidirectional LSTM model can effectively distinguish between normal daily activities and potential fall events. The system supports model training, command-line inference, and includes an interactive Tkinter-based GUI for ease of use.

## Tech Stack
- **Python 3.x**
- **PyTorch** (Deep Learning Framework)
- **Scikit-learn** (Data Preprocessing & Metrics)
- **Pandas & NumPy** (Data Manipulation)
- **Tkinter** (Desktop GUI)
- **Joblib** (Model Serialization)

## Features
- **Bidirectional LSTM Architecture**: Effectively captures temporal dependencies in sensor data.
- **End-to-End Pipeline**: Scripts for training, validation, and testing.
- **Interactive GUI application**: User-friendly Tkinter interface to select models and run inference on CSV files.
- **Hardware Acceleration**: Automatically utilizes GPU (CUDA) if available, otherwise falls back to CPU.
- **Detailed Output**: Provides sequence-level predictions and fall probabilities.

## Installation Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Elderly-Fall-Detection-Using-LSTM.git
   cd Elderly-Fall-Detection-Using-LSTM
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Training the Model
To train the model on your sensor data (CSV format):
```bash
python train.py --data-path data/your_dataset.csv --epochs 20 --batch-size 64 --seq-len 5
```
This will save `best_model.pt`, `scaler.joblib`, and `meta.joblib` in the `models/` directory.

### 2. Command-Line Inference
To run inference on a new dataset via CLI:
```bash
python infer.py --model-path models/best_model.pt --data-path data/test_data.csv
```

### 3. GUI Application
To launch the interactive Desktop GUI:
```bash
python gui_app.py
```
* Use the **Browse** buttons to select your built `models` directory and your inference CSV file.
* Click **Run Detection** to view the predictions and fall probabilities in the table.

## Project Structure
```
Elderly-Fall-Detection-Using-LSTM/
│
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── train.py                # Script to train the LSTM model
├── infer.py                # CLI script for inference
├── gui_app.py              # Tkinter graphical user interface
├── model.py                # PyTorch Bidirectional LSTM model definition
├── data_utils.py           # Data loading and preprocessing utilities
├── data/                   # Directory containing datasets (CSV files)
└── models/                 # Directory where the trained models are saved
```

## Example Output

### GUI Output
When running the GUI, the table displays predictions similarly to:
| Seq # | Prediction | Fall Probability |
|-------|------------|------------------|
| 0     | NO_FALL    | 0.045            |
| 1     | FALL       | 0.982            |
| 2     | FALL       | 0.910            |

### CLI Output
```
Sequence 0: NO_FALL (fall probability = 0.045)
Sequence 1: FALL (fall probability = 0.982)
Sequence 2: FALL (fall probability = 0.910)
```

## Future Improvements
- Expand the dataset to include more diverse daily activities and body types.
- Add support for real-time sensor streaming directly to the app.
- Provide a responsive Web-based Dashboard (e.g., using Streamlit or Flask).
- Implement an alert mechanism (e.g., SMS or Email notification) when a fall is detected.

## Contributing
Contributions are welcome!
1. Fork the project.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## License
Distributed under the MIT License. See `LICENSE` for more information.
