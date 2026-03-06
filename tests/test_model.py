import torch
import sys
import os

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import LSTMFallDetector

def test_lstm_forward_shape():
    """Test to ensure the LSTMFallDetector produces correctly shaped output."""
    batch_size = 16
    seq_len = 5
    input_size = 3
    num_classes = 2

    # Initialize model
    model = LSTMFallDetector(
        input_size=input_size, 
        hidden_size=64, 
        num_layers=2, 
        num_classes=num_classes
    )
    
    # Create dummy tensor matching (batch, seq_len, input_size)
    dummy_input = torch.randn(batch_size, seq_len, input_size)
    
    # Forward pass
    output = model(dummy_input)
    
    # Check shape
    assert output.shape == (batch_size, num_classes), f"Expected shape {(batch_size, num_classes)} but got {output.shape}"

if __name__ == "__main__":
    test_lstm_forward_shape()
    print("All tests passed.")
