"""
Example usage script for the Threat Detection CNN Model.

This script demonstrates how to:
1. Train the model
2. Run inference on a single image
3. Run batch inference on multiple images
"""

import os
import subprocess
import sys


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def example_train():
    """Example: Train the model."""
    print_section("Example 1: Training the Model")
    
    print("To train the model, run:")
    print("\n  python train.py")
    print("\nOr with custom parameters:")
    print("\n  python train.py --batch-size 64 --epochs 100 --learning-rate 0.0001")
    print("\nAvailable training arguments:")
    print("  --data-dir: Path to data directory (default: 'Data')")
    print("  --batch-size: Batch size (default: 32)")
    print("  --epochs: Number of epochs (default: 50)")
    print("  --learning-rate: Learning rate (default: 0.001)")
    print("  --train-ratio: Training set ratio (default: 0.7)")
    print("  --val-ratio: Validation set ratio (default: 0.15)")
    print("  --test-ratio: Test set ratio (default: 0.15)")
    print("  --output-dir: Output directory for models (default: 'checkpoints')")
    print("  --cpu: Force CPU usage")


def example_inference_single():
    """Example: Single image inference."""
    print_section("Example 2: Single Image Inference")
    
    print("To predict on a single image, run:")
    print("\n  python inference.py --model-path checkpoints/best_model.pth --image-path path/to/image.png")
    print("\nThis will output:")
    print("  - Prediction: 'threat' or 'no_threat'")
    print("  - Confidence score")
    print("  - Class probabilities")


def example_inference_batch():
    """Example: Batch inference."""
    print_section("Example 3: Batch Inference")
    
    print("To predict on multiple images in a directory, run:")
    print("\n  python inference.py --model-path checkpoints/best_model.pth --image-dir path/to/images/")
    print("\nTo save results to a JSON file:")
    print("\n  python inference.py --model-path checkpoints/best_model.pth --image-dir path/to/images/ --output results.json")


def example_python_api():
    """Example: Using the model in Python code."""
    print_section("Example 4: Using the Model in Python Code")
    
    code_example = '''
# Load and use the model in your Python code
import torch
from model import get_model
from inference import load_model, predict_image

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load trained model
model = load_model('checkpoints/best_model.pth', device=device)

# Predict on a single image
prediction, confidence = predict_image(model, 'path/to/image.png', device=device)
print(f"Prediction: {prediction}, Confidence: {confidence:.4f}")

# Get probabilities
prediction, confidence, probabilities = predict_image(
    model, 'path/to/image.png', device=device, return_probabilities=True
)
print(f"Probabilities: {probabilities}")
'''
    print(code_example)


def main():
    """Main function to display all examples."""
    print("\n" + "=" * 80)
    print("  THREAT DETECTION CNN MODEL - USAGE EXAMPLES")
    print("=" * 80)
    
    example_train()
    example_inference_single()
    example_inference_batch()
    example_python_api()
    
    print_section("Quick Start")
    print("1. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("\n2. Train the model:")
    print("   python train.py")
    print("\n3. Run inference:")
    print("   python inference.py --model-path checkpoints/best_model.pth --image-path Data/threat/ss1.png")
    print("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    main()

