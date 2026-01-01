"""
Quick test script to verify the model architecture works correctly.
"""

import torch
from model import get_model

def test_model():
    """Test that the model can be instantiated and run a forward pass."""
    print("Testing model architecture...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = get_model(num_classes=2, input_channels=3, device=device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, 2)")
    
    # Check output
    assert output.shape == (batch_size, 2), f"Expected output shape ({batch_size}, 2), got {output.shape}"
    print("\n✓ Model test passed!")
    
    # Test with softmax
    probabilities = torch.softmax(output, dim=1)
    print(f"\nSample output probabilities:")
    print(probabilities[0])
    print(f"Sum of probabilities: {probabilities[0].sum().item():.4f} (should be ~1.0)")
    
    return model

if __name__ == '__main__':
    test_model()

