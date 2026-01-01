import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import os
import json

from model import get_model


def load_model(model_path, device='cuda'):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint file
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    model = get_model(num_classes=2, input_channels=3, device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def preprocess_image(image_path, device='cuda'):
    """
    Preprocess an image for inference.
    
    Args:
        image_path: Path to the image file
        device: Device to load image on
    
    Returns:
        Preprocessed image tensor
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)
    
    return image_tensor


def predict_image(model, image_path, device='cuda', return_probabilities=False):
    """
    Predict threat/no_threat for a single image.
    
    Args:
        model: Trained model
        image_path: Path to the image file
        device: Device to run inference on
        return_probabilities: If True, return class probabilities
    
    Returns:
        Prediction (string: 'threat' or 'no_threat') and optionally probabilities
    """
    # Preprocess image
    image_tensor = preprocess_image(image_path, device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    # Convert to class label
    class_names = ['no_threat', 'threat']
    prediction = class_names[predicted.item()]
    confidence = probabilities[0][predicted.item()].item()
    
    if return_probabilities:
        prob_dict = {
            'no_threat': probabilities[0][0].item(),
            'threat': probabilities[0][1].item()
        }
        return prediction, confidence, prob_dict
    else:
        return prediction, confidence


def predict_batch(model, image_paths, device='cuda'):
    """
    Predict threat/no_threat for multiple images.
    
    Args:
        model: Trained model
        image_paths: List of paths to image files
        device: Device to run inference on
    
    Returns:
        List of predictions with confidence scores
    """
    results = []
    
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        prediction, confidence, probabilities = predict_image(
            model, image_path, device, return_probabilities=True
        )
        
        results.append({
            'image_path': image_path,
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probabilities
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Inference script for Threat Detection Model')
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--image-path', type=str, default=None,
                       help='Path to a single image for prediction')
    parser.add_argument('--image-dir', type=str, default=None,
                       help='Path to directory containing images for batch prediction')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save prediction results (JSON format)')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    model = load_model(args.model_path, device)
    print("Model loaded successfully!")
    
    # Get image paths
    image_paths = []
    
    if args.image_path:
        if os.path.isfile(args.image_path):
            image_paths = [args.image_path]
        else:
            print(f"Error: Image file not found: {args.image_path}")
            return
    elif args.image_dir:
        if os.path.isdir(args.image_dir):
            image_paths = [
                os.path.join(args.image_dir, f) 
                for f in os.listdir(args.image_dir) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
        else:
            print(f"Error: Directory not found: {args.image_dir}")
            return
    else:
        print("Error: Please provide either --image-path or --image-dir")
        return
    
    if not image_paths:
        print("Error: No images found")
        return
    
    print(f"\nProcessing {len(image_paths)} image(s)...")
    print("-" * 80)
    
    # Make predictions
    results = predict_batch(model, image_paths, device)
    
    # Display results
    for result in results:
        print(f"\nImage: {result['image_path']}")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Probabilities:")
        print(f"    - No Threat: {result['probabilities']['no_threat']:.4f}")
        print(f"    - Threat: {result['probabilities']['threat']:.4f}")
    
    # Save results if output path is provided
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()

