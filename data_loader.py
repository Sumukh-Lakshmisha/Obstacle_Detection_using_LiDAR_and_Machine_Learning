import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split


class ThreatDataset(Dataset):
    """
    Custom dataset for threat/no_threat binary classification.
    """
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: List of paths to images
            labels: List of labels (0 for no_threat, 1 for threat)
            transform: Optional transform to be applied on images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = int(self.labels[idx])  # Ensure label is a Python int
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)  # Convert to torch.long


def get_data_transforms():
    """
    Get data transforms for training and validation/test.
    
    Returns:
        train_transform: Transformations for training data (with augmentation)
        val_transform: Transformations for validation/test data (no augmentation)
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def load_dataset(data_dir='Data', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Load dataset and split into train, validation, and test sets.
    
    Args:
        data_dir: Root directory containing 'threat' and 'no_threat' folders
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
        test_ratio: Proportion of data for testing (default: 0.15)
        random_state: Random seed for reproducibility
    
    Returns:
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    threat_dir = os.path.join(data_dir, 'threat')
    no_threat_dir = os.path.join(data_dir, 'no_threat')
    
    # Collect all image paths and labels
    image_paths = []
    labels = []
    
    # Load threat images (label = 1)
    if os.path.exists(threat_dir):
        threat_files = [f for f in os.listdir(threat_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for file in threat_files:
            image_paths.append(os.path.join(threat_dir, file))
            labels.append(1)
    
    # Load no_threat images (label = 0)
    if os.path.exists(no_threat_dir):
        no_threat_files = [f for f in os.listdir(no_threat_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for file in no_threat_files:
            image_paths.append(os.path.join(no_threat_dir, file))
            labels.append(0)
    
    # Convert to numpy arrays for easier splitting
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    
    # First split: separate train from (val + test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, 
        test_size=(val_ratio + test_ratio), 
        random_state=random_state,
        stratify=labels
    )
    
    # Second split: separate val from test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=(1 - val_size),
        random_state=random_state,
        stratify=temp_labels
    )
    
    print(f"Dataset loaded successfully!")
    print(f"Total images: {len(image_paths)}")
    print(f"  - Threat: {np.sum(labels == 1)}")
    print(f"  - No Threat: {np.sum(labels == 0)}")
    print(f"\nSplit:")
    print(f"  - Training: {len(train_paths)} ({len(train_paths)/len(image_paths)*100:.1f}%)")
    print(f"    * Threat: {np.sum(train_labels == 1)}, No Threat: {np.sum(train_labels == 0)}")
    print(f"  - Validation: {len(val_paths)} ({len(val_paths)/len(image_paths)*100:.1f}%)")
    print(f"    * Threat: {np.sum(val_labels == 1)}, No Threat: {np.sum(val_labels == 0)}")
    print(f"  - Test: {len(test_paths)} ({len(test_paths)/len(image_paths)*100:.1f}%)")
    print(f"    * Threat: {np.sum(test_labels == 1)}, No Threat: {np.sum(test_labels == 0)}")
    
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def get_data_loaders(data_dir='Data', batch_size=32, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, num_workers=4):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data_dir: Root directory containing 'threat' and 'no_threat' folders
        batch_size: Batch size for data loaders
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        num_workers: Number of worker processes for data loading
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Load and split dataset
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = load_dataset(
        data_dir, train_ratio, val_ratio, test_ratio
    )
    
    # Get transforms
    train_transform, val_transform = get_data_transforms()
    
    # Create datasets
    train_dataset = ThreatDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = ThreatDataset(val_paths, val_labels, transform=val_transform)
    test_dataset = ThreatDataset(test_paths, test_labels, transform=val_transform)
    
    # On Windows, num_workers > 0 can cause issues, so set to 0 if on Windows
    import platform
    if platform.system() == 'Windows':
        num_workers = 0
        pin_memory = False
    else:
        pin_memory = True
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader

