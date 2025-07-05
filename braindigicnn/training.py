#!/usr/bin/env python3
"""
🚀 EEG Digit Classification Training
Memory-efficient training using preprocessed checkpoints
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
import time
import os
import sys

# Add parent directory to path to import from main.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import CheckpointManager

class EEGDataset(Dataset):
    """
    Memory-efficient EEG Dataset using indices
    """
    def __init__(self, X_data, y_data, indices):
        self.X_data = X_data
        self.y_data = y_data
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return torch.FloatTensor(self.X_data[actual_idx]), torch.LongTensor([self.y_data[actual_idx]])[0]

class BrainDigiCNN(nn.Module):
    """
    BrainDigiCNN model for EEG digit classification
    Based on paper specifications
    """
    def __init__(self, input_size, num_classes=10):
        super(BrainDigiCNN, self).__init__()
        
        # 1D CNN layers (paper specification)
        self.conv1 = nn.Conv1d(1, 256, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(256)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(256, 128, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 64, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)
        
        self.conv4 = nn.Conv1d(64, 32, kernel_size=7, padding=3)
        self.bn4 = nn.BatchNorm1d(32)
        self.pool4 = nn.MaxPool1d(2)
        
        # Calculate flattened size
        self.flattened_size = self._get_flattened_size(input_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def _get_flattened_size(self, input_size):
        # Calculate size after conv and pooling layers
        size = input_size
        for _ in range(4):  # 4 pooling layers
            size = size // 2
        return 32 * size  # 32 channels from last conv layer
    
    def forward(self, x):
        # Reshape for 1D conv: (batch, 1, features)
        x = x.unsqueeze(1)
        
        # Conv layers
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

def load_training_data():
    """
    Load preprocessed data and split indices
    """
    print("📁 Loading preprocessed training data...")
    
    # Use checkpoint manager from parent directory
    checkpoint_manager = CheckpointManager(checkpoint_dir="../checkpoints")
    
    # Load main data
    print("   Loading normalized features (200GB)...")
    X_processed = checkpoint_manager.load_checkpoint('hf_normalized_data')
    if X_processed is None:
        raise Exception("❌ Failed to load hf_normalized_data.pkl")
    
    # Load split indices
    print("   Loading split indices...")
    split_indices = checkpoint_manager.load_checkpoint('hf_split_indices')
    if split_indices is None:
        raise Exception("❌ Failed to load hf_split_indices.pkl")
    
    # Load labels (from raw extraction)
    print("   Loading labels...")
    raw_data = checkpoint_manager.load_checkpoint('hf_raw_extracted')
    if raw_data is None:
        raise Exception("❌ Failed to load hf_raw_extracted.pkl")
    
    X_raw, y = raw_data
    
    print(f"   ✅ Data loaded successfully:")
    print(f"      Features: {X_processed.shape}")
    print(f"      Labels: {len(y)}")
    print(f"      Train indices: {len(split_indices['train_idx'])}")
    print(f"      Val indices: {len(split_indices['val_idx'])}")
    print(f"      Test indices: {len(split_indices['test_idx'])}")
    
    return X_processed, y, split_indices

def create_data_loaders(X_processed, y, split_indices, batch_size=32):
    """
    Create memory-efficient data loaders
    """
    print(f"📦 Creating data loaders (batch_size={batch_size})...")
    
    # Create datasets
    train_dataset = EEGDataset(X_processed, y, split_indices['train_idx'])
    val_dataset = EEGDataset(X_processed, y, split_indices['val_idx'])
    test_dataset = EEGDataset(X_processed, y, split_indices['test_idx'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"   ✅ Data loaders created:")
    print(f"      Train batches: {len(train_loader)}")
    print(f"      Val batches: {len(val_loader)}")
    print(f"      Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

def train_model():
    """
    Main training function
    """
    print("🚀 EEG Digit Classification Training - BrainDigiCNN")
    print("=" * 60)
    
    # Load data
    X_processed, y, split_indices = load_training_data()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X_processed, y, split_indices, batch_size=32
    )
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Create model
    input_size = X_processed.shape[1]  # 516,096 features
    model = BrainDigiCNN(input_size=input_size, num_classes=10)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model created: {total_params:,} parameters")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n🎯 Training configuration:")
    print(f"   Optimizer: Adam (lr=0.001)")
    print(f"   Loss: CrossEntropyLoss")
    print(f"   Epochs: 20")
    print(f"   Batch size: 32")
    
    print(f"\n🚀 Starting training...")
    print(f"   Ready to train on {len(split_indices['train_idx']):,} samples")
    print(f"   Validation on {len(split_indices['val_idx']):,} samples")
    print(f"   Test on {len(split_indices['test_idx']):,} samples")
    
    # Training loop will be implemented here
    print(f"\n💡 Training implementation ready!")
    print(f"   All data loaded and prepared for training")
    print(f"   Model architecture: BrainDigiCNN")
    print(f"   Input features: {input_size:,}")
    
    return model, train_loader, val_loader, test_loader, device, criterion, optimizer

if __name__ == "__main__":
    # Run training setup
    model, train_loader, val_loader, test_loader, device, criterion, optimizer = train_model()
    
    print(f"\n✅ Training setup completed!")
    print(f"   Ready to implement training loop")
    print(f"   All checkpoints loaded successfully")
    print(f"   Memory-efficient data loading active")
