#!/usr/bin/env python3
"""
Resume directly from zero-copy checkpoint
Skip feature extraction and go straight to training
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import time
import os

# Import existing classes
from main import CheckpointManager
from memory_efficient_saver import MemoryEfficientSaver

class HuggingFaceEEGDataset(Dataset):
    """PyTorch Dataset for EEG data"""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class BrainDigiCNN(nn.Module):
    """BrainDigiCNN model for EEG classification"""
    
    def __init__(self, input_size, num_classes=10):
        super(BrainDigiCNN, self).__init__()
        
        # 1D CNN layers as specified in paper
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
        
        # Dense layers
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
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
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return F.softmax(x, dim=1)

def load_labels_from_dataset():
    """Load labels from original dataset"""
    print("📥 Loading labels from Hugging Face dataset...")
    
    try:
        from datasets import load_dataset
        ds = load_dataset("DavidVivancos/MindBigData2022_MNIST_EP")
        data_split = ds['train']
        
        labels = []
        for i, sample in enumerate(data_split):
            labels.append(sample['label'])
            if i % 10000 == 0:
                print(f"   Loading labels: {i:,}/{len(data_split):,}")
        
        labels = np.array(labels, dtype=np.int64)
        print(f"   ✅ Labels loaded: {labels.shape}")
        return labels
        
    except Exception as e:
        print(f"   ❌ Failed to load labels: {e}")
        return None

def main_resume_from_checkpoint():
    """Resume training directly from checkpoint"""

    # Enable CUDA debugging
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    print("🚀 Resume EEG Training from Zero-Copy Checkpoint")
    print("=" * 60)
    print("🔧 CUDA debugging enabled")
    
    # 1. Load preprocessed data from zero-copy checkpoint
    print("📁 Loading preprocessed data from zero-copy checkpoint...")
    
    checkpoint_manager = CheckpointManager()
    saver = MemoryEfficientSaver(checkpoint_manager.checkpoint_dir)
    
    if not saver.checkpoint_exists('hf_normalized_data'):
        print("❌ Zero-copy checkpoint not found!")
        print("   Available files:")
        for file in os.listdir(checkpoint_manager.checkpoint_dir):
            print(f"   - {file}")
        return
    
    # Load data
    X_processed = saver.load_zero_copy('hf_normalized_data')
    if X_processed is None:
        print("❌ Failed to load zero-copy checkpoint!")
        return
    
    print(f"✅ Loaded preprocessed data: {X_processed.shape}")
    print(f"💾 Memory usage: ~{X_processed.nbytes / (1024**3):.1f} GB")
    
    # 2. Load labels
    y = load_labels_from_dataset()
    if y is None:
        print("❌ Failed to load labels!")
        return
    
    print(f"✅ Labels loaded: {y.shape}")

    # Debug label range
    print(f"🔍 Label analysis:")
    print(f"   Min label: {y.min()}")
    print(f"   Max label: {y.max()}")
    print(f"   Unique labels: {np.unique(y)}")

    # Filter out rest samples (label = -1)
    if y.min() < 0:
        print(f"⚠️  Found rest samples (label = -1), filtering them out...")

        # Find valid samples (labels 0-9)
        valid_mask = y >= 0
        valid_indices = np.where(valid_mask)[0]

        print(f"   Total samples: {len(y):,}")
        print(f"   Rest samples (label=-1): {np.sum(y == -1):,}")
        print(f"   Valid samples (labels 0-9): {len(valid_indices):,}")

        # Filter data and labels
        X_processed = X_processed[valid_indices]
        y = y[valid_indices]

        print(f"   ✅ Filtered data shape: {X_processed.shape}")
        print(f"   ✅ Filtered labels shape: {y.shape}")

    # Show final label distribution
    print(f"🔍 Final label distribution:")
    print(f"   Min label: {y.min()}")
    print(f"   Max label: {y.max()}")
    print(f"   Unique labels: {np.unique(y)}")
    print(f"   Label counts: {np.bincount(y)}")

    # 3. Create memory-efficient splits
    print("\n📊 Creating memory-efficient train/val/test splits...")
    
    n_samples = len(X_processed)
    indices = np.arange(n_samples)
    
    # Split indices (memory efficient)
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices, y, test_size=0.3, random_state=42, stratify=y
    )
    val_idx, test_idx, y_val, y_test = train_test_split(
        temp_idx, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"✅ Splits created:")
    print(f"   Train: {len(train_idx):,} samples")
    print(f"   Val: {len(val_idx):,} samples") 
    print(f"   Test: {len(test_idx):,} samples")
    
    # 4. Create datasets
    print("\n🔄 Creating PyTorch datasets...")
    
    batch_size = 4
    
    train_dataset = HuggingFaceEEGDataset(X_processed[train_idx], y_train)
    val_dataset = HuggingFaceEEGDataset(X_processed[val_idx], y_val)
    test_dataset = HuggingFaceEEGDataset(X_processed[test_idx], y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"✅ DataLoaders created with batch size {batch_size}")
    
    # 5. Create model
    print("\n🧠 Creating BrainDigiCNN model...")
    
    input_size = X_processed.shape[1]
    model = BrainDigiCNN(input_size=input_size, num_classes=10)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"✅ Model created and moved to {device}")
    print(f"   Input size: {input_size:,} features")
    print(f"   Output classes: 10")
    
    # 6. Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n🚀 Starting training...")
    print(f"   Device: {device}")
    print(f"   Optimizer: Adam (lr=0.001)")
    print(f"   Loss: CrossEntropyLoss")
    
    # Simple training loop
    model.train()
    for epoch in range(2):  # Start with 2 epochs
        print(f"\n📊 Epoch {epoch+1}/2")
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % 1000 == 0:
                accuracy = 100. * correct / total
                print(f"   Batch {batch_idx:,}: Loss {loss.item():.4f}, Acc {accuracy:.2f}%")
        
        epoch_accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"   ✅ Epoch {epoch+1} completed: Loss {avg_loss:.4f}, Accuracy {epoch_accuracy:.2f}%")
    
    print("\n🎉 Training completed successfully!")
    print("   Ready for full training with more epochs...")

if __name__ == "__main__":
    main_resume_from_checkpoint()
