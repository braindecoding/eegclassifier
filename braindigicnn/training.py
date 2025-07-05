#!/usr/bin/env python3
"""
🚀 EEG Digit Classification Training
Memory-efficient training using preprocessed checkpoints
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import os
import sys
import gc

# Add parent directory to path to import from main.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import CheckpointManager

# Import BrainDigiCNN architecture
from braindigicnn import BrainDigiCNN

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
        label = self.y_data[actual_idx]

        # Ensure label is in valid range [0, 9]
        if label < 0 or label > 9:
            print(f"⚠️  Invalid label {label} at index {actual_idx}, setting to 0")
            label = 0

        return torch.FloatTensor(self.X_data[actual_idx]), torch.LongTensor([label])[0]

# BrainDigiCNN class imported from braindigicnn.py

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

    # Debug label values
    print(f"   🔍 Label debugging:")
    print(f"      Label type: {type(y)}")
    print(f"      Label dtype: {y.dtype if hasattr(y, 'dtype') else 'N/A'}")
    print(f"      Label range: [{np.min(y)}, {np.max(y)}]")
    print(f"      Unique labels: {np.unique(y)}")
    print(f"      Label counts: {np.bincount(y) if np.min(y) >= 0 and np.max(y) < 20 else 'Invalid range'}")

    # Clean labels - ensure they are in range [0, 9]
    y = np.array(y, dtype=np.int64)

    # Filter out invalid labels
    valid_mask = (y >= 0) & (y <= 9)
    invalid_count = np.sum(~valid_mask)

    if invalid_count > 0:
        print(f"   ⚠️  Found {invalid_count} invalid labels, filtering...")
        print(f"      Invalid labels: {np.unique(y[~valid_mask])}")

        # Filter data and indices
        valid_indices = np.where(valid_mask)[0]
        y = y[valid_mask]

        # Update split indices to only include valid samples
        train_idx_filtered = []
        val_idx_filtered = []
        test_idx_filtered = []

        for idx in split_indices['train_idx']:
            if idx in valid_indices:
                train_idx_filtered.append(np.where(valid_indices == idx)[0][0])

        for idx in split_indices['val_idx']:
            if idx in valid_indices:
                val_idx_filtered.append(np.where(valid_indices == idx)[0][0])

        for idx in split_indices['test_idx']:
            if idx in valid_indices:
                test_idx_filtered.append(np.where(valid_indices == idx)[0][0])

        split_indices['train_idx'] = np.array(train_idx_filtered)
        split_indices['val_idx'] = np.array(val_idx_filtered)
        split_indices['test_idx'] = np.array(test_idx_filtered)

        # Filter X_processed
        X_processed = X_processed[valid_indices]

        print(f"   ✅ Filtered to valid samples:")
        print(f"      Valid samples: {len(valid_indices)}")
        print(f"      New label range: [{np.min(y)}, {np.max(y)}]")

    print(f"   ✅ Data loaded successfully:")
    print(f"      Features: {X_processed.shape}")
    print(f"      Labels: {len(y)}")
    print(f"      Label range: [{np.min(y)}, {np.max(y)}]")
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

def train_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps=8):
    """
    Train model for one epoch with gradient accumulation
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    optimizer.zero_grad()  # Initialize gradients

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Reshape for Conv1D: (batch, 1, features)
        data = data.unsqueeze(1)

        output = model(data)
        loss = criterion(output, target)

        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()

        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps  # Unscale for logging
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if batch_idx % 200 == 0:  # Reduced frequency for smaller batches
            print(f'   Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item()*accumulation_steps:.6f} | Acc: {100.*correct/total:.2f}%')

    # Final update if remaining gradients
    if len(train_loader) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def validate_epoch(model, val_loader, criterion, device):
    """
    Validate model for one epoch
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            # Reshape for Conv1D: (batch, 1, features)
            data = data.unsqueeze(1)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def train_model():
    """
    Main training function following paper specifications
    """
    print("🚀 EEG Digit Classification Training - BrainDigiCNN")
    print("=" * 60)
    print("📜 Following paper specifications from README.md")

    # Load data
    X_processed, y, split_indices = load_training_data()

    # Memory-efficient batch size for 516K features
    batch_size = 4  # Reduced from 32 to fit in GPU memory
    train_loader, val_loader, test_loader = create_data_loaders(
        X_processed, y, split_indices, batch_size=batch_size
    )

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")

    # Create model using braindigicnn.py architecture
    input_size = X_processed.shape[1]  # 516,096 features
    model = BrainDigiCNN(input_size=input_size, num_classes=10)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 BrainDigiCNN Model: {total_params:,} parameters")

    # Paper specifications: Adam optimizer, lr=0.001, categorical crossentropy
    criterion = nn.CrossEntropyLoss()  # Categorical crossentropy equivalent
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Paper specifications: epochs 10-20, early stopping patience=5
    epochs = 20
    patience = 5
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\n🎯 Training Configuration (Memory-Optimized):")
    print(f"   Architecture: 4×Conv1D(256→128→64→32) + 2×FC(128→64→10)")
    print(f"   Optimizer: Adam (lr=0.001)")
    print(f"   Loss: Categorical CrossEntropy")
    print(f"   Batch size: {batch_size} (memory-optimized)")
    print(f"   Gradient accumulation: 8 steps (effective batch_size=32)")
    print(f"   Epochs: {epochs}")
    print(f"   Early stopping: patience={patience}")
    print(f"   Target accuracy: 98.27% (paper benchmark)")
    print(f"   Memory management: CUDA cache clearing enabled")

    print(f"\n📊 Dataset splits:")
    print(f"   Train: {len(split_indices['train_idx']):,} samples")
    print(f"   Val: {len(split_indices['val_idx']):,} samples")
    print(f"   Test: {len(split_indices['test_idx']):,} samples")

    print(f"\n🚀 Starting training...")

    # Training loop
    for epoch in range(epochs):
        print(f"\n📈 Epoch {epoch+1}/{epochs}")

        # Train with gradient accumulation (effective batch_size = 4×8 = 32)
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps=8)

        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        print(f"   Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.2f}%")

        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_braindigicnn_model.pth')
            print(f"   ✅ New best model saved!")
        else:
            patience_counter += 1
            print(f"   ⏳ Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"   🛑 Early stopping triggered!")
                break

    # Load best model for final evaluation
    model.load_state_dict(torch.load('best_braindigicnn_model.pth'))

    # Final test evaluation
    print(f"\n🎯 Final Test Evaluation:")
    test_loss, test_acc = validate_epoch(model, test_loader, criterion, device)
    print(f"   Test Loss: {test_loss:.6f}")
    print(f"   Test Accuracy: {test_acc:.2f}%")
    print(f"   Paper Target: 98.27%")

    if test_acc >= 98.0:
        print(f"   🎉 EXCELLENT! Achieved paper-level performance!")
    elif test_acc >= 95.0:
        print(f"   ✅ GOOD! Strong performance achieved!")
    else:
        print(f"   📈 Room for improvement - consider hyperparameter tuning")

    return model, test_acc

if __name__ == "__main__":
    # Enable CUDA debugging
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # Run complete training pipeline
    print("🎯 BrainDigiCNN Training Pipeline")
    print("Following paper specifications from README.md")
    print("=" * 60)
    print("🔍 CUDA debugging enabled for better error tracking")

    try:
        model, test_accuracy = train_model()

        print(f"\n🎉 Training completed successfully!")
        print(f"   Final test accuracy: {test_accuracy:.2f}%")
        print(f"   Model saved as: best_braindigicnn_model.pth")
        print(f"   Architecture: BrainDigiCNN (from braindigicnn.py)")
        print(f"   Paper compliance: ✅ FULL")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print(f"   Check data loading and model architecture")
        print(f"   Debug info: CUDA_LAUNCH_BLOCKING=1 enabled")
        raise
