#!/usr/bin/env python3
"""
Debug script untuk menganalisis masalah akurasi rendah
"""

import torch
import torch.nn as nn
import numpy as np
from main import MindBigDataLoader, CheckpointManager
from main_pytorch import BrainDigiCNN, EEGDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def debug_data_pipeline():
    """Debug data loading dan preprocessing"""
    print("=== DEBUGGING DATA PIPELINE ===")
    
    # Load checkpoint data
    checkpoint_manager = CheckpointManager()
    
    if checkpoint_manager.checkpoint_exists('preprocessed_data'):
        print("📁 Loading preprocessed data...")
        X_processed = checkpoint_manager.load_checkpoint('preprocessed_data')
        
        organized_data = checkpoint_manager.load_checkpoint('organized_data')
        y = organized_data['y']
        
        print(f"✅ Data loaded:")
        print(f"   X shape: {X_processed.shape}")
        print(f"   y shape: {y.shape}")
        print(f"   X dtype: {X_processed.dtype}")
        print(f"   y dtype: {y.dtype}")
        
        # Check data statistics
        print(f"\n📊 Data Statistics:")
        print(f"   X min: {X_processed.min():.6f}")
        print(f"   X max: {X_processed.max():.6f}")
        print(f"   X mean: {X_processed.mean():.6f}")
        print(f"   X std: {X_processed.std():.6f}")
        
        # Check labels
        print(f"\n🏷️  Label Statistics:")
        unique_labels, counts = np.unique(y, return_counts=True)
        print(f"   Unique labels: {unique_labels}")
        print(f"   Label counts: {counts}")
        print(f"   Label distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"     Label {label}: {count} samples ({count/len(y)*100:.1f}%)")
        
        # Check for NaN or infinite values
        print(f"\n🔍 Data Quality Check:")
        nan_count = np.isnan(X_processed).sum()
        inf_count = np.isinf(X_processed).sum()
        print(f"   NaN values: {nan_count}")
        print(f"   Infinite values: {inf_count}")
        
        if nan_count > 0 or inf_count > 0:
            print("❌ Data contains NaN or infinite values!")
            return None, None
        
        return X_processed, y
    else:
        print("❌ No preprocessed data found!")
        return None, None

def debug_model_architecture(input_size):
    """Debug model architecture"""
    print(f"\n=== DEBUGGING MODEL ARCHITECTURE ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    model = BrainDigiCNN(input_size=input_size, num_classes=10)
    model = model.to(device)
    
    print(f"✅ Model created:")
    print(f"   Input size: {input_size}")
    print(f"   Flatten size: {model.flatten_size}")
    
    # Test forward pass
    print(f"\n🔍 Testing forward pass:")
    batch_size = 4
    test_input = torch.randn(batch_size, input_size).to(device)
    
    try:
        with torch.no_grad():
            output = model(test_input)
        
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")
        
        # Check if output makes sense
        if output.shape[1] != 10:
            print(f"❌ Wrong output shape! Expected (batch_size, 10), got {output.shape}")
            return None
        
        # Apply softmax to see probabilities
        probs = torch.softmax(output, dim=1)
        print(f"   Probability range: [{probs.min():.4f}, {probs.max():.4f}]")
        print(f"   Probability sum per sample: {probs.sum(dim=1)}")
        
        return model
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return None

def debug_training_step(model, X, y):
    """Debug single training step"""
    print(f"\n=== DEBUGGING TRAINING STEP ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare small batch
    batch_size = 8
    X_batch = torch.FloatTensor(X[:batch_size]).to(device)
    y_batch = torch.LongTensor(y[:batch_size]).to(device)
    
    print(f"Batch shapes:")
    print(f"   X_batch: {X_batch.shape}")
    print(f"   y_batch: {y_batch.shape}")
    print(f"   y_batch values: {y_batch}")
    
    # Forward pass
    model.train()
    output = model(X_batch)
    
    print(f"   Output shape: {output.shape}")
    print(f"   Output sample: {output[0]}")
    
    # Loss calculation
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, y_batch)
    
    print(f"   Loss: {loss.item():.6f}")
    
    # Check predictions
    _, predicted = torch.max(output, 1)
    print(f"   Predictions: {predicted}")
    print(f"   Targets: {y_batch}")
    
    accuracy = (predicted == y_batch).float().mean()
    print(f"   Batch accuracy: {accuracy.item():.4f}")
    
    # Backward pass
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    
    print(f"   Gradient norm: {total_norm:.6f}")
    
    if total_norm < 1e-7:
        print("❌ Gradients are too small! Vanishing gradient problem.")
    elif total_norm > 100:
        print("❌ Gradients are too large! Exploding gradient problem.")
    else:
        print("✅ Gradients look normal.")
    
    optimizer.step()
    
    return loss.item(), accuracy.item()

def debug_data_normalization(X):
    """Debug and fix data normalization"""
    print(f"\n=== DEBUGGING DATA NORMALIZATION ===")
    
    print(f"Original data stats:")
    print(f"   Min: {X.min():.6f}")
    print(f"   Max: {X.max():.6f}")
    print(f"   Mean: {X.mean():.6f}")
    print(f"   Std: {X.std():.6f}")
    
    # Try different normalization strategies
    
    # 1. Z-score normalization
    X_zscore = (X - X.mean()) / (X.std() + 1e-8)
    print(f"\nZ-score normalized:")
    print(f"   Min: {X_zscore.min():.6f}")
    print(f"   Max: {X_zscore.max():.6f}")
    print(f"   Mean: {X_zscore.mean():.6f}")
    print(f"   Std: {X_zscore.std():.6f}")
    
    # 2. Min-max normalization
    X_minmax = (X - X.min()) / (X.max() - X.min() + 1e-8)
    print(f"\nMin-max normalized:")
    print(f"   Min: {X_minmax.min():.6f}")
    print(f"   Max: {X_minmax.max():.6f}")
    print(f"   Mean: {X_minmax.mean():.6f}")
    print(f"   Std: {X_minmax.std():.6f}")
    
    # 3. Robust normalization (per sample)
    X_robust = np.zeros_like(X)
    for i in range(X.shape[0]):
        sample = X[i]
        sample_std = sample.std()
        if sample_std > 1e-8:
            X_robust[i] = (sample - sample.mean()) / sample_std
        else:
            X_robust[i] = sample - sample.mean()
    
    print(f"\nRobust normalized (per sample):")
    print(f"   Min: {X_robust.min():.6f}")
    print(f"   Max: {X_robust.max():.6f}")
    print(f"   Mean: {X_robust.mean():.6f}")
    print(f"   Std: {X_robust.std():.6f}")
    
    return X_zscore, X_minmax, X_robust

def main_debug():
    """Main debugging function"""
    print("🔍 Starting comprehensive debugging...")
    
    # 1. Debug data pipeline
    X, y = debug_data_pipeline()
    if X is None:
        return
    
    # 2. Debug data normalization
    X_zscore, X_minmax, X_robust = debug_data_normalization(X)
    
    # 3. Debug model architecture
    model = debug_model_architecture(X.shape[1])
    if model is None:
        return
    
    # 4. Debug training step with different normalizations
    print(f"\n🧪 Testing different normalizations:")
    
    print(f"\n--- Original data ---")
    loss_orig, acc_orig = debug_training_step(model, X, y)
    
    print(f"\n--- Z-score normalized ---")
    loss_zscore, acc_zscore = debug_training_step(model, X_zscore, y)
    
    print(f"\n--- Min-max normalized ---")
    loss_minmax, acc_minmax = debug_training_step(model, X_minmax, y)
    
    print(f"\n--- Robust normalized ---")
    loss_robust, acc_robust = debug_training_step(model, X_robust, y)
    
    # Summary
    print(f"\n📊 NORMALIZATION COMPARISON:")
    print(f"   Original:  Loss={loss_orig:.4f}, Acc={acc_orig:.4f}")
    print(f"   Z-score:   Loss={loss_zscore:.4f}, Acc={acc_zscore:.4f}")
    print(f"   Min-max:   Loss={loss_minmax:.4f}, Acc={acc_minmax:.4f}")
    print(f"   Robust:    Loss={loss_robust:.4f}, Acc={acc_robust:.4f}")
    
    # Recommendation
    best_acc = max(acc_orig, acc_zscore, acc_minmax, acc_robust)
    if best_acc == acc_zscore:
        print(f"\n💡 RECOMMENDATION: Use Z-score normalization")
        return X_zscore
    elif best_acc == acc_minmax:
        print(f"\n💡 RECOMMENDATION: Use Min-max normalization")
        return X_minmax
    elif best_acc == acc_robust:
        print(f"\n💡 RECOMMENDATION: Use Robust normalization")
        return X_robust
    else:
        print(f"\n💡 RECOMMENDATION: Use original data")
        return X

if __name__ == "__main__":
    main_debug()
