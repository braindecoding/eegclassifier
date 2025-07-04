#!/usr/bin/env python3
"""
EEG Digit Classification using Hugging Face MindBigData2022_MNIST_EP dataset
Optimized version with Band-wise EMD-HHT preprocessing
"""

import numpy as np
import matplotlib.pyplot as plt
plt.ion()  # Enable interactive mode for real-time plotting
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import os
import pickle

# Import existing classes from main.py
from main import EEGSignalProcessor, CheckpointManager, OptimizedPreprocessor

# Hugging Face datasets
from datasets import load_dataset

class HuggingFaceEEGDataset(Dataset):
    """
    PyTorch Dataset wrapper for Hugging Face EEG data
    """
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class BrainDigiCNN(nn.Module):
    """
    1D CNN Model exactly matching Table 4 from BrainDigiCNN paper
    """
    def __init__(self, input_size, num_classes=10):
        super(BrainDigiCNN, self).__init__()
        
        # Conv1D layers (exact from paper Table 4)
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
        
        # Calculate flatten size
        self.flatten_size = self._get_flatten_size(input_size)
        
        # Dense layers (exact from paper Table 4)
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        # Dropout for regularization
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
    
    def _get_flatten_size(self, input_size):
        # Calculate size after conv layers
        size = input_size
        for _ in range(4):  # 4 pooling layers
            size = size // 2
        return 32 * size  # 32 filters from last conv layer
    
    def forward(self, x):
        # Reshape for Conv1d: (batch_size, 1, features)
        x = x.unsqueeze(1)
        
        # Conv layers with ReLU + BN + MaxPool (exact from Table 4)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense layers (exact from paper)
        x = self.dropout1(F.relu(self.fc1(x)))  # Dense 128 + ReLU
        x = self.dropout2(F.relu(self.fc2(x)))  # Dense 64 + ReLU
        x = self.fc3(x)                         # Dense 10
        x = F.softmax(x, dim=1)                 # Softmax (as per Table 4)
        
        return x

def load_huggingface_data():
    """
    Load MindBigData2022_MNIST_EP dataset from Hugging Face
    """
    print("📥 Loading MindBigData2022_MNIST_EP from Hugging Face...")
    
    try:
        # Load dataset
        print("   🔄 Downloading dataset...")
        ds = load_dataset("DavidVivancos/MindBigData2022_MNIST_EP")
        
        print(f"   ✅ Dataset loaded successfully")
        print(f"   Dataset info: {ds}")
        
        # Extract train split (assuming it exists)
        if 'train' in ds:
            train_data = ds['train']
            print(f"   Train samples: {len(train_data)}")
        else:
            # Use the first available split
            split_name = list(ds.keys())[0]
            train_data = ds[split_name]
            print(f"   Using split '{split_name}': {len(train_data)} samples")
        
        # Explore data structure
        print(f"   Sample keys: {train_data[0].keys()}")
        
        # Extract features and labels
        print("   🔄 Extracting features and labels...")
        
        features = []
        labels = []
        
        for i, sample in enumerate(train_data):
            if i % 1000 == 0:
                print(f"   Progress: {i}/{len(train_data)}")
            
            # Extract EEG data (adjust key names based on actual dataset structure)
            if 'eeg' in sample:
                eeg_data = np.array(sample['eeg'])
            elif 'signal' in sample:
                eeg_data = np.array(sample['signal'])
            elif 'data' in sample:
                eeg_data = np.array(sample['data'])
            else:
                # Try to find the main data field
                data_keys = [k for k in sample.keys() if k not in ['label', 'digit', 'target']]
                if data_keys:
                    eeg_data = np.array(sample[data_keys[0]])
                else:
                    print(f"   ❌ Cannot find EEG data in sample keys: {sample.keys()}")
                    continue
            
            # Extract label
            if 'label' in sample:
                label = sample['label']
            elif 'digit' in sample:
                label = sample['digit']
            elif 'target' in sample:
                label = sample['target']
            else:
                print(f"   ❌ Cannot find label in sample keys: {sample.keys()}")
                continue
            
            features.append(eeg_data)
            labels.append(label)
        
        features = np.array(features)
        labels = np.array(labels)
        
        print(f"   ✅ Data extraction completed")
        print(f"   Features shape: {features.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Unique labels: {np.unique(labels)}")
        
        return features, labels
        
    except Exception as e:
        print(f"   ❌ Error loading Hugging Face dataset: {e}")
        print(f"   Falling back to original text file method...")
        return None, None

def extract_split_data(split_data):
    """
    GPU-optimized extraction of features and labels from dataset split
    """
    print(f"   🚀 GPU-optimized feature extraction from {len(split_data)} samples...")

    features = []
    labels = []

    # Batch processing for better performance
    batch_size = 1000
    total_samples = len(split_data)

    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)

        # Correct way to access Hugging Face dataset samples
        batch_samples = []
        for i in range(start_idx, end_idx):
            try:
                sample = split_data[i]  # Access individual sample by index
                batch_samples.append(sample)
            except Exception as e:
                print(f"   ❌ Error accessing sample {i}: {e}")
                continue

        # Progress update
        progress = (end_idx / total_samples) * 100
        print(f"   Progress: {end_idx}/{total_samples} ({progress:.1f}%)")

        # Process batch
        batch_features = []
        batch_labels = []

        # Debug: Check sample type and try different access methods
        if len(batch_samples) > 0:
            first_sample = batch_samples[0]
            print(f"   🔍 Debug - Sample type: {type(first_sample)}")

            if isinstance(first_sample, str):
                print(f"   🔍 Debug - Sample is string (first 200 chars): {first_sample[:200]}")
                print(f"   🔧 Trying alternative access methods...")

                # Try accessing as pandas-like
                try:
                    # Maybe it's a datasets.Dataset that needs different access
                    alt_sample = dict(split_data[start_idx])
                    print(f"   ✅ Alternative access successful: {type(alt_sample)}")
                    if isinstance(alt_sample, dict):
                        print(f"   🔍 Alt sample keys: {list(alt_sample.keys())[:10]}...")
                        batch_samples = [dict(split_data[i]) for i in range(start_idx, end_idx)]
                        print(f"   🔄 Updated batch_samples with dict conversion")
                except Exception as e:
                    print(f"   ❌ Alternative access failed: {e}")

            elif isinstance(first_sample, dict):
                print(f"   🔍 Debug - Sample keys: {list(first_sample.keys())[:10]}...")
                print(f"   ✅ Sample is dictionary - should work correctly")
            else:
                print(f"   🔍 Debug - Unexpected type, content: {first_sample}")

                # Try converting to dict
                try:
                    dict_sample = dict(first_sample)
                    print(f"   ✅ Conversion to dict successful: {list(dict_sample.keys())[:10]}...")
                    batch_samples = [dict(sample) for sample in batch_samples]
                    print(f"   🔄 Updated batch_samples with dict conversion")
                except Exception as e:
                    print(f"   ❌ Dict conversion failed: {e}")

        for sample in batch_samples:
            try:
                # Handle different sample types
                if isinstance(sample, str):
                    print(f"   ❌ Sample is string, not dictionary. Need different parsing approach.")
                    print(f"   Sample content: {sample[:100]}...")
                    continue
                elif not isinstance(sample, dict):
                    print(f"   ❌ Unexpected sample type: {type(sample)}")
                    continue

                # Extract EEG data from flattened time-series format
                # Format: 'CHANNEL-TIMEPOINT' (e.g., 'AF3-0', 'AF3-1', ..., 'AF4-255')

                # Get all EEG channel keys (exclude label)
                eeg_keys = [k for k in sample.keys() if k not in ['label', 'digit', 'target', 'y', 'class']]
            except Exception as e:
                print(f"   ❌ Error processing sample: {e}")
                continue

            if not eeg_keys:
                continue

            # Group by channel and reconstruct time series
            channels_data = {}

            for key in eeg_keys:
                if '-' in key:  # Format: 'CHANNEL-TIMEPOINT'
                    channel, timepoint = key.rsplit('-', 1)
                    try:
                        timepoint_idx = int(timepoint)
                        if channel not in channels_data:
                            channels_data[channel] = {}
                        channels_data[channel][timepoint_idx] = float(sample[key])
                    except ValueError:
                        continue

            # Convert to numpy arrays per channel
            channel_arrays = []
            for channel in sorted(channels_data.keys()):
                # Sort by timepoint index and extract values
                timepoints = sorted(channels_data[channel].keys())
                channel_signal = [channels_data[channel][tp] for tp in timepoints]
                channel_arrays.append(channel_signal)

            if not channel_arrays:
                continue

            # Stack channels: shape (n_channels, n_timepoints)
            eeg_data = np.array(channel_arrays, dtype=np.float32)

            # Extract label
            label = None
            for key in ['label', 'digit', 'target', 'y', 'class']:
                if key in sample:
                    label = int(sample[key])
                    break

            if label is None:
                continue

            batch_features.append(eeg_data)
            batch_labels.append(label)

        # Add batch to main lists
        features.extend(batch_features)
        labels.extend(batch_labels)

        # Memory cleanup for large datasets
        del batch_features, batch_labels

    print(f"   ✅ Feature extraction completed: {len(features)} samples")

    # Convert to numpy arrays with GPU-friendly dtype
    features_array = np.array(features, dtype=np.float32)  # float32 for GPU efficiency
    labels_array = np.array(labels, dtype=np.int64)        # int64 for PyTorch compatibility

    print(f"   📊 Final arrays: features {features_array.shape}, labels {labels_array.shape}")

    return features_array, labels_array

def preprocess_huggingface_data(X_raw, y, use_checkpoint=True):
    """
    Apply Band-wise EMD-HHT preprocessing to Hugging Face data
    """
    print("\n🧠 Preprocessing Hugging Face EEG data...")
    
    checkpoint_manager = CheckpointManager()
    
    if use_checkpoint and checkpoint_manager.checkpoint_exists('hf_normalized_data'):
        print("   📁 Loading preprocessed Hugging Face data from checkpoint...")
        X_processed = checkpoint_manager.load_checkpoint('hf_normalized_data')
        print(f"   ✅ Loaded preprocessed data: {X_processed.shape}")
        return X_processed
    
    # Use OptimizedPreprocessor for Band-wise EMD-HHT
    print("   🚀 Starting Band-wise EMD-HHT preprocessing...")
    
    optimized_processor = OptimizedPreprocessor(
        sampling_rate=128,  # Adjust based on actual sampling rate
        n_processes=None,
        batch_size=64
    )
    
    X_processed = optimized_processor.process_optimized(X_raw)
    
    # Apply robust normalization
    print("   🔄 Applying robust normalization...")
    
    batch_size = 1000
    n_samples = X_processed.shape[0]
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        
        if start_idx % 5000 == 0:
            print(f"   Progress: {end_idx}/{n_samples} ({end_idx/n_samples*100:.1f}%)")
        
        for i in range(start_idx, end_idx):
            sample = X_processed[i]
            sample_mean = sample.mean()
            sample_std = sample.std()
            
            if sample_std > 1e-6:
                sample -= sample_mean
                sample /= sample_std
                np.clip(sample, -10, 10, out=sample)
            else:
                sample -= sample_mean
    
    print(f"   ✅ Preprocessing completed")
    print(f"   Final shape: {X_processed.shape}")
    
    # Save checkpoint
    if use_checkpoint:
        checkpoint_manager.save_checkpoint('hf_normalized_data', X_processed)
    
    return X_processed

def main_huggingface_pipeline():
    """
    Main pipeline for Hugging Face dataset with standardized benchmarking
    """
    print("🚀 EEG Digit Classification - Hugging Face Dataset")
    print("=" * 60)
    print("🎯 Using standardized dataset for apple-to-apple comparison!")

    # 1. Load Hugging Face dataset
    print("📥 Loading MindBigData2022_MNIST_EP from Hugging Face...")

    try:
        ds = load_dataset("DavidVivancos/MindBigData2022_MNIST_EP")
        print(f"   ✅ Dataset loaded successfully")
        print(f"   Available splits: {list(ds.keys())}")

        # Use first available split for now (can be improved later)
        split_name = list(ds.keys())[0]
        data_split = ds[split_name]
        print(f"   Using split: {split_name} ({len(data_split)} samples)")

        # Extract data from the split
        print("   🔄 Extracting features and labels...")
        X_raw, y = extract_split_data(data_split)

        print(f"   ✅ Data extracted: {X_raw.shape}, labels: {y.shape}")
        print(f"   Unique labels: {np.unique(y)}")

    except Exception as e:
        print(f"   ❌ Failed to load Hugging Face dataset: {e}")
        print("   💡 Make sure to install: pip install datasets")
        return

    # 2. Preprocess data with Band-wise EMD-HHT
    X_processed = preprocess_huggingface_data(X_raw, y)

    # 3. Create reproducible splits for fair comparison
    print("\n📊 Creating reproducible train/val/test splits...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_processed, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # 3. Use standard dataset splits (if available) or create reproducible splits
    print("\n📊 Using dataset splits...")

    # Check if dataset has predefined splits
    if hasattr(ds, 'keys') and len(ds.keys()) > 1:
        print("   ✅ Using predefined dataset splits for standardized benchmarking")
        # Use predefined splits for fair comparison with other methods
        # This ensures apple-to-apple comparison with other research

        # Extract each split separately
        splits = list(ds.keys())
        print(f"   Available splits: {splits}")

        # Process each split
        if 'train' in splits:
            X_train, y_train = extract_split_data(ds['train'])
        if 'validation' in splits or 'val' in splits:
            val_key = 'validation' if 'validation' in splits else 'val'
            X_val, y_val = extract_split_data(ds[val_key])
        if 'test' in splits:
            X_test, y_test = extract_split_data(ds['test'])

        # If missing validation set, create from train
        if 'validation' not in splits and 'val' not in splits:
            print("   Creating validation set from training data...")
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
            )
    else:
        print("   ⚠️  No predefined splits found, creating reproducible splits...")
        print("   Note: For standardized benchmarking, predefined splits are preferred")

        # Create reproducible splits for consistency
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_processed, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

    print(f"   📊 Final split sizes:")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")
    print(f"   Test: {len(X_test)} samples")
    print(f"   ")
    print(f"   🎯 Standardized splits enable fair comparison with other methods!")
    
    # 4. Create PyTorch datasets and dataloaders
    print("\n🔄 Creating PyTorch datasets...")
    
    batch_size = 4  # Memory-optimized batch size
    
    train_dataset = HuggingFaceEEGDataset(X_train, y_train)
    val_dataset = HuggingFaceEEGDataset(X_val, y_val)
    test_dataset = HuggingFaceEEGDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 5. Create model
    print("\n🧠 Creating BrainDigiCNN model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    model = BrainDigiCNN(input_size=X_processed.shape[1], num_classes=10)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")
    
    # 6. Train model
    print("\n🚀 Starting training...")
    
    # Training will be implemented similar to main_pytorch.py
    # with memory-efficient gradient accumulation
    
    print("🎯 Hugging Face pipeline setup completed!")
    print("   Ready for training with optimized dataset loading!")

if __name__ == "__main__":
    main_huggingface_pipeline()
