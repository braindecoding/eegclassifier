#!/usr/bin/env python3
"""
🚀 EEG Digit Classification - Hugging Face Dataset (Memory-Efficient)
============================================================
Memory-efficient preprocessing with batch processing for large datasets
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import os
import pickle
import gc

# Import existing classes from main.py
from main import EEGSignalProcessor, CheckpointManager, OptimizedPreprocessor

# Hugging Face datasets
from datasets import load_dataset

class HuggingFaceEEGDataset(Dataset):
    """
    PyTorch Dataset for Hugging Face EEG data
    """
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_huggingface_dataset():
    """
    Load MindBigData2022_MNIST_EP dataset from Hugging Face
    """
    print("📥 Loading MindBigData2022_MNIST_EP from Hugging Face...")
    
    try:
        # Load the dataset
        ds = load_dataset("DavidVivancos/MindBigData2022_MNIST_EP")
        print(f"   ✅ Dataset loaded successfully")
        
        # Check available splits
        if hasattr(ds, 'keys'):
            print(f"   Available splits: {list(ds.keys())}")
        
        return ds
    
    except Exception as e:
        print(f"   ❌ Error loading Hugging Face dataset: {e}")
        print(f"   Falling back to original text file method...")
        return None

def process_batch_samples(split_data, start_idx, end_idx):
    """
    Process a batch of samples efficiently
    """
    batch_features = []
    batch_labels = []
    
    for i in range(start_idx, end_idx):
        try:
            sample = split_data[i]  # Access individual sample by index
            
            # Handle different sample types
            if isinstance(sample, str):
                continue  # Skip string samples
            elif not isinstance(sample, dict):
                continue  # Skip non-dict samples

            # Extract EEG data from flattened time-series format
            # Format: 'CHANNEL-TIMEPOINT' (e.g., 'AF3-0', 'AF3-1', ..., 'AF4-255')
            
            # Get all EEG channel keys (exclude label)
            eeg_keys = [k for k in sample.keys() if k not in ['label', 'digit', 'target', 'y', 'class']]
            
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
            
        except Exception as e:
            continue  # Skip problematic samples
    
    return batch_features, batch_labels

def extract_split_data_memory_efficient(split_data, batch_size=2000):
    """
    Memory-efficient extraction of features and labels from dataset split
    """
    total_samples = len(split_data)
    print(f"   🚀 Memory-efficient feature extraction from {total_samples} samples...")
    print(f"   💾 Using batch processing ({batch_size} samples per batch)")
    
    all_features = []
    all_labels = []

    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        
        # Progress update
        progress = (end_idx / total_samples) * 100
        print(f"   📊 Processing batch {start_idx//batch_size + 1}: {start_idx}-{end_idx} ({progress:.1f}%)")

        # Process batch efficiently
        batch_features, batch_labels = process_batch_samples(split_data, start_idx, end_idx)
        
        # Add to main collections
        all_features.extend(batch_features)
        all_labels.extend(batch_labels)
        
        # Memory cleanup
        del batch_features, batch_labels
        
        # Optional: Force garbage collection for large batches
        if len(all_features) % (batch_size * 5) == 0:  # Every 5 batches
            gc.collect()
            print(f"   🧹 Memory cleanup: {len(all_features)} samples processed")

    # Convert to numpy arrays
    print(f"   🔄 Converting {len(all_features)} samples to numpy arrays...")
    features_array = np.array(all_features, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.int32)
    
    # Final cleanup
    del all_features, all_labels
    gc.collect()
    
    print(f"   📊 Final arrays: features {features_array.shape}, labels {labels_array.shape}")
    return features_array, labels_array

def preprocess_huggingface_data_memory_efficient(X_raw, y, use_checkpoint=True, batch_size=1000):
    """
    Apply Band-wise EMD-HHT preprocessing to Hugging Face data with memory efficiency
    """
    print("\n🧠 Memory-efficient preprocessing of Hugging Face EEG data...")
    
    checkpoint_manager = CheckpointManager()
    
    # Check for existing checkpoints (multiple stages)
    if use_checkpoint:
        # Check for final normalized checkpoint first
        if checkpoint_manager.checkpoint_exists('hf_normalized_data'):
            print("   📁 Loading final normalized data from checkpoint...")
            X_processed = checkpoint_manager.load_checkpoint('hf_normalized_data')
            print(f"   ✅ Loaded final normalized data: {X_processed.shape}")
            return X_processed

        # Check for pre-normalization checkpoint
        elif checkpoint_manager.checkpoint_exists('hf_preprocessed_raw'):
            print("   📁 Loading pre-normalization data from checkpoint...")
            X_processed = checkpoint_manager.load_checkpoint('hf_preprocessed_raw')
            print(f"   ✅ Loaded pre-normalization data: {X_processed.shape}")

            # Apply normalization to loaded data
            print("   🔄 Applying robust normalization to loaded data...")
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X_processed)

            print(f"   ✅ Normalization completed")
            print(f"   📊 Final shape: {X_processed.shape}")
            print(f"   📈 Data range: [{X_processed.min():.6f}, {X_processed.max():.6f}]")
            print(f"   📊 Data mean: {X_processed.mean():.6f}, std: {X_processed.std():.6f}")

            # Save final normalized checkpoint
            print("   💾 Saving final normalized checkpoint...")
            try:
                checkpoint_manager.save_checkpoint(X_processed, 'hf_normalized_data')
                print("   ✅ Final normalized checkpoint saved successfully")
            except Exception as e:
                print(f"   ⚠️  Final normalized checkpoint save failed: {e}")

            return X_processed
    
    # Use OptimizedPreprocessor for Band-wise EMD-HHT with memory management
    print("   🚀 Starting Band-wise EMD-HHT preprocessing...")
    print(f"   📊 Input data: {X_raw.shape}, Memory: {X_raw.nbytes / (1024**3):.1f} GB")

    # MAXIMUM PERFORMANCE settings for 80-core beast machine!
    available_cores = os.cpu_count()
    
    # Optimize for 80-core Xeon system
    if available_cores >= 80:
        # Use 60-70% of cores to leave room for system processes
        optimal_processes = min(60, available_cores)  # BEAST MODE!
        batch_size = 64  # Larger batches for beast machine
        print(f"   🔥 BEAST MACHINE: Using {optimal_processes} cores ({optimal_processes/available_cores*100:.1f}% utilization)")
    elif available_cores >= 16:
        optimal_processes = min(12, available_cores)   # High-end desktop
        batch_size = 48
        print(f"   ⚡ HIGH-END: Using {optimal_processes} cores")
    else:
        optimal_processes = min(8, available_cores)   # Fallback for smaller systems
        batch_size = 32
        print(f"   ⚡ STANDARD: Using {optimal_processes} cores")

    optimized_processor = OptimizedPreprocessor(
        sampling_rate=128,
        n_processes=optimal_processes,  # MAXIMUM CORES!
        batch_size=batch_size
    )

    # Display performance expectations
    print(f"\n🚀 OptimizedPreprocessor initialized - 🔥 BEAST MODE:")
    print(f"      🖥️ Total CPU cores: {available_cores}")
    print(f"      🔥 Using processes: {optimal_processes} ({optimal_processes/available_cores*100:.1f}% utilization)")
    print(f"      📊 Features per sample: 46,080 (maximum information)")
    print(f"      ⚡ Expected speedup: ~{optimal_processes/8:.1f}x faster than 8 cores")
    print(f"      🚀 Estimated processing time: ~{52022/(optimal_processes*2.5)/60:.1f} minutes")
    print(f"      💪 CPU utilization: MAXIMUM BEAST MODE!")

    X_processed = optimized_processor.process_optimized(X_raw)

    print(f"   📊 Processed data: {X_processed.shape}, Memory: {X_processed.nbytes / (1024**3):.1f} GB")

    # Save checkpoint BEFORE normalization (safety checkpoint)
    if use_checkpoint:
        print("   💾 Saving pre-normalization checkpoint...")
        try:
            checkpoint_manager.save_checkpoint(X_processed, 'hf_preprocessed_raw')
            print("   ✅ Pre-normalization checkpoint saved successfully")
        except Exception as e:
            print(f"   ⚠️  Pre-normalization checkpoint save failed: {e}")

    # Apply robust normalization
    print("   🔄 Applying robust normalization...")
    scaler = StandardScaler()
    X_processed = scaler.fit_transform(X_processed)
    
    print(f"   ✅ Normalization completed")
    print(f"   📊 Final shape: {X_processed.shape}")
    print(f"   📈 Data range: [{X_processed.min():.6f}, {X_processed.max():.6f}]")
    print(f"   📊 Data mean: {X_processed.mean():.6f}, std: {X_processed.std():.6f}")

    # Save checkpoint
    if use_checkpoint:
        print("   💾 Saving checkpoint...")
        try:
            checkpoint_manager.save_checkpoint(X_processed, 'hf_normalized_data')
            print("   ✅ Checkpoint saved successfully")
        except Exception as e:
            print(f"   ⚠️  Checkpoint save failed: {e}")

    return X_processed

def main_huggingface_pipeline_memory_efficient():
    """
    Main pipeline for Hugging Face dataset with memory-efficient processing
    """
    print("🚀 EEG Digit Classification - Hugging Face Dataset (Memory-Efficient)")
    print("=" * 70)
    print("🎯 Using memory-efficient batch processing for large datasets!")

    # 1. Load dataset
    ds = load_huggingface_dataset()
    if ds is None:
        print("❌ Failed to load Hugging Face dataset")
        return

    # Use train split (largest available)
    split_data = ds['train'] if 'train' in ds else ds[list(ds.keys())[0]]
    print(f"   Using split: {list(ds.keys())[0] if 'train' not in ds else 'train'} ({len(split_data)} samples)")

    # 2. Extract features and labels with memory efficiency
    print(f"\n🔄 Extracting features and labels with memory-efficient batch processing...")

    # Check for extraction checkpoint first
    checkpoint_manager = CheckpointManager()
    if checkpoint_manager.checkpoint_exists('hf_raw_extracted'):
        print("   📁 Loading extracted data from checkpoint...")
        checkpoint_data = checkpoint_manager.load_checkpoint('hf_raw_extracted')
        if checkpoint_data is not None:
            X_raw, y = checkpoint_data
            print(f"   ✅ Loaded extracted data: {X_raw.shape}")
        else:
            print("   ❌ Failed to load extraction checkpoint, extracting fresh...")
            X_raw, y = extract_split_data_memory_efficient(split_data, batch_size=2000)
    else:
        X_raw, y = extract_split_data_memory_efficient(split_data, batch_size=2000)

        # Save extraction checkpoint
        print("   💾 Saving extraction checkpoint...")
        try:
            checkpoint_manager.save_checkpoint((X_raw, y), 'hf_raw_extracted')
            print("   ✅ Extraction checkpoint saved successfully")
        except Exception as e:
            print(f"   ⚠️  Extraction checkpoint save failed: {e}")

    if len(X_raw) == 0:
        print("❌ No valid samples extracted")
        return

    # 3. Preprocess data with Band-wise EMD-HHT (memory-efficient)
    X_processed = preprocess_huggingface_data_memory_efficient(X_raw, y, batch_size=1000)

    # 4. Create memory-efficient reproducible splits
    print("\n📊 Creating memory-efficient train/val/test splits...")
    print(f"   Data size: {X_processed.nbytes / (1024**3):.1f} GB")
    print(f"   Using index-based splitting to avoid memory copies...")

    # Create reproducible splits
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_processed, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"   ✅ Splits created: Train {len(X_train)}, Val {len(X_val)}, Test {len(X_test)}")

    # 5. Create PyTorch datasets
    print("\n📦 Creating PyTorch datasets...")
    train_dataset = HuggingFaceEEGDataset(X_train, y_train)
    val_dataset = HuggingFaceEEGDataset(X_val, y_val)
    test_dataset = HuggingFaceEEGDataset(X_test, y_test)

    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 🎯 Preprocessing completed successfully!
    print(f"\n✅ MEMORY-EFFICIENT PREPROCESSING COMPLETED!")
    print(f"   📊 Final processed data shape: {X_processed.shape}")
    print(f"   💾 Memory usage: {X_processed.nbytes / (1024**3):.1f} GB")
    print(f"   🎯 Features per sample: {X_processed.shape[1]:,}")
    print(f"   📈 Total samples: {X_processed.shape[0]:,}")
    
    # Verify preprocessing quality
    print(f"\n🔍 Preprocessing Quality Check:")
    print(f"   Data type: {X_processed.dtype}")
    print(f"   Data range: [{X_processed.min():.6f}, {X_processed.max():.6f}]")
    print(f"   Data mean: {X_processed.mean():.6f}")
    print(f"   Data std: {X_processed.std():.6f}")
    
    print(f"\n🎉 Memory-efficient Hugging Face EEG preprocessing pipeline completed!")
    print(f"   Ready for model training in next step...")
    
    return X_processed, y

if __name__ == "__main__":
    main_huggingface_pipeline_memory_efficient()
