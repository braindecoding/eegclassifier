#!/usr/bin/env python3
"""
Memory-Efficient EEG Digit Classification using Hugging Face dataset
Optimized with 6 IMFs per band instead of 10 for memory efficiency
"""

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import time
import os
import pickle

# Import existing classes from main.py
from main import EEGSignalProcessor, CheckpointManager

# Hugging Face datasets
from datasets import load_dataset

class MemoryEfficientPreprocessor:
    """
    Memory-efficient preprocessor with 6 IMFs per band instead of 10
    """
    
    def __init__(self, sampling_rate=128, n_processes=None, batch_size=32):
        self.sampling_rate = sampling_rate
        
        # BEAST MODE optimization
        total_cores = os.cpu_count()
        if n_processes is None:
            if total_cores >= 80:
                self.n_processes = min(60, total_cores)
                self.batch_size = 16
                mode = "🔥 BEAST MODE"
            elif total_cores >= 40:
                self.n_processes = min(32, total_cores)
                self.batch_size = 24
                mode = "🚀 HIGH-PERF"
            else:
                self.n_processes = min(8, total_cores)
                self.batch_size = 32
                mode = "⚡ STANDARD"
        else:
            self.n_processes = min(n_processes, total_cores)
            self.batch_size = batch_size
            mode = "🔧 CUSTOM"

        print(f"💾 Memory-Efficient Preprocessor - {mode}:")
        print(f"   🖥️  Total CPU cores: {total_cores}")
        print(f"   🔥 Using processes: {self.n_processes}")
        print(f"   📦 Batch size: {self.batch_size}")
        print(f"   💡 IMFs per band: 6 (reduced from 10 for memory efficiency)")
        print(f"   📊 Memory reduction: ~40% less than standard approach")
        
        # Calculate expected features
        expected_features = 6 * 6 * 768  # 6 bands × 6 IMFs × 768 features
        print(f"   🎯 Features per sample: {expected_features:,} (vs 46,080 with 10 IMFs)")
        
        if total_cores >= 80:
            estimated_time = 52022 / (self.n_processes * 2.5) / 60
            print(f"   ⚡ Estimated processing time: ~{estimated_time:.1f} minutes")

    def process_single_sample_memory_efficient(self, args):
        """
        Process single sample with memory-efficient 6 IMFs per band
        """
        sample_idx, eeg_sample, sampling_rate = args
        
        try:
            # Initialize processor with memory-efficient settings
            processor = EEGSignalProcessor(sampling_rate=sampling_rate)
            
            # Override EMD to use 6 IMFs instead of 10
            original_emd = processor.empirical_mode_decomposition
            
            def memory_efficient_emd(data, num_imf=6):
                return original_emd(data, num_imf=6)
            
            processor.empirical_mode_decomposition = memory_efficient_emd
            
            # Process each channel with memory-efficient EMD
            processed_channels = []
            for ch in range(eeg_sample.shape[0]):
                try:
                    features = processor.process_eeg_signal(eeg_sample[ch])
                    processed_channels.append(features)
                except Exception as e:
                    # Fallback: use raw signal
                    processed_channels.append(eeg_sample[ch])
            
            # Ensure consistent length
            min_length = min(len(ch) for ch in processed_channels)
            processed_channels = [ch[:min_length] for ch in processed_channels]
            
            processed_sample = np.concatenate(processed_channels)
            return sample_idx, processed_sample
            
        except Exception as e:
            print(f"   ❌ Error processing sample {sample_idx}: {e}")
            return sample_idx, eeg_sample.flatten()

    def process_optimized_memory_efficient(self, X_raw):
        """
        Process with memory-efficient multiprocessing
        """
        print(f"   🚀 Starting memory-efficient Band-wise EMD-HHT preprocessing...")
        print(f"   📊 Input: {X_raw.shape}")
        print(f"   💾 Memory optimization: 6 IMFs per band (40% memory reduction)")
        
        from multiprocessing import Pool
        import time
        
        # Prepare arguments
        args_list = [(i, sample, self.sampling_rate) for i, sample in enumerate(X_raw)]
        start_time = time.time()
        
        # Process with multiprocessing
        with Pool(processes=self.n_processes) as pool:
            results = []
            chunk_size = max(1, len(args_list) // 50)
            
            for i in range(0, len(args_list), chunk_size):
                chunk = args_list[i:i + chunk_size]
                chunk_results = pool.map(self.process_single_sample_memory_efficient, chunk)
                results.extend(chunk_results)
                
                # Enhanced progress tracking
                progress = min(100, (i + len(chunk)) * 100 // len(args_list))
                elapsed = time.time() - start_time
                samples_processed = i + len(chunk)
                
                if samples_processed > 0:
                    speed = samples_processed / elapsed
                    remaining_samples = len(args_list) - samples_processed
                    eta_seconds = remaining_samples / speed if speed > 0 else 0
                    eta_minutes = eta_seconds / 60
                    
                    print(f"   💾 Memory-Efficient Progress: {progress:3.1f}% ({samples_processed:,}/{len(args_list):,}) | "
                          f"Speed: {speed:.1f} samples/s | "
                          f"Elapsed: {elapsed/60:.1f}m | "
                          f"ETA: {eta_minutes:.1f}m")
        
        # Sort results and extract processed data
        results.sort(key=lambda x: x[0])
        X_processed = np.array([result[1] for result in results])
        
        elapsed_total = time.time() - start_time
        print(f"   ✅ Memory-efficient processing completed in {elapsed_total:.1f}s")
        print(f"   📊 Output shape: {X_processed.shape}")
        print(f"   💾 Memory usage: ~{X_processed.nbytes / (1024**3):.1f} GB (40% less than standard)")
        
        return X_processed

def main_memory_efficient_pipeline():
    """
    Main pipeline with memory-efficient processing
    """
    print("🚀 Memory-Efficient EEG Digit Classification - Hugging Face Dataset")
    print("=" * 70)
    print("💾 Optimized with 6 IMFs per band for memory efficiency")
    print("🎯 Using standardized dataset for apple-to-apple comparison!")
    
    # Load Hugging Face dataset (same as before)
    print("📥 Loading MindBigData2022_MNIST_EP from Hugging Face...")
    
    try:
        ds = load_dataset("DavidVivancos/MindBigData2022_MNIST_EP")
        print(f"   ✅ Dataset loaded successfully")
        print(f"   Available splits: {list(ds.keys())}")
        
        split_name = list(ds.keys())[0]
        data_split = ds[split_name]
        print(f"   Using split: {split_name} ({len(data_split)} samples)")
        
        # Extract data (same extraction logic)
        from main_huggingface import extract_split_data
        X_raw, y = extract_split_data(data_split)
        
        print(f"   ✅ Data extracted: {X_raw.shape}, labels: {y.shape}")
        
    except Exception as e:
        print(f"   ❌ Failed to load dataset: {e}")
        return
    
    # Memory-efficient preprocessing
    print("\n🧠 Memory-Efficient Preprocessing...")
    
    checkpoint_manager = CheckpointManager()
    
    if checkpoint_manager.checkpoint_exists('hf_memory_efficient_data'):
        print("   📁 Loading memory-efficient data from checkpoint...")
        X_processed = checkpoint_manager.load_checkpoint('hf_memory_efficient_data')
        print(f"   ✅ Loaded: {X_processed.shape}")
    else:
        # Process with memory-efficient approach
        processor = MemoryEfficientPreprocessor()
        X_processed = processor.process_optimized_memory_efficient(X_raw)
        
        # Save with zero-copy method
        print(f"\n💾 Saving memory-efficient checkpoint...")
        try:
            from memory_efficient_saver import MemoryEfficientSaver
            saver = MemoryEfficientSaver(checkpoint_manager.checkpoint_dir)
            success = saver.save_zero_copy(X_processed, 'hf_memory_efficient_data')
            
            if success:
                print(f"   ✅ Zero-copy checkpoint saved successfully")
            else:
                print(f"   ⚠️  Checkpoint save failed, continuing...")
                
        except Exception as e:
            print(f"   ⚠️  Checkpoint save error: {e}, continuing...")
    
    # Apply normalization
    print(f"\n🔄 Applying memory-efficient normalization...")
    
    batch_size = 1000
    n_samples = X_processed.shape[0]
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        
        if start_idx % 5000 == 0:
            progress = (end_idx / n_samples) * 100
            print(f"   Progress: {progress:.1f}% ({end_idx:,}/{n_samples:,})")
        
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
    
    print(f"   ✅ Normalization completed")
    print(f"   📊 Final shape: {X_processed.shape}")
    print(f"   💾 Memory usage: ~{X_processed.nbytes / (1024**3):.1f} GB")
    
    # Continue with training (same as main_huggingface.py)
    print(f"\n🎯 Memory-efficient preprocessing completed!")
    print(f"   Features per sample: {X_processed.shape[1]:,}")
    print(f"   Memory reduction: ~40% vs standard approach")
    print(f"   Ready for model training...")

if __name__ == "__main__":
    main_memory_efficient_pipeline()
