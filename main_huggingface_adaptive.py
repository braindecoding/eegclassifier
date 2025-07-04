#!/usr/bin/env python3
"""
Adaptive EEG Digit Classification using Hugging Face dataset
Smart IMF count (6-10) based on signal complexity
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

class AdaptivePreprocessor:
    """
    Adaptive preprocessor with smart IMF count (6-10) based on signal complexity
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

        print(f"🧠 Adaptive Preprocessor - {mode}:")
        print(f"   🖥️  Total CPU cores: {total_cores}")
        print(f"   🔥 Using processes: {self.n_processes}")
        print(f"   📦 Batch size: {self.batch_size}")
        print(f"   🎯 IMFs per band: 6-10 (adaptive based on signal complexity)")
        print(f"   💡 Memory optimization: Smart IMF allocation")
        
        # Calculate expected features range
        min_features = 6 * 6 * 768  # 6 bands × 6 IMFs × 768 features
        max_features = 6 * 10 * 768  # 6 bands × 10 IMFs × 768 features
        print(f"   📊 Features per sample: {min_features:,} - {max_features:,} (adaptive)")
        
        if total_cores >= 80:
            estimated_time = 52022 / (self.n_processes * 2.5) / 60
            print(f"   ⚡ Estimated processing time: ~{estimated_time:.1f} minutes")

    def process_single_sample_adaptive(self, args):
        """
        Process single sample with adaptive IMF count (6-10)
        """
        sample_idx, eeg_sample, sampling_rate = args
        
        try:
            # Initialize processor
            processor = EEGSignalProcessor(sampling_rate=sampling_rate)
            
            # Process each channel with adaptive EMD
            processed_channels = []
            total_imfs_used = 0
            
            for ch in range(eeg_sample.shape[0]):
                try:
                    # Use adaptive EMD (6-10 IMFs based on signal complexity)
                    features = processor.process_eeg_signal(eeg_sample[ch])
                    processed_channels.append(features)
                    
                    # Track IMF usage for monitoring
                    if hasattr(processor, '_last_imf_count'):
                        total_imfs_used += processor._last_imf_count
                        
                except Exception as e:
                    # Fallback: use raw signal
                    processed_channels.append(eeg_sample[ch])
            
            # Ensure consistent length
            min_length = min(len(ch) for ch in processed_channels)
            processed_channels = [ch[:min_length] for ch in processed_channels]
            
            processed_sample = np.concatenate(processed_channels)
            
            # Store IMF usage info for monitoring
            avg_imfs = total_imfs_used / eeg_sample.shape[0] if eeg_sample.shape[0] > 0 else 6
            
            return sample_idx, processed_sample, avg_imfs
            
        except Exception as e:
            print(f"   ❌ Error processing sample {sample_idx}: {e}")
            return sample_idx, eeg_sample.flatten(), 6

    def process_optimized_adaptive(self, X_raw):
        """
        Process with adaptive multiprocessing
        """
        print(f"   🚀 Starting adaptive Band-wise EMD-HHT preprocessing...")
        print(f"   📊 Input: {X_raw.shape}")
        print(f"   🧠 Adaptive strategy: 6-10 IMFs based on signal complexity")
        
        from multiprocessing import Pool
        import time
        
        # Prepare arguments
        args_list = [(i, sample, self.sampling_rate) for i, sample in enumerate(X_raw)]
        start_time = time.time()
        
        # Track IMF usage statistics
        imf_usage_stats = []
        
        # Process with multiprocessing
        with Pool(processes=self.n_processes) as pool:
            results = []
            chunk_size = max(1, len(args_list) // 50)
            
            for i in range(0, len(args_list), chunk_size):
                chunk = args_list[i:i + chunk_size]
                chunk_results = pool.map(self.process_single_sample_adaptive, chunk)
                results.extend(chunk_results)
                
                # Collect IMF usage stats
                chunk_imfs = [result[2] for result in chunk_results if len(result) > 2]
                imf_usage_stats.extend(chunk_imfs)
                
                # Enhanced progress tracking with IMF stats
                progress = min(100, (i + len(chunk)) * 100 // len(args_list))
                elapsed = time.time() - start_time
                samples_processed = i + len(chunk)
                
                if samples_processed > 0:
                    speed = samples_processed / elapsed
                    remaining_samples = len(args_list) - samples_processed
                    eta_seconds = remaining_samples / speed if speed > 0 else 0
                    eta_minutes = eta_seconds / 60
                    
                    # Calculate average IMF usage
                    if imf_usage_stats:
                        avg_imfs = np.mean(imf_usage_stats)
                        imf_distribution = {
                            '6 IMFs': sum(1 for x in imf_usage_stats if 5.5 <= x < 6.5),
                            '7 IMFs': sum(1 for x in imf_usage_stats if 6.5 <= x < 7.5),
                            '8 IMFs': sum(1 for x in imf_usage_stats if 7.5 <= x < 8.5),
                            '9 IMFs': sum(1 for x in imf_usage_stats if 8.5 <= x < 9.5),
                            '10 IMFs': sum(1 for x in imf_usage_stats if x >= 9.5)
                        }
                        
                        print(f"   🧠 Adaptive Progress: {progress:3.1f}% ({samples_processed:,}/{len(args_list):,}) | "
                              f"Speed: {speed:.1f} samples/s | "
                              f"Elapsed: {elapsed/60:.1f}m | "
                              f"ETA: {eta_minutes:.1f}m | "
                              f"Avg IMFs: {avg_imfs:.1f}")
                        
                        if samples_processed % 10000 == 0:
                            print(f"      📊 IMF Distribution: {imf_distribution}")
                    else:
                        print(f"   🧠 Adaptive Progress: {progress:3.1f}% ({samples_processed:,}/{len(args_list):,}) | "
                              f"Speed: {speed:.1f} samples/s | "
                              f"Elapsed: {elapsed/60:.1f}m | "
                              f"ETA: {eta_minutes:.1f}m")
        
        # Sort results and extract processed data
        results.sort(key=lambda x: x[0])
        X_processed = np.array([result[1] for result in results])
        
        elapsed_total = time.time() - start_time
        
        # Final IMF usage statistics
        if imf_usage_stats:
            avg_imfs = np.mean(imf_usage_stats)
            min_imfs = np.min(imf_usage_stats)
            max_imfs = np.max(imf_usage_stats)
            
            imf_counts = {
                '6 IMFs': sum(1 for x in imf_usage_stats if 5.5 <= x < 6.5),
                '7 IMFs': sum(1 for x in imf_usage_stats if 6.5 <= x < 7.5),
                '8 IMFs': sum(1 for x in imf_usage_stats if 7.5 <= x < 8.5),
                '9 IMFs': sum(1 for x in imf_usage_stats if 8.5 <= x < 9.5),
                '10 IMFs': sum(1 for x in imf_usage_stats if x >= 9.5)
            }
            
            print(f"   ✅ Adaptive processing completed in {elapsed_total:.1f}s")
            print(f"   📊 Output shape: {X_processed.shape}")
            print(f"   🧠 IMF Usage Statistics:")
            print(f"      Average IMFs per channel: {avg_imfs:.2f}")
            print(f"      IMF range: {min_imfs:.1f} - {max_imfs:.1f}")
            print(f"      Distribution: {imf_counts}")
            print(f"   💾 Memory usage: ~{X_processed.nbytes / (1024**3):.1f} GB")
            
            # Calculate memory efficiency
            if avg_imfs < 10:
                memory_savings = (10 - avg_imfs) / 10 * 100
                print(f"   💡 Memory savings vs fixed 10 IMFs: ~{memory_savings:.1f}%")
        else:
            print(f"   ✅ Processing completed in {elapsed_total:.1f}s")
            print(f"   📊 Output shape: {X_processed.shape}")
        
        return X_processed

def main_adaptive_pipeline():
    """
    Main pipeline with adaptive IMF processing
    """
    print("🚀 Adaptive EEG Digit Classification - Hugging Face Dataset")
    print("=" * 70)
    print("🧠 Smart IMF count (6-10) based on signal complexity")
    print("🎯 Using standardized dataset for apple-to-apple comparison!")
    
    # Load Hugging Face dataset
    print("📥 Loading MindBigData2022_MNIST_EP from Hugging Face...")
    
    try:
        ds = load_dataset("DavidVivancos/MindBigData2022_MNIST_EP")
        print(f"   ✅ Dataset loaded successfully")
        print(f"   Available splits: {list(ds.keys())}")
        
        split_name = list(ds.keys())[0]
        data_split = ds[split_name]
        print(f"   Using split: {split_name} ({len(data_split)} samples)")
        
        # Extract data
        from main_huggingface import extract_split_data
        X_raw, y = extract_split_data(data_split)
        
        print(f"   ✅ Data extracted: {X_raw.shape}, labels: {y.shape}")
        
    except Exception as e:
        print(f"   ❌ Failed to load dataset: {e}")
        return
    
    # Adaptive preprocessing
    print("\n🧠 Adaptive Preprocessing...")
    
    checkpoint_manager = CheckpointManager()
    
    if checkpoint_manager.checkpoint_exists('hf_adaptive_data'):
        print("   📁 Loading adaptive data from checkpoint...")
        X_processed = checkpoint_manager.load_checkpoint('hf_adaptive_data')
        print(f"   ✅ Loaded: {X_processed.shape}")
    else:
        # Process with adaptive approach
        processor = AdaptivePreprocessor()
        X_processed = processor.process_optimized_adaptive(X_raw)
        
        # Save with zero-copy method
        print(f"\n💾 Saving adaptive checkpoint...")
        try:
            from memory_efficient_saver import MemoryEfficientSaver
            saver = MemoryEfficientSaver(checkpoint_manager.checkpoint_dir)
            success = saver.save_zero_copy(X_processed, 'hf_adaptive_data')
            
            if success:
                print(f"   ✅ Zero-copy checkpoint saved successfully")
            else:
                print(f"   ⚠️  Checkpoint save failed, continuing...")
                
        except Exception as e:
            print(f"   ⚠️  Checkpoint save error: {e}, continuing...")
    
    print(f"\n🎯 Adaptive preprocessing completed!")
    print(f"   Features per sample: {X_processed.shape[1]:,}")
    print(f"   Adaptive IMF strategy: Optimized memory vs quality balance")
    print(f"   Ready for model training...")

if __name__ == "__main__":
    main_adaptive_pipeline()
