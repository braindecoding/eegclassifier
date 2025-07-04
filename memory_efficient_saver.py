#!/usr/bin/env python3
"""
Memory-efficient checkpoint saver using memory-mapped files
Zero-copy saving to prevent memory spikes
"""

import numpy as np
import os
import json
import gc
from pathlib import Path

class MemoryEfficientSaver:
    """
    Save large arrays without memory spikes using memory-mapped files
    """
    
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def save_zero_copy(self, data, stage_name):
        """
        Save data using zero-copy memory-mapped approach
        
        Args:
            data: numpy array to save
            stage_name: name for the checkpoint
        """
        print(f"💾 Zero-copy saving: {stage_name}")
        print(f"   Data shape: {data.shape}")
        print(f"   Data size: {data.nbytes / (1024**3):.1f} GB")
        print(f"   Memory usage before save: {self._get_memory_usage():.1f} GB")
        
        try:
            # Save metadata
            metadata = {
                'shape': data.shape,
                'dtype': str(data.dtype),
                'stage_name': stage_name
            }
            
            metadata_file = self.checkpoint_dir / f"{stage_name}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
            
            # Save data using memory-mapped file (zero-copy)
            data_file = self.checkpoint_dir / f"{stage_name}_data.npy"
            
            # Create memory-mapped file
            mmap_array = np.memmap(
                data_file, 
                dtype=data.dtype, 
                mode='w+', 
                shape=data.shape
            )
            
            # Copy data in chunks to avoid memory spike
            chunk_size = 1000  # Process 1000 samples at a time
            n_samples = data.shape[0]
            
            for i in range(0, n_samples, chunk_size):
                end_idx = min(i + chunk_size, n_samples)
                mmap_array[i:end_idx] = data[i:end_idx]
                
                # Force write to disk and clear cache
                mmap_array.flush()
                
                if i % 10000 == 0:
                    progress = (end_idx / n_samples) * 100
                    print(f"   💾 Saving progress: {progress:.1f}% ({end_idx:,}/{n_samples:,})")
                    print(f"   Memory usage: {self._get_memory_usage():.1f} GB")
            
            # Final flush and cleanup
            mmap_array.flush()
            del mmap_array  # Release memory map
            gc.collect()    # Force garbage collection
            
            print(f"   ✅ Zero-copy save completed")
            print(f"   Memory usage after save: {self._get_memory_usage():.1f} GB")
            print(f"   File size: {data_file.stat().st_size / (1024**3):.1f} GB")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Zero-copy save failed: {e}")
            return False
    
    def load_zero_copy(self, stage_name):
        """
        Load data using memory-mapped approach (lazy loading)
        
        Args:
            stage_name: name of the checkpoint
            
        Returns:
            memory-mapped array or None if failed
        """
        print(f"📥 Zero-copy loading: {stage_name}")
        
        try:
            # Load metadata
            metadata_file = self.checkpoint_dir / f"{stage_name}_metadata.json"
            if not metadata_file.exists():
                print(f"   ❌ Metadata not found: {metadata_file}")
                return None
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            print(f"   📊 Shape: {metadata['shape']}")
            print(f"   📊 Dtype: {metadata['dtype']}")
            
            # Load data using memory-mapped file (lazy loading)
            data_file = self.checkpoint_dir / f"{stage_name}_data.npy"
            if not data_file.exists():
                print(f"   ❌ Data file not found: {data_file}")
                return None
            
            # Create memory-mapped array (doesn't load into RAM)
            mmap_array = np.memmap(
                data_file,
                dtype=metadata['dtype'],
                mode='r',
                shape=tuple(metadata['shape'])
            )
            
            print(f"   ✅ Zero-copy loading completed")
            print(f"   📊 Array shape: {mmap_array.shape}")
            print(f"   💾 Memory usage: {self._get_memory_usage():.1f} GB (no RAM increase)")
            
            return mmap_array
            
        except Exception as e:
            print(f"   ❌ Zero-copy load failed: {e}")
            return None
    
    def checkpoint_exists(self, stage_name):
        """
        Check if zero-copy checkpoint exists
        """
        metadata_file = self.checkpoint_dir / f"{stage_name}_metadata.json"
        data_file = self.checkpoint_dir / f"{stage_name}_data.npy"
        return metadata_file.exists() and data_file.exists()
    
    def _get_memory_usage(self):
        """
        Get current memory usage in GB
        """
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024**3)
        except ImportError:
            return 0.0
    
    def save_compressed_chunks(self, data, stage_name, chunk_size=5000):
        """
        Alternative: Save as compressed chunks with immediate cleanup
        """
        print(f"💾 Compressed chunk saving: {stage_name}")
        print(f"   Data shape: {data.shape}")
        print(f"   Chunk size: {chunk_size}")
        
        try:
            n_samples = data.shape[0]
            n_chunks = (n_samples + chunk_size - 1) // chunk_size
            
            # Save metadata
            metadata = {
                'original_shape': data.shape,
                'dtype': str(data.dtype),
                'n_chunks': n_chunks,
                'chunk_size': chunk_size
            }
            
            metadata_file = self.checkpoint_dir / f"{stage_name}_chunks_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
            
            # Save chunks with immediate cleanup
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, n_samples)
                
                # Extract chunk
                chunk = data[start_idx:end_idx].copy()
                
                # Save compressed
                chunk_file = self.checkpoint_dir / f"{stage_name}_chunk_{i:04d}.npz"
                np.savez_compressed(chunk_file, data=chunk)
                
                # Immediate cleanup
                del chunk
                gc.collect()
                
                if i % 5 == 0:
                    progress = ((i + 1) / n_chunks) * 100
                    print(f"   💾 Chunk progress: {progress:.1f}% ({i+1}/{n_chunks})")
                    print(f"   Memory usage: {self._get_memory_usage():.1f} GB")
            
            print(f"   ✅ Compressed chunk saving completed")
            return True
            
        except Exception as e:
            print(f"   ❌ Compressed chunk save failed: {e}")
            return False

def test_memory_efficient_saver():
    """
    Test the memory-efficient saver
    """
    print("🧪 Testing memory-efficient saver...")
    
    # Create test data
    test_data = np.random.random((1000, 100)).astype(np.float32)
    print(f"Created test data: {test_data.shape}")
    
    saver = MemoryEfficientSaver()
    
    # Test zero-copy save
    success = saver.save_zero_copy(test_data, 'test_zero_copy')
    if success:
        # Test zero-copy load
        loaded_data = saver.load_zero_copy('test_zero_copy')
        if loaded_data is not None:
            print(f"✅ Zero-copy test passed")
        else:
            print(f"❌ Zero-copy load failed")
    else:
        print(f"❌ Zero-copy save failed")

if __name__ == "__main__":
    test_memory_efficient_saver()
