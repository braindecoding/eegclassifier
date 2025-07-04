#!/usr/bin/env python3
"""
Chunked checkpoint saver for large datasets
Saves data in smaller chunks to avoid memory spikes
"""

import numpy as np
import pickle
import os
from main import CheckpointManager

class ChunkedCheckpointSaver:
    """
    Save large datasets in chunks to avoid memory issues
    """
    
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_chunked_data(self, data, stage_name, chunk_size=5000):
        """
        Save large numpy array in chunks
        
        Args:
            data: numpy array to save
            stage_name: name for the checkpoint
            chunk_size: number of samples per chunk
        """
        print(f"💾 Saving large dataset in chunks...")
        print(f"   Data shape: {data.shape}")
        print(f"   Data size: {data.nbytes / (1024**3):.1f} GB")
        print(f"   Chunk size: {chunk_size} samples")
        
        n_samples = data.shape[0]
        n_chunks = (n_samples + chunk_size - 1) // chunk_size
        
        print(f"   Total chunks: {n_chunks}")
        
        # Save metadata
        metadata = {
            'original_shape': data.shape,
            'dtype': str(data.dtype),
            'n_chunks': n_chunks,
            'chunk_size': chunk_size,
            'stage_name': stage_name
        }
        
        metadata_file = os.path.join(self.checkpoint_dir, f"{stage_name}_metadata.pkl")
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"   ✅ Metadata saved: {metadata_file}")
        
        # Save chunks
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, n_samples)
            
            chunk_data = data[start_idx:end_idx]
            chunk_file = os.path.join(self.checkpoint_dir, f"{stage_name}_chunk_{i:04d}.npz")
            
            try:
                np.savez_compressed(chunk_file, data=chunk_data)
                chunk_size_mb = os.path.getsize(chunk_file) / (1024**2)
                print(f"   ✅ Chunk {i+1}/{n_chunks} saved: {chunk_size_mb:.1f} MB")
                
                # Clear chunk from memory
                del chunk_data
                
            except Exception as e:
                print(f"   ❌ Failed to save chunk {i}: {e}")
                return False
        
        print(f"   🎉 All chunks saved successfully!")
        return True
    
    def load_chunked_data(self, stage_name):
        """
        Load data from chunks
        
        Args:
            stage_name: name of the checkpoint
            
        Returns:
            numpy array or None if failed
        """
        print(f"📥 Loading chunked dataset: {stage_name}")
        
        # Load metadata
        metadata_file = os.path.join(self.checkpoint_dir, f"{stage_name}_metadata.pkl")
        if not os.path.exists(metadata_file):
            print(f"   ❌ Metadata file not found: {metadata_file}")
            return None
        
        try:
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            print(f"   📊 Original shape: {metadata['original_shape']}")
            print(f"   📊 Data type: {metadata['dtype']}")
            print(f"   📊 Number of chunks: {metadata['n_chunks']}")
            
        except Exception as e:
            print(f"   ❌ Failed to load metadata: {e}")
            return None
        
        # Load chunks
        chunks = []
        for i in range(metadata['n_chunks']):
            chunk_file = os.path.join(self.checkpoint_dir, f"{stage_name}_chunk_{i:04d}.npz")
            
            if not os.path.exists(chunk_file):
                print(f"   ❌ Chunk file not found: {chunk_file}")
                return None
            
            try:
                chunk_data = np.load(chunk_file)['data']
                chunks.append(chunk_data)
                print(f"   ✅ Chunk {i+1}/{metadata['n_chunks']} loaded")
                
            except Exception as e:
                print(f"   ❌ Failed to load chunk {i}: {e}")
                return None
        
        # Concatenate chunks
        try:
            full_data = np.concatenate(chunks, axis=0)
            print(f"   🎉 Data reconstructed: {full_data.shape}")
            
            # Verify shape
            if full_data.shape == tuple(metadata['original_shape']):
                print(f"   ✅ Shape verification passed")
                return full_data
            else:
                print(f"   ❌ Shape mismatch: expected {metadata['original_shape']}, got {full_data.shape}")
                return None
                
        except Exception as e:
            print(f"   ❌ Failed to concatenate chunks: {e}")
            return None
    
    def chunked_checkpoint_exists(self, stage_name):
        """
        Check if chunked checkpoint exists
        """
        metadata_file = os.path.join(self.checkpoint_dir, f"{stage_name}_metadata.pkl")
        return os.path.exists(metadata_file)
    
    def list_chunked_checkpoints(self):
        """
        List available chunked checkpoints
        """
        checkpoints = []
        if not os.path.exists(self.checkpoint_dir):
            return checkpoints
        
        for file in os.listdir(self.checkpoint_dir):
            if file.endswith('_metadata.pkl'):
                stage_name = file.replace('_metadata.pkl', '')
                
                # Calculate total size
                total_size = 0
                i = 0
                while True:
                    chunk_file = os.path.join(self.checkpoint_dir, f"{stage_name}_chunk_{i:04d}.npz")
                    if os.path.exists(chunk_file):
                        total_size += os.path.getsize(chunk_file)
                        i += 1
                    else:
                        break
                
                checkpoints.append({
                    'stage': stage_name,
                    'chunks': i,
                    'size_mb': total_size / (1024**2)
                })
        
        return checkpoints

def test_chunked_saver():
    """
    Test the chunked saver with dummy data
    """
    print("🧪 Testing chunked checkpoint saver...")
    
    # Create dummy data
    dummy_data = np.random.random((1000, 100)).astype(np.float32)
    print(f"Created dummy data: {dummy_data.shape}")
    
    # Save in chunks
    saver = ChunkedCheckpointSaver()
    success = saver.save_chunked_data(dummy_data, 'test_data', chunk_size=250)
    
    if success:
        # Load back
        loaded_data = saver.load_chunked_data('test_data')
        
        if loaded_data is not None:
            # Verify
            if np.array_equal(dummy_data, loaded_data):
                print("✅ Test passed: Data integrity verified")
            else:
                print("❌ Test failed: Data mismatch")
        else:
            print("❌ Test failed: Could not load data")
    else:
        print("❌ Test failed: Could not save data")

if __name__ == "__main__":
    test_chunked_saver()
