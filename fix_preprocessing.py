#!/usr/bin/env python3
"""
Script untuk menghapus checkpoint preprocessed_data yang salah
dan memaksa menggunakan HHT preprocessing yang benar
"""

import os
from main import CheckpointManager

def fix_preprocessing():
    """
    Hapus checkpoint preprocessed_data yang menggunakan preprocessing salah
    """
    print("🔧 Fixing preprocessing pipeline...")
    
    checkpoint_manager = CheckpointManager()
    
    # Check existing checkpoints
    checkpoints = checkpoint_manager.list_checkpoints()
    print("📁 Current checkpoints:")
    for cp in checkpoints:
        print(f"   {cp['stage']}: {cp['size_mb']} MB")
    
    # Remove preprocessed_data checkpoint to force HHT preprocessing
    preprocessed_file = os.path.join(checkpoint_manager.checkpoint_dir, 'preprocessed_data.pkl')
    
    if os.path.exists(preprocessed_file):
        print(f"\n🗑️  Removing incorrect preprocessed_data checkpoint...")
        os.remove(preprocessed_file)
        print(f"✅ Removed: {preprocessed_file}")
        print("   Next run will use proper HHT preprocessing from paper!")
    else:
        print(f"\n✅ No preprocessed_data checkpoint found.")
    
    print(f"\n📋 Summary:")
    print(f"   ✅ Raw data checkpoint: KEPT (reusable)")
    print(f"   ✅ Organized data checkpoint: KEPT (reusable)")
    print(f"   🗑️  Preprocessed data checkpoint: REMOVED (will regenerate with HHT)")
    
    print(f"\n🧠 Next run will use:")
    print(f"   1. Lowpass filter (45 Hz)")
    print(f"   2. Notch filter (50 Hz)")
    print(f"   3. Frequency band decomposition (Delta, Theta, Alpha, Beta, Gamma)")
    print(f"   4. EMD (Empirical Mode Decomposition)")
    print(f"   5. HHT (Hilbert-Huang Transform)")
    print(f"   6. Feature concatenation")
    
    print(f"\n🎯 This should significantly improve accuracy towards 98%!")

if __name__ == "__main__":
    fix_preprocessing()
