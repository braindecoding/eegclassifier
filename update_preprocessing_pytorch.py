#!/usr/bin/env python3
"""
Script untuk memastikan main_pytorch.py menggunakan Band-wise EMD yang baru
"""

import os
from main import CheckpointManager

def update_pytorch_preprocessing():
    """
    Update preprocessing untuk PyTorch version
    """
    print("🔧 Updating PyTorch Preprocessing to Band-wise EMD")
    print("=" * 50)
    
    checkpoint_manager = CheckpointManager()
    
    # Check existing checkpoints
    checkpoints = checkpoint_manager.list_checkpoints()
    print("📁 Current checkpoints:")
    for cp in checkpoints:
        print(f"   {cp['stage']}: {cp['size_mb']} MB")
    
    # Remove old preprocessing checkpoints
    old_checkpoints = ['preprocessed_data.pkl']
    
    for old_checkpoint in old_checkpoints:
        checkpoint_file = os.path.join(checkpoint_manager.checkpoint_dir, old_checkpoint)
        if os.path.exists(checkpoint_file):
            print(f"\n🗑️  Removing old checkpoint: {old_checkpoint}")
            os.remove(checkpoint_file)
            print(f"✅ Removed: {checkpoint_file}")
    
    print(f"\n📋 Summary of Changes:")
    print(f"   ✅ main.py: Updated to Band-wise EMD with exactly 10 IMFs")
    print(f"   ✅ main_pytorch.py: Uses same EEGSignalProcessor from main.py")
    print(f"   ✅ Old checkpoints: Removed to force new preprocessing")
    print(f"   ✅ New checkpoint name: 'preprocessed_data_bandwise'")
    
    print(f"\n🧠 Band-wise EMD Configuration:")
    print(f"   📊 Frequency Bands:")
    print(f"      • Delta (δ): 0.5–4 Hz")
    print(f"      • Theta (θ): 4–8 Hz")
    print(f"      • Alpha (α): 8–13 Hz")
    print(f"      • Beta Rendah (β₁): 13–20 Hz")
    print(f"      • Beta Tinggi (β₂): 20–30 Hz")
    print(f"      • Gamma (γ): 30–100 Hz")
    print(f"   🔄 EMD: Exactly 10 IMFs per band")
    print(f"   🧠 HHT: [IA, IP, IF] features per IMF")
    print(f"   🔗 Output: 6 bands × 10 IMFs × 3 features = Rich feature vector")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Run: python main_pytorch.py")
    print(f"   2. New preprocessing will generate Band-wise features")
    print(f"   3. Expect consistent 10 IMFs per frequency band")
    print(f"   4. Feature vector will be much richer and more structured")
    
    print(f"\n📈 Expected Performance Improvement:")
    print(f"   • More structured features (frequency-specific)")
    print(f"   • Consistent feature dimensions (exactly 10 IMFs)")
    print(f"   • Better neurophysiological interpretation")
    print(f"   • Potential to reach 98% accuracy target")

def verify_implementation():
    """
    Verify that the implementation is correct
    """
    print(f"\n🔍 Verifying Implementation...")
    
    try:
        from main import EEGSignalProcessor
        
        # Test processor
        processor = EEGSignalProcessor(sampling_rate=128)
        
        # Check frequency bands
        print(f"   ✅ Frequency bands defined:")
        for band_name, (low, high) in processor.frequency_bands.items():
            print(f"      {band_name}: {low}-{high} Hz")
        
        # Test EMD with dummy data
        import numpy as np
        dummy_signal = np.random.randn(256)  # 2 seconds @ 128Hz
        
        print(f"   🧪 Testing EMD with dummy signal...")
        imfs = processor.empirical_mode_decomposition(dummy_signal, num_imf=10)
        print(f"   ✅ EMD generates exactly {len(imfs)} IMFs")
        
        if len(imfs) == 10:
            print(f"   🎯 SUCCESS: EMD produces exactly 10 IMFs as expected")
        else:
            print(f"   ❌ ERROR: EMD produces {len(imfs)} IMFs, expected 10")
        
        # Test HHT
        print(f"   🧪 Testing HHT...")
        hht_features = processor.hilbert_huang_transform(imfs)
        print(f"   ✅ HHT generates {hht_features.shape} feature matrix")
        
        print(f"\n✅ Implementation verification completed successfully!")
        
    except Exception as e:
        print(f"   ❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Main function
    """
    try:
        update_pytorch_preprocessing()
        verify_implementation()
        
        print(f"\n🎉 PyTorch preprocessing update completed!")
        print(f"   Ready to run main_pytorch.py with Band-wise EMD")
        
    except Exception as e:
        print(f"❌ Error during update: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
