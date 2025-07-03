#!/usr/bin/env python3
"""
Script untuk membersihkan file sebelum run ulang dengan implementasi baru
"""

import os
import glob
from main import CheckpointManager

def clean_for_rerun():
    """
    Bersihkan semua file yang perlu dihapus sebelum run ulang
    """
    print("🧹 Cleaning files for fresh run with new implementation")
    print("=" * 60)
    
    files_removed = []
    
    # 1. Clean checkpoint files
    print("\n1. 🗑️  Cleaning checkpoint files...")
    
    checkpoint_manager = CheckpointManager()
    checkpoint_dir = checkpoint_manager.checkpoint_dir
    
    # List of old checkpoint files to remove
    old_checkpoints = [
        'preprocessed_data.pkl',
        'preprocessed_data_bandwise.pkl'
    ]
    
    for checkpoint_file in old_checkpoints:
        file_path = os.path.join(checkpoint_dir, checkpoint_file)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                files_removed.append(file_path)
                print(f"   ✅ Removed: {file_path}")
            except Exception as e:
                print(f"   ❌ Failed to remove {file_path}: {e}")
        else:
            print(f"   ℹ️  Not found: {file_path}")
    
    # Keep raw_data and organized_data checkpoints
    print(f"   ℹ️  Keeping: raw_data.pkl (reusable)")
    print(f"   ℹ️  Keeping: organized_data.pkl (reusable)")
    
    # 2. Clean model files
    print("\n2. 🗑️  Cleaning model files...")
    
    model_patterns = [
        'best_model.pth',           # PyTorch model
        '*.h5',                     # TensorFlow model files
        'training_progress.pkl',    # Training progress
        'model_*.pkl'               # Any other model files
    ]
    
    for pattern in model_patterns:
        matching_files = glob.glob(pattern)
        for file_path in matching_files:
            try:
                os.remove(file_path)
                files_removed.append(file_path)
                print(f"   ✅ Removed: {file_path}")
            except Exception as e:
                print(f"   ❌ Failed to remove {file_path}: {e}")
    
    if not any(glob.glob(pattern) for pattern in model_patterns):
        print(f"   ℹ️  No model files found")
    
    # 3. Clean plot files
    print("\n3. 🗑️  Cleaning plot files...")
    
    plot_patterns = [
        '*.png',
        '*.jpg',
        '*.jpeg',
        '*.pdf'
    ]
    
    for pattern in plot_patterns:
        matching_files = glob.glob(pattern)
        for file_path in matching_files:
            # Skip important files that shouldn't be deleted
            if file_path.lower() in ['readme.png', 'logo.png', 'architecture.png']:
                print(f"   ℹ️  Skipping: {file_path} (important file)")
                continue
                
            try:
                os.remove(file_path)
                files_removed.append(file_path)
                print(f"   ✅ Removed: {file_path}")
            except Exception as e:
                print(f"   ❌ Failed to remove {file_path}: {e}")
    
    if not any(glob.glob(pattern) for pattern in plot_patterns):
        print(f"   ℹ️  No plot files found")
    
    # 4. Clean temporary files
    print("\n4. 🗑️  Cleaning temporary files...")
    
    temp_patterns = [
        '__pycache__',
        '*.pyc',
        '*.pyo',
        '.pytest_cache',
        '*.log'
    ]
    
    for pattern in temp_patterns:
        if pattern == '__pycache__':
            # Remove __pycache__ directories
            for root, dirs, files in os.walk('.'):
                for dir_name in dirs:
                    if dir_name == '__pycache__':
                        dir_path = os.path.join(root, dir_name)
                        try:
                            import shutil
                            shutil.rmtree(dir_path)
                            files_removed.append(dir_path)
                            print(f"   ✅ Removed directory: {dir_path}")
                        except Exception as e:
                            print(f"   ❌ Failed to remove {dir_path}: {e}")
        else:
            matching_files = glob.glob(pattern)
            for file_path in matching_files:
                try:
                    os.remove(file_path)
                    files_removed.append(file_path)
                    print(f"   ✅ Removed: {file_path}")
                except Exception as e:
                    print(f"   ❌ Failed to remove {file_path}: {e}")
    
    # 5. Summary
    print(f"\n📊 Summary:")
    print(f"   Total files removed: {len(files_removed)}")
    
    if files_removed:
        print(f"   Files removed:")
        for file_path in files_removed:
            print(f"     • {file_path}")
    
    print(f"\n✅ Cleanup completed!")
    
    # 6. What's preserved
    print(f"\n📁 Preserved files:")
    print(f"   ✅ EP1.01.txt (raw dataset)")
    print(f"   ✅ main.py (updated with Band-wise EMD)")
    print(f"   ✅ main_pytorch.py (updated with Table 5 hyperparameters)")
    print(f"   ✅ checkpoints/raw_data.pkl (if exists)")
    print(f"   ✅ checkpoints/organized_data.pkl (if exists)")
    
    # 7. Next steps
    print(f"\n🚀 Ready for fresh run!")
    print(f"   Next steps:")
    print(f"   1. Run: python main.py (TensorFlow version)")
    print(f"      OR")
    print(f"   2. Run: python main_pytorch.py (PyTorch version)")
    print(f"   ")
    print(f"   Both will use:")
    print(f"   • Band-wise EMD with exactly 10 IMFs per frequency band")
    print(f"   • 6 frequency bands: Delta, Theta, Alpha, Beta Low, Beta High, Gamma")
    print(f"   • HHT feature extraction: [IA, IP, IF] per IMF")
    print(f"   • Model architecture exactly matching Table 4")
    print(f"   • Hyperparameters exactly matching Table 5")
    print(f"   • Target: 98% accuracy")

def main():
    """
    Main function
    """
    try:
        clean_for_rerun()
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
