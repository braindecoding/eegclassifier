#!/usr/bin/env python3
"""
Upload EEG preprocessing checkpoints to Hugging Face Hub
"""

from huggingface_hub import HfApi
import os

def get_folder_size(folder_path):
    """Calculate total size of folder in bytes"""
    total_size = 0
    if not os.path.exists(folder_path):
        return 0

    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(filepath)
            except OSError:
                continue
    return total_size

def upload_checkpoints():
    """Upload checkpoints to Hugging Face Hub with error handling"""

    print("🚀 EEG Preprocessing Checkpoints Upload")
    print("=" * 50)

    # Check if checkpoints folder exists
    checkpoint_folder = "./checkpoints"
    if not os.path.exists(checkpoint_folder):
        print(f"❌ Checkpoints folder not found: {checkpoint_folder}")
        return False

    # Check folder size
    folder_size = get_folder_size(checkpoint_folder)
    print(f"📁 Uploading folder: {checkpoint_folder}")
    print(f"📊 Folder size: {folder_size / (1024**3):.1f} GB")

    # List files to upload
    files = []
    for root, dirs, filenames in os.walk(checkpoint_folder):
        for filename in filenames:
            files.append(filename)

    print(f"📄 Files to upload: {len(files)}")
    for file in files:
        file_path = os.path.join(checkpoint_folder, file)
        file_size = os.path.getsize(file_path) / (1024**3)
        print(f"   - {file} ({file_size:.1f} GB)")

    # Check for HF token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ HF_TOKEN environment variable not set")
        print("   Please set your Hugging Face token:")
        print("   export HF_TOKEN=your_token_here")
        return False

    print(f"🎯 Target repo: braindecoding/BANDWiseEMDHHT")
    print(f"📦 Repo type: dataset")

    try:
        print("\n🔄 Starting upload...")
        api = HfApi(token=hf_token)

        api.upload_folder(
            folder_path=checkpoint_folder,
            repo_id="braindecoding/BANDWiseEMDHHT",
            repo_type="dataset",
        )

        print("✅ Upload completed successfully!")
        print("🌐 Checkpoints available at: https://huggingface.co/datasets/braindecoding/BANDWiseEMDHHT")
        return True

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print("💡 Common issues:")
        print("   - Check your internet connection")
        print("   - Verify HF_TOKEN is correct")
        print("   - Ensure you have write access to the repo")
        return False

if __name__ == "__main__":
    upload_checkpoints()
