#!/usr/bin/env python3
"""
BrainDigiCNN: EEG Digit Classification with PyTorch
Ported from TensorFlow version for better CUDA compatibility
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from scipy import signal
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import os
import pickle
import time
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
warnings.filterwarnings('ignore')

def setup_device():
    """
    Setup PyTorch device (CUDA/CPU) with comprehensive testing
    """
    print("\n=== PyTorch Device Configuration ===")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Test CUDA operations
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.mm(x, x)
            print("✅ CUDA operations test passed")
            
            # Test Conv1D operation
            conv = nn.Conv1d(1, 32, 3).cuda()
            test_input = torch.randn(1, 1, 100).cuda()
            output = conv(test_input)
            print("✅ CUDA Conv1D test passed")
            
            return device
            
        except Exception as e:
            print(f"❌ CUDA test failed: {e}")
            print("🔄 Falling back to CPU")
            return torch.device('cpu')
    else:
        print("⚠️  CUDA not available, using CPU")
        return torch.device('cpu')

class BrainDigiCNN(nn.Module):
    """
    PyTorch implementation of BrainDigiCNN for EEG digit classification
    """
    
    def __init__(self, input_size, num_classes=10):
        super(BrainDigiCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(256)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.3)
        
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.3)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.dropout3 = nn.Dropout(0.3)
        
        # Calculate flattened size
        self.flatten_size = self._get_flatten_size(input_size)
        
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout5 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)
        
    def _get_flatten_size(self, input_size):
        """Calculate the size after convolution and pooling layers"""
        x = torch.randn(1, 1, input_size)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        return x.numel()
    
    def forward(self, x):
        # Ensure input has correct shape [batch_size, 1, sequence_length]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # Conv layers
        x = self.dropout1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense layers
        x = self.dropout4(F.relu(self.fc1(x)))
        x = self.dropout5(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

class EEGDataset(Dataset):
    """
    Custom Dataset for EEG data
    """
    
    def __init__(self, X, y, transform=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label

def train_model(model, train_loader, val_loader, device, epochs=20, lr=0.001):
    """
    Train the PyTorch model
    """
    print(f"\n🚀 Starting training on {device}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'   Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'   Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'   Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'   Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 60)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc
    }

def evaluate_model(model, test_loader, device):
    """
    Evaluate the trained model
    """
    print("\n📊 Evaluating model...")
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    
    print(f"📈 Test Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_predictions,
        'targets': all_targets
    }

# Import existing classes from TensorFlow version
from main import MindBigDataLoader, EEGSignalProcessor, CheckpointManager, OptimizedPreprocessor

def main_pipeline_pytorch(file_path, use_checkpoint=True, clear_checkpoints=False):
    """
    PyTorch version of the main pipeline
    """
    print("=== BrainDigiCNN: EEG Digit Classification with PyTorch ===\n")

    # Setup device
    device = setup_device()

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager()

    if clear_checkpoints:
        checkpoint_manager.clear_checkpoints()

    if use_checkpoint:
        checkpoint_manager.get_checkpoint_info()

    # 1. Load and organize data (reuse from TensorFlow version)
    print("\n1. Loading MindBigData...")

    # Initialize loader
    loader = MindBigDataLoader(file_path)

    # Check for raw data checkpoint
    if use_checkpoint and checkpoint_manager.checkpoint_exists('raw_data'):
        print("   📁 Loading from checkpoint...")
        data = checkpoint_manager.load_checkpoint('raw_data')
        loader.data = data
    else:
        print("   📥 Loading from file...")
        data = loader.load_data(device_filter="EP", code_filter=list(range(10)))

        if data is None or len(data) == 0:
            print("No data loaded! Please check file path and format.")
            return

        if use_checkpoint:
            checkpoint_manager.save_checkpoint('raw_data', data)

    # Get data info
    loader.get_data_info()

    # 2. Organize data by trials
    print("\n2. Organizing data by trials...")

    if use_checkpoint and checkpoint_manager.checkpoint_exists('organized_data'):
        print("   📁 Loading organized data from checkpoint...")
        checkpoint_data = checkpoint_manager.load_checkpoint('organized_data')
        X_raw, y = checkpoint_data['X_raw'], checkpoint_data['y']
    else:
        print("   🔄 Organizing trials...")
        X_raw, y = loader.organize_by_trials()

        if X_raw is None or len(X_raw) == 0:
            print("No organized data available!")
            return

        if use_checkpoint:
            checkpoint_data = {'X_raw': X_raw, 'y': y}
            checkpoint_manager.save_checkpoint('organized_data', checkpoint_data)

    print(f"   Organized data shape: {X_raw.shape}")
    print(f"   Labels shape: {y.shape}")
    print(f"   Unique labels: {sorted(set(y))}")

    # 3. Preprocessing (reuse optimized version)
    print("\n3. Preprocessing EEG signals...")

    if use_checkpoint and checkpoint_manager.checkpoint_exists('preprocessed_data'):
        print("   📁 Loading preprocessed data from checkpoint...")
        X_processed = checkpoint_manager.load_checkpoint('preprocessed_data')
    else:
        print("   🚀 Starting optimized preprocessing...")

        optimized_processor = OptimizedPreprocessor(
            sampling_rate=128,
            n_processes=None,
            batch_size=64
        )

        X_processed = optimized_processor.process_optimized(X_raw)

        if use_checkpoint:
            checkpoint_manager.save_checkpoint('preprocessed_data', X_processed)

    print(f"   ✅ Processed data shape: {X_processed.shape}")

    # 4. Prepare data for PyTorch
    print("\n4. Preparing data for PyTorch...")

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_processed, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"   Train set: {X_train.shape}")
    print(f"   Validation set: {X_val.shape}")
    print(f"   Test set: {X_test.shape}")

    # Create datasets and dataloaders
    batch_size = 32
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    test_dataset = EEGDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 5. Create and train model
    print("\n5. Creating PyTorch model...")

    input_size = X_processed.shape[1]
    model = BrainDigiCNN(input_size=input_size, num_classes=10)
    model = model.to(device)

    print(f"   Model created with input size: {input_size}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 6. Train model
    print("\n6. Training model...")

    training_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=20,
        lr=0.001
    )

    # 7. Evaluate model
    print("\n7. Evaluating model...")

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))

    test_results = evaluate_model(model, test_loader, device)

    # 8. Plot results
    plot_training_history(training_history)
    plot_confusion_matrix(test_results['targets'], test_results['predictions'])

    print(f"\n🎉 Training completed!")
    print(f"   Best validation accuracy: {training_history['best_val_acc']:.2f}%")
    print(f"   Test accuracy: {test_results['accuracy']:.4f}")

    return model, test_results

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['val_losses'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(history['train_accuracies'], label='Train Accuracy')
    ax2.plot(history['val_accuracies'], label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history_pytorch.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix - PyTorch BrainDigiCNN')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix_pytorch.png', dpi=300, bbox_inches='tight')
    plt.show()

# Main execution
if __name__ == "__main__":
    file_path = "EP1.01.txt"

    print("=== BrainDigiCNN: PyTorch Version ===")
    print(f"Using dataset: {file_path}")

    # Checkpoint options
    use_checkpoint = True
    clear_checkpoints = False

    if os.path.exists(file_path):
        print(f"Dataset file found: {file_path}")

        if use_checkpoint:
            print("📁 Checkpoint system enabled")
        else:
            print("⚠️  Checkpoint system disabled")

        print("Starting PyTorch training pipeline...")
        model, results = main_pipeline_pytorch(file_path, use_checkpoint, clear_checkpoints)
    else:
        print(f"\nFile {file_path} not found!")
        print("Please make sure EP1.01.txt is in the same directory as this script.")

    print("\n=== PyTorch Version Benefits ===")
    print("✅ Better CUDA compatibility")
    print("✅ More intuitive debugging")
    print("✅ Dynamic computation graph")
    print("✅ Easier model customization")
    print("✅ Better memory management")
