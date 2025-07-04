#!/usr/bin/env python3
"""
BrainDigiCNN: EEG Digit Classification with PyTorch
Ported from TensorFlow version for better CUDA compatibility
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()  # Enable interactive mode for real-time plotting
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
            print(f"⚠️  CUDA test failed, using CPU: {e}")
            return torch.device('cpu')
    else:
        print("⚠️  CUDA not available, using CPU")
        return torch.device('cpu')

class BrainDigiCNN(nn.Module):
    """
    PyTorch implementation of BrainDigiCNN for EEG digit classification
    Exact architecture from paper for 98% accuracy
    """

    def __init__(self, input_size, num_classes=10):
        super(BrainDigiCNN, self).__init__()

        # Layer 1: Conv1D + BN + MaxPooling (256 filters, kernel=7)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(256)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # Layer 2: Conv1D + BN + MaxPooling (128 filters, kernel=7)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Layer 3: Conv1D + BN + MaxPooling (64 filters, kernel=7)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # Layer 4: Conv1D + BN + MaxPooling (32 filters, kernel=7) - Missing in previous implementation!
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=7, padding=3)
        self.bn4 = nn.BatchNorm1d(32)
        self.pool4 = nn.MaxPool1d(kernel_size=2)

        # Calculate flattened size
        self.flatten_size = self._get_flatten_size(input_size)

        # Fully Connected Layers (exact from paper)
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, num_classes)
        
    def _get_flatten_size(self, input_size):
        """Calculate the size after convolution and pooling layers"""
        x = torch.randn(1, 1, input_size)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        return x.numel()

    def forward(self, x):
        # Ensure input has correct shape [batch_size, 1, sequence_length]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Conv layers (exact from paper)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # Dense layers (exact from paper)
        x = self.dropout1(F.relu(self.fc1(x)))  # Dense 128 + ReLU
        x = self.dropout2(F.relu(self.fc2(x)))  # Dense 64 + ReLU
        x = self.fc3(x)                         # Dense 10
        x = F.softmax(x, dim=1)                 # Softmax (as per Table 4)

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
    Train the PyTorch model with advanced techniques for 98% accuracy
    """
    print(f"\n🚀 Starting training on {device}")
    print(f"   Target accuracy: 98% (as reported in paper)")

    criterion = nn.CrossEntropyLoss()  # Categorical Crossentropy as per Table 5
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam optimizer as per Table 5

    # Simple scheduler for paper compliance
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
        scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'   🎯 New best validation accuracy: {val_acc:.2f}%')

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'   Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'   Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'   Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'   Best Val Acc: {best_val_acc:.2f}%')
        print('-' * 60)

        # Plot training curves every 5 epochs
        if (epoch + 1) % 5 == 0:
            plot_training_curves_realtime(train_losses, val_losses, train_accuracies, val_accuracies, epoch + 1)
    
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

    # 3. Band-wise EMD-HHT Preprocessing
    print("\n3. Band-wise EMD-HHT Preprocessing...")
    print("   🧠 Using Band-wise EMD with exactly 10 IMFs per frequency band")
    print("   📊 Frequency bands: Delta, Theta, Alpha, Beta Low, Beta High, Gamma")

    if use_checkpoint and checkpoint_manager.checkpoint_exists('preprocessed_data_bandwise'):
        print("   📁 Loading band-wise preprocessed data from checkpoint...")
        X_processed = checkpoint_manager.load_checkpoint('preprocessed_data_bandwise')
    else:
        print("   🚀 Starting Band-wise EMD-HHT preprocessing...")

        optimized_processor = OptimizedPreprocessor(
            sampling_rate=128,
            n_processes=None,
            batch_size=64
        )

        X_processed = optimized_processor.process_optimized(X_raw)

        if use_checkpoint:
            checkpoint_manager.save_checkpoint('preprocessed_data_bandwise', X_processed)

    print(f"   ✅ Processed data shape: {X_processed.shape}")

    # 3.5. Apply memory-efficient robust normalization
    print("\n3.5. Applying memory-efficient robust normalization...")

    if use_checkpoint and checkpoint_manager.checkpoint_exists('normalized_data_bandwise'):
        print("   📁 Loading normalized data from checkpoint...")
        X_processed = checkpoint_manager.load_checkpoint('normalized_data_bandwise')
        print(f"   ✅ Normalized data loaded from checkpoint")
        print(f"   Normalized data stats: min={X_processed.min():.6f}, max={X_processed.max():.6f}, mean={X_processed.mean():.6f}, std={X_processed.std():.6f}")
    else:
        print(f"   Original data stats: min={X_processed.min():.6f}, max={X_processed.max():.6f}, mean={X_processed.mean():.6f}, std={X_processed.std():.6f}")
        print(f"   Data size: {X_processed.nbytes / (1024**3):.1f} GB")

        # Memory-efficient batch processing
        print("   Processing samples in batches to save memory...")

        batch_size = 1000  # Process 1000 samples at a time
        n_samples = X_processed.shape[0]

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)

            print(f"   Progress: {end_idx}/{n_samples} ({end_idx/n_samples*100:.1f}%)")

            # Process batch in-place
            for i in range(start_idx, end_idx):
                # Get sample reference (no copy)
                sample = X_processed[i]
                sample_mean = sample.mean()
                sample_std = sample.std()

                # In-place normalization
                if sample_std > 1e-8:
                    sample -= sample_mean  # Subtract mean in-place
                    sample /= sample_std   # Divide by std in-place
                else:
                    sample -= sample_mean  # Only subtract mean if std is too small

        print(f"   ✅ Normalization completed")
        print(f"   Normalized data stats: min={X_processed.min():.6f}, max={X_processed.max():.6f}, mean={X_processed.mean():.6f}, std={X_processed.std():.6f}")

        # Save normalized data checkpoint
        if use_checkpoint:
            checkpoint_manager.save_checkpoint('normalized_data_bandwise', X_processed)

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

    # Create datasets and dataloaders with paper-compliant batch size
    batch_size = 32  # Batch size as per Table 5
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    test_dataset = EEGDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 5. Create and train model
    print("\n5. Creating PyTorch model...")

    input_size = X_processed.shape[1]
    model = BrainDigiCNN(input_size=input_size, num_classes=10)
    model = model.to(device)

    print(f"   Model created with input size: {input_size}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 6. Train model with paper-optimized parameters
    print("\n6. Training model with paper-optimized parameters...")

    training_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=20,  # Epochs as per Table 5 (10-20 range)
        lr=0.001    # Learning rate as per Table 5
    )

    # 7. Evaluate model
    print("\n7. Evaluating model...")

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))

    test_results = evaluate_model(model, test_loader, device)

    # 8. Comprehensive visualization and analysis
    print("\n8. Generating comprehensive visualizations...")

    # Plot comprehensive training history
    plot_training_history(training_history)

    # Plot confusion matrix
    plot_confusion_matrix(test_results['targets'], test_results['predictions'])

    # Plot accuracy comparison with paper
    plot_accuracy_comparison()

    # Save final training progress
    save_training_progress(training_history, len(training_history['train_losses']))

    # Performance analysis
    best_val_acc = training_history['best_val_acc']
    target_acc = 98.0
    gap_to_target = target_acc - best_val_acc

    print(f"\n🎉 Training completed!")
    print(f"   Best validation accuracy: {best_val_acc:.2f}%")
    print(f"   Test accuracy: {test_results['accuracy']:.4f}")
    print(f"   Gap to paper target (98%): {gap_to_target:.2f}%")

    if best_val_acc >= 98.0:
        print("🏆 CONGRATULATIONS! Achieved paper-level accuracy!")
    elif best_val_acc >= 95.0:
        print("🎯 Excellent performance! Very close to paper results.")
    elif best_val_acc >= 90.0:
        print("👍 Good performance! Consider hyperparameter tuning.")
    else:
        print("📈 Room for improvement. Check data preprocessing and model architecture.")

    return model, test_results

def plot_training_curves_realtime(train_losses, val_losses, train_accs, val_accs, current_epoch):
    """Plot real-time training curves during training"""
    plt.figure(figsize=(15, 5))

    # Loss plot
    plt.subplot(1, 3, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.title(f'Training and Validation Loss (Epoch {current_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    plt.title(f'Training and Validation Accuracy (Epoch {current_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Progress towards 98% target
    plt.subplot(1, 3, 3)
    target_acc = 98.0
    best_val_acc = max(val_accs) if val_accs else 0
    progress = (best_val_acc / target_acc) * 100

    plt.bar(['Current Best', 'Target (Paper)'], [best_val_acc, target_acc],
            color=['blue' if best_val_acc < target_acc else 'green', 'red'], alpha=0.7)
    plt.title(f'Progress to Paper Accuracy\n{progress:.1f}% of target achieved')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)

    # Add text annotations
    plt.text(0, best_val_acc + 2, f'{best_val_acc:.2f}%', ha='center', fontweight='bold')
    plt.text(1, target_acc + 2, f'{target_acc:.1f}%', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'training_progress_epoch_{current_epoch}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_training_history(history):
    """Plot comprehensive training history"""
    plt.figure(figsize=(20, 12))

    # Loss plot
    plt.subplot(2, 3, 1)
    epochs = range(1, len(history['train_losses']) + 1)
    plt.plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Accuracy plot
    plt.subplot(2, 3, 2)
    plt.plot(epochs, history['train_accuracies'], 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, history['val_accuracies'], 'r-', label='Validation Accuracy', linewidth=2)
    plt.axhline(y=98, color='g', linestyle='--', label='Paper Target (98%)', linewidth=2)
    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Loss difference (overfitting indicator)
    plt.subplot(2, 3, 3)
    loss_diff = [val - train for train, val in zip(history['train_losses'], history['val_losses'])]
    plt.plot(epochs, loss_diff, 'purple', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Validation - Training Loss\n(Overfitting Indicator)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Difference')
    plt.grid(True, alpha=0.3)

    # Accuracy difference
    plt.subplot(2, 3, 4)
    acc_diff = [train - val for train, val in zip(history['train_accuracies'], history['val_accuracies'])]
    plt.plot(epochs, acc_diff, 'orange', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Training - Validation Accuracy\n(Overfitting Indicator)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Difference (%)')
    plt.grid(True, alpha=0.3)

    # Final performance summary
    plt.subplot(2, 3, 5)
    final_metrics = ['Train Acc', 'Val Acc', 'Best Val Acc', 'Paper Target']
    final_values = [
        history['train_accuracies'][-1],
        history['val_accuracies'][-1],
        history['best_val_acc'],
        98.0
    ]
    colors = ['blue', 'red', 'green', 'gold']
    bars = plt.bar(final_metrics, final_values, color=colors, alpha=0.7)
    plt.title('Final Performance Summary', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)

    # Add value labels on bars
    for bar, value in zip(bars, final_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.2f}%', ha='center', fontweight='bold')

    # Training efficiency plot
    plt.subplot(2, 3, 6)
    improvement_rate = []
    for i in range(1, len(history['val_accuracies'])):
        improvement = history['val_accuracies'][i] - history['val_accuracies'][i-1]
        improvement_rate.append(improvement)

    if improvement_rate:
        plt.plot(range(2, len(history['val_accuracies']) + 1), improvement_rate, 'green', linewidth=2)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Validation Accuracy Improvement Rate', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy Improvement (%)')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('comprehensive_training_history_pytorch.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_training_progress(history, epoch, filename='training_progress.pkl'):
    """Save training progress for resuming"""
    progress_data = {
        'history': history,
        'epoch': epoch,
        'timestamp': time.time()
    }

    with open(filename, 'wb') as f:
        pickle.dump(progress_data, f)

    print(f"📁 Training progress saved to {filename}")

def load_training_progress(filename='training_progress.pkl'):
    """Load training progress for resuming"""
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                progress_data = pickle.load(f)
            print(f"📁 Training progress loaded from {filename}")
            return progress_data
        except Exception as e:
            print(f"❌ Failed to load training progress: {e}")
            return None
    return None

def plot_accuracy_comparison():
    """Plot comparison with paper results"""
    plt.figure(figsize=(10, 6))

    # Paper results (hypothetical progression to 98%)
    paper_epochs = list(range(1, 51))
    paper_accuracy = [20 + 78 * (1 - np.exp(-epoch/15)) for epoch in paper_epochs]

    plt.plot(paper_epochs, paper_accuracy, 'g--', linewidth=3, label='Paper Target Progression', alpha=0.7)
    plt.axhline(y=98, color='red', linestyle='-', linewidth=2, label='Paper Final Result (98%)')

    plt.title('Accuracy Comparison: Our Implementation vs Paper', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)

    plt.tight_layout()
    plt.savefig('accuracy_comparison_with_paper.png', dpi=300, bbox_inches='tight')
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
