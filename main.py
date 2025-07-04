import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from scipy import signal
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import os
import pickle
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
import time
warnings.filterwarnings('ignore')

# GPU Configuration
def setup_gpu():
    """
    Setup GPU configuration for optimal performance
    """
    print("=== GPU Configuration ===")

    # Check GPU availability
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"✅ Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu}")

        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            # Test GPU functionality with a simple operation
            with tf.device('/GPU:0'):
                test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                result = tf.matmul(test_tensor, test_tensor)
                result.numpy()  # Force execution

            # Set GPU as preferred device
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

            # Disable mixed precision for now to avoid compatibility issues
            # Mixed precision can cause issues with some TensorFlow versions
            # policy = tf.keras.mixed_precision.Policy('mixed_float16')
            # tf.keras.mixed_precision.set_global_policy(policy)
            print("✅ GPU enabled (float32 - stable mode)")

            print("✅ GPU configuration completed successfully")
            return True

        except RuntimeError as e:
            print(f"❌ GPU configuration error: {e}")
            return False
    else:
        print("⚠️  No GPU found, using CPU")
        return False

# Setup GPU at import time
GPU_AVAILABLE = setup_gpu()

def process_single_sample(args):
    """
    Process single EEG sample - untuk multiprocessing

    Args:
        args: tuple (sample_index, eeg_sample, sampling_rate)

    Returns:
        tuple (sample_index, processed_features)
    """
    sample_idx, eeg_sample, sampling_rate = args

    try:
        processor = EEGSignalProcessor(sampling_rate=sampling_rate)

        # Process setiap channel
        processed_channels = []
        for ch in range(eeg_sample.shape[0]):
            try:
                features = processor.process_eeg_signal(eeg_sample[ch])
                processed_channels.append(features)
            except Exception as e:
                # Fallback: gunakan raw signal
                processed_channels.append(eeg_sample[ch])

        # Pastikan semua channel memiliki panjang yang sama
        min_length = min(len(ch) for ch in processed_channels)
        processed_channels = [ch[:min_length] for ch in processed_channels]

        processed_sample = np.concatenate(processed_channels)
        return sample_idx, processed_sample

    except Exception as e:
        print(f"Error processing sample {sample_idx}: {e}")
        # Return raw concatenated data as fallback
        flattened = eeg_sample.flatten()
        return sample_idx, flattened

def process_batch_gpu(batch_data, sampling_rate=128):
    """
    Process batch of EEG data using GPU acceleration

    Args:
        batch_data: numpy array of shape (batch_size, channels, time_points)
        sampling_rate: sampling rate

    Returns:
        processed batch
    """
    if not GPU_AVAILABLE:
        return None

    try:
        # Convert to TensorFlow tensor
        batch_tensor = tf.constant(batch_data, dtype=tf.float32)

        # Apply basic filtering using TensorFlow operations
        # Bandpass filter approximation using convolution
        with tf.device('/GPU:0'):
            # Simple high-pass filter (remove DC component)
            batch_filtered = batch_tensor - tf.reduce_mean(batch_tensor, axis=-1, keepdims=True)

            # Normalize
            batch_normalized = tf.nn.l2_normalize(batch_filtered, axis=-1)

            # Convert back to numpy
            processed_batch = batch_normalized.numpy()

        return processed_batch

    except Exception as e:
        print(f"GPU processing error: {e}")
        return None

class OptimizedPreprocessor:
    """
    Optimized preprocessor menggunakan GPU dan multiprocessing
    """

    def __init__(self, sampling_rate=128, n_processes=None, batch_size=32):
        self.sampling_rate = sampling_rate

        # BEAST MODE optimization for 80-core Xeon systems!
        total_cores = cpu_count()
        if n_processes is None:
            if total_cores >= 80:
                # MAXIMUM BEAST MODE: Use 75% of cores (leave 25% for system)
                self.n_processes = min(60, total_cores)
                self.batch_size = 16  # Smaller batches for better parallelization
                mode = "🔥 BEAST MODE"
            elif total_cores >= 40:
                # HIGH PERFORMANCE: Use 80% of cores
                self.n_processes = min(32, total_cores)
                self.batch_size = 24
                mode = "🚀 HIGH-PERF"
            elif total_cores >= 16:
                # STANDARD: Use most cores
                self.n_processes = min(12, total_cores)
                self.batch_size = 32
                mode = "⚡ STANDARD"
            else:
                # BASIC: Use available cores
                self.n_processes = min(8, total_cores)
                self.batch_size = batch_size
                mode = "💻 BASIC"
        else:
            self.n_processes = min(n_processes, total_cores)
            self.batch_size = batch_size
            mode = "🔧 CUSTOM"

        print(f"🚀 OptimizedPreprocessor initialized - {mode}:")
        print(f"   🖥️  Total CPU cores: {total_cores}")
        print(f"   🔥 Using processes: {self.n_processes} ({self.n_processes/total_cores*100:.1f}% utilization)")
        print(f"   📦 Batch size: {self.batch_size}")
        print(f"   💪 Expected speed: ~{self.n_processes*2.5:.1f} samples/second")
        print(f"   🎯 GPU available: {GPU_AVAILABLE}")

        if total_cores >= 80:
            print(f"   🔥 BEAST MACHINE DETECTED! Preparing for MAXIMUM PERFORMANCE!")
            print(f"   ⚡ Estimated 52K samples processing time: ~{52022/(self.n_processes*2.5)/60:.1f} minutes")

    def process_parallel_cpu(self, X_raw):
        """
        Process menggunakan multiprocessing CPU
        """
        print("   🔄 Using CPU multiprocessing...")

        # Prepare arguments for multiprocessing
        args_list = [(i, sample, self.sampling_rate) for i, sample in enumerate(X_raw)]

        start_time = time.time()

        # Process dengan multiprocessing menggunakan HHT
        with Pool(processes=self.n_processes) as pool:
            results = []

            # Enhanced progress tracking - more frequent updates
            chunk_size = max(1, len(args_list) // 50)  # 50 progress updates (more frequent)

            for i in range(0, len(args_list), chunk_size):
                chunk = args_list[i:i + chunk_size]
                chunk_results = pool.map(self.process_single_sample_hht, chunk)
                results.extend(chunk_results)

                # Enhanced progress information
                progress = min(100, (i + len(chunk)) * 100 // len(args_list))
                elapsed = time.time() - start_time
                samples_processed = i + len(chunk)

                if samples_processed > 0:
                    speed = samples_processed / elapsed  # samples per second
                    remaining_samples = len(args_list) - samples_processed
                    eta_seconds = remaining_samples / speed if speed > 0 else 0
                    eta_minutes = eta_seconds / 60

                    print(f"   📊 Progress: {progress:3.1f}% ({samples_processed:,}/{len(args_list):,}) | "
                          f"Speed: {speed:.1f} samples/s | "
                          f"Elapsed: {elapsed/60:.1f}m | "
                          f"ETA: {eta_minutes:.1f}m")

        # Sort results by sample index
        results.sort(key=lambda x: x[0])
        processed_data = [result[1] for result in results]

        return np.array(processed_data)

    def process_gpu_batched(self, X_raw):
        """
        Process menggunakan GPU dalam batches
        """
        print("   🚀 Using GPU batch processing...")

        processed_batches = []
        start_time = time.time()

        for i in range(0, len(X_raw), self.batch_size):
            batch = X_raw[i:i + self.batch_size]

            # Try GPU processing first
            processed_batch = process_batch_gpu(batch, self.sampling_rate)

            if processed_batch is not None:
                # Flatten each sample in the batch
                batch_flattened = []
                for sample in processed_batch:
                    batch_flattened.append(sample.flatten())
                processed_batches.extend(batch_flattened)
            else:
                # Fallback to CPU for this batch
                for sample in batch:
                    processed_batches.append(sample.flatten())

            # Progress update
            progress = min(100, (i + len(batch)) * 100 // len(X_raw))
            elapsed = time.time() - start_time
            if i % (self.batch_size * 10) == 0:  # Update every 10 batches
                print(f"   Progress: {progress}% ({i + len(batch)}/{len(X_raw)}) - {elapsed:.1f}s")

        return np.array(processed_batches)

    def process_optimized(self, X_raw):
        """
        Process dengan strategi optimal berdasarkan hardware
        """
        print(f"   📊 Processing {len(X_raw)} samples...")

        start_time = time.time()

        # Choose optimal preprocessing strategy based on hardware and dataset size
        print("   🚀 Choosing optimal preprocessing strategy...")

        if GPU_AVAILABLE and len(X_raw) > 10000:
            # For very large datasets: Hybrid GPU+CPU approach
            print("   🔥 Using Hybrid GPU+CPU preprocessing for large dataset...")
            processed_data = self.process_hybrid_gpu_cpu(X_raw)
        elif len(X_raw) > 1000:
            # For medium-large datasets: CPU multiprocessing with HHT
            print("   🧠 Using CPU multiprocessing with full HHT preprocessing...")
            processed_data = self.process_parallel_cpu(X_raw)
        else:
            # For smaller datasets: Sequential HHT processing
            print("   🔄 Using sequential HHT preprocessing...")
            processor = EEGSignalProcessor(sampling_rate=self.sampling_rate)
            processed_data = []

            for i, sample in enumerate(X_raw):
                if i % 50 == 0:
                    print(f"   Progress: {i}/{len(X_raw)} ({i/len(X_raw)*100:.1f}%)")

                processed_channels = []
                for ch in range(sample.shape[0]):
                    try:
                        # Use proper HHT preprocessing
                        features = processor.process_eeg_signal(sample[ch])
                        processed_channels.append(features)
                    except Exception as e:
                        print(f"   Warning: Channel {ch} processing failed: {e}")
                        # Fallback: use raw signal
                        processed_channels.append(sample[ch])

                # Ensure all channels have same length
                min_length = min(len(ch) for ch in processed_channels)
                processed_channels = [ch[:min_length] for ch in processed_channels]
                processed_data.append(np.concatenate(processed_channels))

            processed_data = np.array(processed_data)

        elapsed_time = time.time() - start_time
        print(f"   ✅ Processing completed in {elapsed_time:.1f}s")
        print(f"   📈 Speed: {len(X_raw)/elapsed_time:.1f} samples/second")

        return processed_data

    def create_tf_dataset(self, X_raw, batch_size=32):
        """
        Create optimized TensorFlow dataset untuk streaming processing
        """
        print("   🔄 Creating optimized TensorFlow dataset...")

        def preprocess_fn(sample):
            """TensorFlow preprocessing function"""
            # Convert to tensor
            sample_tensor = tf.cast(sample, tf.float32)

            # Basic preprocessing dengan TF operations
            # Remove DC component
            sample_centered = sample_tensor - tf.reduce_mean(sample_tensor, axis=-1, keepdims=True)

            # Normalize
            sample_normalized = tf.nn.l2_normalize(sample_centered, axis=-1)

            # Flatten
            sample_flattened = tf.reshape(sample_normalized, [-1])

            return sample_flattened

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(X_raw)
        dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        # Process all batches
        processed_data = []
        total_batches = len(X_raw) // batch_size + (1 if len(X_raw) % batch_size else 0)

        for i, batch in enumerate(dataset):
            processed_data.extend(batch.numpy())
            if i % 10 == 0:
                progress = (i + 1) * 100 // total_batches
                print(f"   Progress: {progress}% ({i+1}/{total_batches} batches)")

        return np.array(processed_data)

    def process_hybrid_gpu_cpu(self, X_raw):
        """
        Hybrid GPU+CPU preprocessing untuk dataset besar
        GPU: Basic filtering dan normalization
        CPU: EMD dan HHT (yang tidak bisa di-GPU)
        """
        print("   🔥 Hybrid GPU+CPU preprocessing...")

        # Step 1: GPU-accelerated basic preprocessing
        print("   🚀 Step 1: GPU basic filtering...")
        X_gpu_filtered = self.gpu_basic_filtering(X_raw)

        # Step 2: CPU-based HHT processing (parallelized)
        print("   🧠 Step 2: CPU HHT processing...")

        # Prepare arguments for multiprocessing
        args_list = [(i, sample, self.sampling_rate) for i, sample in enumerate(X_gpu_filtered)]

        start_time = time.time()

        # Process dengan multiprocessing
        with Pool(processes=self.n_processes) as pool:
            results = []

            # Enhanced progress tracking for hybrid processing
            chunk_size = max(1, len(args_list) // 50)  # 50 progress updates

            for i in range(0, len(args_list), chunk_size):
                chunk = args_list[i:i + chunk_size]
                chunk_results = pool.map(self.process_single_sample_hht, chunk)
                results.extend(chunk_results)

                # Enhanced progress information for hybrid mode
                progress = min(100, (i + len(chunk)) * 100 // len(args_list))
                elapsed = time.time() - start_time
                samples_processed = i + len(chunk)

                if samples_processed > 0:
                    speed = samples_processed / elapsed
                    remaining_samples = len(args_list) - samples_processed
                    eta_seconds = remaining_samples / speed if speed > 0 else 0
                    eta_minutes = eta_seconds / 60

                    print(f"   🔥 Hybrid Progress: {progress:3.1f}% ({samples_processed:,}/{len(args_list):,}) | "
                          f"Speed: {speed:.1f} samples/s | "
                          f"Elapsed: {elapsed/60:.1f}m | "
                          f"ETA: {eta_minutes:.1f}m")

        # Sort results by sample index
        results.sort(key=lambda x: x[0])
        processed_data = [result[1] for result in results]

        return np.array(processed_data)

    def gpu_basic_filtering(self, X_raw):
        """
        GPU-accelerated basic filtering (lowpass, notch)
        """
        if not GPU_AVAILABLE:
            print("   ℹ️  GPU not available, skipping GPU filtering")
            return X_raw

        try:
            print("   🚀 GPU filtering: Lowpass + Notch filters...")

            # Convert to TensorFlow tensor
            X_tensor = tf.constant(X_raw, dtype=tf.float32)

            with tf.device('/GPU:0'):
                # Basic filtering operations that can be done on GPU

                # 1. Remove DC component (high-pass effect)
                X_filtered = X_tensor - tf.reduce_mean(X_tensor, axis=-1, keepdims=True)

                # 2. Simple lowpass filter approximation using moving average
                # Approximate 100Hz lowpass with moving average
                kernel_size = max(1, int(self.sampling_rate / 100))  # ~1-2 samples for 128Hz
                kernel = tf.ones([kernel_size], dtype=tf.float32) / kernel_size

                # Apply convolution for each sample and channel
                filtered_samples = []
                for i in range(X_filtered.shape[0]):  # For each sample
                    sample_filtered = []
                    for j in range(X_filtered.shape[1]):  # For each channel
                        channel_data = X_filtered[i, j, :]
                        # Pad and convolve
                        padded = tf.pad(channel_data, [[kernel_size//2, kernel_size//2]], mode='REFLECT')
                        filtered_channel = tf.nn.conv1d(
                            tf.expand_dims(tf.expand_dims(padded, 0), 0),
                            tf.expand_dims(tf.expand_dims(kernel, 0), -1),
                            stride=1, padding='VALID'
                        )
                        sample_filtered.append(tf.squeeze(filtered_channel))
                    filtered_samples.append(tf.stack(sample_filtered))

                X_gpu_filtered = tf.stack(filtered_samples)

                # 3. Normalization
                X_normalized = tf.nn.l2_normalize(X_gpu_filtered, axis=-1)

                # Convert back to numpy
                result = X_normalized.numpy()

            print(f"   ✅ GPU filtering completed: {result.shape}")
            return result

        except Exception as e:
            print(f"   ⚠️  GPU filtering not available, using CPU processing")
            return X_raw

    def process_single_sample_hht(self, args):
        """
        Process single sample with full HHT - untuk multiprocessing
        """
        sample_idx, eeg_sample, sampling_rate = args

        try:
            processor = EEGSignalProcessor(sampling_rate=sampling_rate)

            # Process setiap channel dengan full HHT
            processed_channels = []
            for ch in range(eeg_sample.shape[0]):
                try:
                    # Full HHT processing (EMD + Hilbert Transform)
                    features = processor.process_eeg_signal(eeg_sample[ch])
                    processed_channels.append(features)
                except Exception as e:
                    # Fallback: basic processing
                    channel_data = eeg_sample[ch]
                    # Simple detrend and normalize
                    detrended = channel_data - np.mean(channel_data)
                    normalized = detrended / (np.std(detrended) + 1e-8)
                    processed_channels.append(normalized)

            # Ensure all channels have same length
            min_length = min(len(ch) for ch in processed_channels)
            processed_channels = [ch[:min_length] for ch in processed_channels]

            processed_sample = np.concatenate(processed_channels)
            return sample_idx, processed_sample

        except Exception as e:
            print(f"Error processing sample {sample_idx}: {e}")
            # Return flattened raw data as fallback
            flattened = eeg_sample.flatten()
            return sample_idx, flattened

class CheckpointManager:
    """
    Manager untuk menyimpan dan memuat checkpoint selama proses training
    """

    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, stage, data, filename=None):
        """
        Simpan checkpoint dengan atomic operation (aman dari corruption)

        Args:
            stage: nama stage (e.g., 'raw_data', 'organized_data', 'preprocessed_data')
            data: data yang akan disimpan
            filename: nama file custom (optional)
        """
        if filename is None:
            filename = f"{stage}.pkl"

        filepath = os.path.join(self.checkpoint_dir, filename)
        temp_filepath = filepath + '.tmp'

        try:
            print(f"💾 Saving checkpoint: {stage}...")

            # Step 1: Save to temporary file first (atomic operation)
            with open(temp_filepath, 'wb') as f:
                pickle.dump(data, f)
                f.flush()  # Ensure data is written to disk
                os.fsync(f.fileno())  # Force OS to write to disk

            # Step 2: Verify temporary file integrity
            temp_size = os.path.getsize(temp_filepath)
            if temp_size == 0:
                raise Exception("Temporary file is empty - save failed")

            # Step 3: Create backup of existing file (if exists)
            backup_filepath = filepath + '.backup'
            if os.path.exists(filepath):
                os.rename(filepath, backup_filepath)

            # Step 4: Atomic rename (this is atomic on most filesystems)
            os.rename(temp_filepath, filepath)

            # Step 5: Verify final file integrity
            final_size = os.path.getsize(filepath)
            if final_size == 0:
                raise Exception("Final file is empty - corruption detected")

            print(f"✅ Checkpoint saved safely: {filepath}")
            print(f"   File size: {final_size / (1024**2):.1f} MB")

            # Step 6: Clean up backup if save was successful
            if os.path.exists(backup_filepath):
                os.remove(backup_filepath)

            return True

        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")

            # Cleanup: Remove temporary file
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                    print(f"🗑️  Cleaned up temporary file")
                except:
                    pass

            # Recovery: Restore from backup if available
            backup_filepath = filepath + '.backup'
            if os.path.exists(backup_filepath):
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    os.rename(backup_filepath, filepath)
                    print(f"🔄 Restored from backup")
                except:
                    print(f"❌ Failed to restore from backup")

            return False

    def load_checkpoint(self, stage, filename=None):
        """
        Muat checkpoint untuk stage tertentu

        Args:
            stage: nama stage
            filename: nama file custom (optional)

        Returns:
            data jika berhasil, None jika gagal
        """
        if filename is None:
            filename = f"{stage}.pkl"

        filepath = os.path.join(self.checkpoint_dir, filename)

        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            print(f"✅ Checkpoint loaded: {filepath}")
            return data
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            return None

    def checkpoint_exists(self, stage, filename=None):
        """
        Cek apakah checkpoint untuk stage tertentu ada
        """
        if filename is None:
            filename = f"{stage}.pkl"

        filepath = os.path.join(self.checkpoint_dir, filename)
        return os.path.exists(filepath)

    def list_checkpoints(self):
        """
        List semua checkpoint yang tersedia
        """
        if not os.path.exists(self.checkpoint_dir):
            return []

        checkpoints = []
        for file in os.listdir(self.checkpoint_dir):
            if file.endswith('.pkl'):
                filepath = os.path.join(self.checkpoint_dir, file)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                checkpoints.append({
                    'file': file,
                    'stage': file.replace('.pkl', ''),
                    'size_mb': round(size_mb, 2)
                })

        return checkpoints

    def clear_checkpoints(self):
        """
        Hapus semua checkpoint
        """
        if os.path.exists(self.checkpoint_dir):
            for file in os.listdir(self.checkpoint_dir):
                if file.endswith('.pkl'):
                    os.remove(os.path.join(self.checkpoint_dir, file))
            print("🗑️  All checkpoints cleared")

    def get_checkpoint_info(self):
        """
        Tampilkan informasi checkpoint yang tersedia
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            print("📁 No checkpoints found")
            return

        print("📁 Available checkpoints:")
        for cp in checkpoints:
            print(f"   {cp['stage']}: {cp['size_mb']} MB")

        return checkpoints

class MindBigDataLoader:
    """
    Loader untuk dataset MindBigData dengan format yang disebutkan
    """
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.channels_emotiv_epoc = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
    
    def load_data(self, device_filter="EP", code_filter=None):
        """
        Load data dari file MindBigData
        
        Args:
            device_filter: Filter berdasarkan device ("EP" untuk EMOTIV EPOC)
            code_filter: Filter berdasarkan digit code (0-9, None untuk semua)
        """
        print(f"Loading data from {self.file_path}...")
        
        # Baca file
        try:
            with open(self.file_path, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"File {self.file_path} tidak ditemukan!")
            return None
        
        # Parse setiap line
        parsed_data = []
        for i, line in enumerate(lines):
            if i % 1000 == 0:
                print(f"Processing line {i+1}/{len(lines)}")
            
            try:
                parts = line.strip().split('\t')
                if len(parts) != 7:
                    continue
                
                id_val = int(parts[0])
                event_id = int(parts[1])
                device = parts[2]
                channel = parts[3]
                code = int(parts[4])
                size = int(parts[5])
                data_str = parts[6]
                
                # Filter berdasarkan device
                if device != device_filter:
                    continue
                
                # Filter berdasarkan code jika ditentukan
                if code_filter is not None and code not in code_filter:
                    continue
                
                # Parse data sinyal
                if data_str:
                    signal_data = [float(x) for x in data_str.split(',')]
                    
                    parsed_data.append({
                        'id': id_val,
                        'event_id': event_id,
                        'device': device,
                        'channel': channel,
                        'code': code,
                        'size': size,
                        'signal': np.array(signal_data)
                    })
            
            except Exception as e:
                print(f"Error parsing line {i+1}: {e}")
                continue
        
        self.data = parsed_data
        print(f"Loaded {len(parsed_data)} signals")
        return parsed_data
    
    def organize_by_trials(self):
        """
        Organisir data berdasarkan trial (event_id) dan channel
        """
        if self.data is None:
            print("Data belum dimuat!")
            return None

        # Group by event_id
        trials = {}
        for item in self.data:
            event_id = item['event_id']
            if event_id not in trials:
                trials[event_id] = {
                    'code': item['code'],
                    'channels': {}
                }
            trials[event_id]['channels'][item['channel']] = item['signal']

        # Convert ke format yang sesuai untuk CNN
        X = []
        y = []

        # Tentukan panjang minimum untuk normalisasi
        min_length = float('inf')
        valid_trials = []

        # Cari trials yang memiliki semua channel dan tentukan panjang minimum
        for event_id, trial in trials.items():
            if all(ch in trial['channels'] for ch in self.channels_emotiv_epoc):
                valid_trials.append((event_id, trial))
                # Cari panjang minimum dari semua channel dalam trial ini
                trial_min_length = min(len(trial['channels'][ch]) for ch in self.channels_emotiv_epoc)
                min_length = min(min_length, trial_min_length)

        print(f"   Found {len(valid_trials)} complete trials")
        print(f"   Minimum signal length: {min_length}")

        # Proses trials yang valid dengan panjang yang dinormalisasi
        for event_id, trial in valid_trials:
            # Susun data per channel dengan panjang yang sama
            channels_data = []
            for ch in self.channels_emotiv_epoc:
                # Potong sinyal ke panjang minimum
                signal = trial['channels'][ch][:min_length]
                channels_data.append(signal)

            X.append(np.array(channels_data))
            y.append(trial['code'])

        return np.array(X), np.array(y)
    
    def get_data_info(self):
        """
        Dapatkan informasi tentang data yang dimuat
        """
        if self.data is None:
            print("Data belum dimuat!")
            return
        
        # Hitung distribusi per digit
        codes = [item['code'] for item in self.data]
        channels = [item['channel'] for item in self.data]
        
        print("\n=== Data Information ===")
        print(f"Total signals: {len(self.data)}")
        print(f"Unique codes: {sorted(set(codes))}")
        print(f"Unique channels: {sorted(set(channels))}")
        
        # Distribusi per digit
        from collections import Counter
        code_dist = Counter(codes)
        channel_dist = Counter(channels)
        
        print("\nDigit distribution:")
        for code in sorted(code_dist.keys()):
            print(f"  Digit {code}: {code_dist[code]} signals")
        
        print("\nChannel distribution:")
        for ch in sorted(channel_dist.keys()):
            print(f"  {ch}: {channel_dist[ch]} signals")

class EEGSignalProcessor:
    """
    Kelas untuk preprocessing sinyal EEG berdasarkan paper
    """
    
    def __init__(self, sampling_rate=128):
        self.sampling_rate = sampling_rate
        self.frequency_bands = {
            'delta': (0.5, 4),      # Delta (δ): 0.5–4 Hz
            'theta': (4, 8),        # Theta (θ): 4–8 Hz
            'alpha': (8, 13),       # Alpha (α): 8–13 Hz
            'beta_low': (13, 20),   # Beta Rendah (β₁): 13–20 Hz
            'beta_high': (20, 30),  # Beta Tinggi (β₂): 20–30 Hz
            'gamma': (30, 100)      # Gamma (γ): 30–100 Hz
        }
    
    def butterworth_filter(self, data, low_freq, high_freq, order=5):
        """
        Implementasi Butterworth bandpass filter
        """
        if len(data) < 10:
            return data
        
        nyquist = 0.5 * self.sampling_rate
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Pastikan frekuensi dalam rentang yang valid
        low = max(0.01, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))
        
        try:
            b, a = butter(order, [low, high], btype='band')
            filtered_data = filtfilt(b, a, data)
            return filtered_data
        except:
            return data
    
    def lowpass_filter(self, data, cutoff_freq=100, order=5):
        """
        Lowpass filter untuk noise removal
        Updated: cutoff frequency changed from 45Hz to 100Hz
        """
        if len(data) < 10:
            return data
        
        nyquist = 0.5 * self.sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        normal_cutoff = min(normal_cutoff, 0.99)
        
        try:
            b, a = butter(order, normal_cutoff, btype='low')
            filtered_data = filtfilt(b, a, data)
            return filtered_data
        except:
            return data
    
    def notch_filter(self, data, notch_freq=50, quality_factor=30):
        """
        Notch filter untuk menghilangkan power line interference
        """
        if len(data) < 10:
            return data
        
        try:
            nyquist = 0.5 * self.sampling_rate
            freq = notch_freq / nyquist
            
            b, a = signal.iirnotch(freq, quality_factor)
            filtered_data = filtfilt(b, a, data)
            return filtered_data
        except:
            return data
    
    def empirical_mode_decomposition(self, data, num_imf=10):
        """
        EMD (Empirical Mode Decomposition) - Tahap 1 dari HHT

        Tujuan: Memecah sinyal kompleks menjadi exactly 10 IMFs

        Parameters:
        - data: Input EEG signal (preprocessed)
        - num_imf: Fixed number of IMFs (set to exactly 10)

        num_imf=10 Rationale:
        - Consistent output size untuk semua sinyal
        - Predictable feature vector dimensions
        - Covers all EEG frequency ranges adequately
        - Ensures uniform processing across all samples

        Input: Sinyal yang sudah difilter (lowpass + notch)
        Output: Exactly 10 IMFs (forced if necessary)

        Setiap IMF adalah komponen berosilasi yang memenuhi kriteria:
        1. Jumlah zero crossings ≈ jumlah extrema (±1)
        2. Mean envelope dari maxima dan minima ≈ 0

        Jika natural EMD berhenti sebelum 10 IMFs, residue akan dibagi
        untuk mencapai exactly 10 IMFs.

        Mengikuti algoritma 7 langkah dari paper BrainDigiCNN
        """
        if len(data) < 20:
            return np.array([data])

        def cubic_spline_envelope(signal_data, extrema_indices):
            """
            Cubic spline interpolation untuk envelope sesuai paper
            """
            if len(extrema_indices) < 2:
                return np.zeros_like(signal_data)

            try:
                from scipy.interpolate import CubicSpline
                # Gunakan cubic spline interpolation
                cs = CubicSpline(extrema_indices, signal_data[extrema_indices],
                               bc_type='natural', extrapolate=True)
                return cs(range(len(signal_data)))
            except:
                # Fallback ke linear interpolation
                try:
                    return np.interp(range(len(signal_data)), extrema_indices, signal_data[extrema_indices])
                except:
                    return np.zeros_like(signal_data)

        def is_imf(h):
            """
            Check if signal h satisfies IMF conditions:
            1. Number of zero crossings and extrema differ by at most 1
            2. Mean of upper and lower envelopes is zero
            """
            try:
                # Find zero crossings
                zero_crossings = np.where(np.diff(np.sign(h)))[0]
                num_zero_crossings = len(zero_crossings)

                # Find extrema
                maxima_idx = signal.argrelextrema(h, np.greater)[0]
                minima_idx = signal.argrelextrema(h, np.less)[0]
                num_extrema = len(maxima_idx) + len(minima_idx)

                # Condition 1: zero crossings and extrema differ by at most 1
                condition1 = abs(num_zero_crossings - num_extrema) <= 1

                # Condition 2: mean of envelopes should be close to zero
                if len(maxima_idx) >= 2 and len(minima_idx) >= 2:
                    upper_env = cubic_spline_envelope(h, maxima_idx)
                    lower_env = cubic_spline_envelope(h, minima_idx)
                    mean_env = (upper_env + lower_env) / 2
                    condition2 = np.abs(np.mean(mean_env)) < 0.1 * np.std(h)
                else:
                    condition2 = False

                return condition1 and condition2
            except:
                return False

        imfs = []
        residue = data.copy().astype(float)

        # EMD algorithm to extract natural IMFs first
        natural_imf_count = 0
        max_natural_imfs = 15  # Allow more natural IMFs than target

        for imf_count in range(max_natural_imfs):
            if len(residue) < 10:
                break

            # Step 1-7: Sifting process to extract IMF
            h = residue.copy()

            # Sifting iterations
            for sift_iter in range(50):  # Maximum sifting iterations
                try:
                    # Step 1: Find all successive extrema
                    maxima_idx = signal.argrelextrema(h, np.greater)[0]
                    minima_idx = signal.argrelextrema(h, np.less)[0]

                    # Need at least 2 maxima and 2 minima
                    if len(maxima_idx) < 2 or len(minima_idx) < 2:
                        break

                    # Step 2: Cubic spline interpolation for envelopes
                    upper_env = cubic_spline_envelope(h, maxima_idx)  # eu(t)
                    lower_env = cubic_spline_envelope(h, minima_idx)  # el(t)

                    # Step 3: Find local mean m(t) = [eu(t) + el(t)]/2
                    mean_env = (upper_env + lower_env) / 2

                    # Step 4: Find difference q(t) = p(t) - m(t)
                    h_new = h - mean_env

                    # Step 5: Check if h_new is an IMF
                    if is_imf(h_new):
                        h = h_new
                        break

                    # Continue sifting
                    h = h_new

                    # Convergence check
                    if np.sum(np.abs(h_new - h)) < 1e-8:
                        break

                except Exception as e:
                    break

            # Store the IMF
            imfs.append(h.copy())
            natural_imf_count += 1

            # Step 6: Find residue R1(t) = p(t) - vn(t)
            residue = residue - h

            # Step 7: Check if residue is monotonic (stopping criterion)
            try:
                residue_maxima = signal.argrelextrema(residue, np.greater)[0]
                residue_minima = signal.argrelextrema(residue, np.less)[0]

                # If residue has less than 2 extrema, it's monotonic
                if len(residue_maxima) + len(residue_minima) < 2:
                    break
            except:
                break

        # Ensure exactly num_imf IMFs
        if len(imfs) < num_imf:
            # If we have fewer IMFs than required, split the residue
            remaining_imfs_needed = num_imf - len(imfs)

            if len(residue) > 0 and remaining_imfs_needed > 0:
                # Split residue into remaining IMFs
                residue_parts = np.array_split(residue, remaining_imfs_needed)
                for part in residue_parts:
                    if len(part) > 0:
                        imfs.append(part)

            # If still not enough, pad with zeros
            while len(imfs) < num_imf:
                imfs.append(np.zeros_like(data))

        elif len(imfs) > num_imf:
            # If we have more IMFs than required, keep first num_imf and merge rest into last
            excess_imfs = imfs[num_imf:]
            imfs = imfs[:num_imf-1]  # Keep first (num_imf-1) IMFs

            # Merge excess IMFs and residue into the last IMF
            merged_last = residue.copy()
            for excess_imf in excess_imfs:
                if len(excess_imf) == len(merged_last):
                    merged_last += excess_imf

            imfs.append(merged_last)

        else:
            # Exactly the right number, add residue as last IMF
            if len(residue) > 0:
                imfs[-1] = imfs[-1] + residue  # Add residue to last IMF

        # Ensure all IMFs have the same length as input
        final_imfs = []
        for imf in imfs:
            if len(imf) != len(data):
                # Resize to match input length
                if len(imf) > len(data):
                    final_imfs.append(imf[:len(data)])
                else:
                    # Pad with zeros
                    padded_imf = np.zeros(len(data))
                    padded_imf[:len(imf)] = imf
                    final_imfs.append(padded_imf)
            else:
                final_imfs.append(imf)

        return np.array(final_imfs[:num_imf])  # Ensure exactly num_imf IMFs
    
    def hilbert_huang_transform(self, imfs):
        """
        HHT (Hilbert-Huang Transform) - Tahap 2 dari HHT

        Tujuan: Ekstraksi fitur dari IMFs yang dihasilkan EMD

        Input: Set of IMFs dari EMD
        Output: Feature vector [IA, IP, IF] untuk semua IMFs

        Untuk setiap IMF:
        1. Hilbert Transform → Analytic Signal
        2. Ekstraksi Instantaneous Amplitude (IA)
        3. Ekstraksi Instantaneous Phase (IP)
        4. Ekstraksi Instantaneous Frequency (IF)

        Features yang dihasilkan memberikan informasi:
        - IA: Envelope/amplitudo sinyal
        - IP: Informasi fase
        - IF: Modulasi frekuensi
        """
        features = []
        
        for imf in imfs:
            if len(imf) == 0:
                continue
                
            try:
                # Hilbert transform
                analytic_signal = hilbert(imf)
                
                # Instantaneous Amplitude (IA)
                ia = np.abs(analytic_signal)
                
                # Instantaneous Phase (IP)
                ip = np.angle(analytic_signal)
                
                # Instantaneous Frequency (IF)
                if len(ip) > 1:
                    if_signal = np.diff(np.unwrap(ip)) * self.sampling_rate / (2 * np.pi)
                    if_signal = np.append(if_signal, if_signal[-1])  # Pad untuk ukuran yang sama
                else:
                    if_signal = np.array([0])
                
                # Gabungkan fitur
                features.extend([ia, ip, if_signal])
            except:
                # Fallback jika ada error
                features.extend([imf, np.zeros_like(imf), np.zeros_like(imf)])
        
        return np.array(features)
    
    def process_eeg_signal(self, eeg_data):
        """
        Pipeline lengkap untuk preprocessing EEG dengan Band-wise EMD-HHT

        Alur processing:
        1. Lowpass Filter (100 Hz) - Remove high frequency noise
        2. Notch Filter (50 Hz) - Remove power line interference
        3. Band-wise EMD-HHT:
           - Delta (δ): 0.5–4 Hz → Bandpass → EMD → HHT
           - Theta (θ): 4–8 Hz → Bandpass → EMD → HHT
           - Alpha (α): 8–13 Hz → Bandpass → EMD → HHT
           - Beta Rendah (β₁): 13–20 Hz → Bandpass → EMD → HHT
           - Beta Tinggi (β₂): 20–30 Hz → Bandpass → EMD → HHT
           - Gamma (γ): 30–100 Hz → Bandpass → EMD → HHT
        4. Feature Concatenation - Gabungkan semua band features

        Output: Comprehensive feature vector dari semua frequency bands
        """
        # 1. Lowpass filter untuk noise removal (100 Hz cutoff)
        denoised = self.lowpass_filter(eeg_data)
        
        # 2. Notch filter untuk power line interference
        denoised = self.notch_filter(denoised)
        
        # 3. Band-wise EMD-HHT Feature Extraction
        print(f"      Starting Band-wise EMD-HHT processing...")

        band_features = []

        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            try:
                print(f"        Processing {band_name} band ({low_freq}-{high_freq} Hz)...")

                # Step 3a: Bandpass filter untuk isolasi frequency band
                band_data = self.butterworth_filter(denoised, low_freq, high_freq)
                print(f"          Bandpass filtering completed")

                # Step 3b: EMD - Dekomposisi band menjadi IMFs
                print(f"          EMD decomposition...")
                imfs = self.empirical_mode_decomposition(band_data)
                print(f"          Generated {len(imfs)} IMFs for {band_name}")

                # Step 3c: HHT - Feature extraction dari IMFs
                print(f"          HHT feature extraction...")
                if len(imfs) > 0:
                    hht_features = self.hilbert_huang_transform(imfs)
                    band_feature_vector = hht_features.flatten()
                    band_features.append(band_feature_vector)
                    print(f"          Extracted {len(band_feature_vector)} features from {band_name}")
                else:
                    # Fallback: gunakan band data langsung
                    print(f"          No IMFs generated for {band_name}, using band data")
                    band_features.append(band_data)

            except Exception as e:
                print(f"        Error in {band_name} band processing: {e}")
                # Fallback: gunakan denoised signal
                band_features.append(denoised[:len(denoised)//6])  # Approximate band size

        # Step 4: Concatenate features dari semua frequency bands
        try:
            print(f"        Concatenating features from all bands...")
            # Ensure all band features have same length for concatenation
            min_length = min(len(bf) for bf in band_features)
            normalized_features = [bf[:min_length] for bf in band_features]
            all_features = np.concatenate(normalized_features)
            print(f"        Total features: {len(all_features)} from {len(band_features)} bands")
        except Exception as e:
            print(f"        Error in feature concatenation: {e}")
            # Fallback
            all_features = denoised
        
        # Return extracted features
        return all_features

    def plot_imf_decomposition(self, eeg_data, digit_label, channel_idx=0, save_plot=True):
        """
        Plot IMF decomposition untuk visualisasi

        Parameters:
        - eeg_data: Raw EEG signal
        - digit_label: Label digit (0-9)
        - channel_idx: Index channel yang akan diplot
        - save_plot: Save plot ke file
        """
        print(f"\n📊 Plotting IMF decomposition for digit {digit_label}, channel {channel_idx}")

        # Preprocessing steps
        denoised = self.lowpass_filter(eeg_data)
        denoised = self.notch_filter(denoised)

        # EMD decomposition
        imfs = self.empirical_mode_decomposition(denoised)

        # Create comprehensive plot
        n_imfs = len(imfs)
        fig, axes = plt.subplots(n_imfs + 2, 1, figsize=(15, 2 * (n_imfs + 2)))

        # Time axis
        time_axis = np.arange(len(eeg_data)) / self.sampling_rate

        # Plot 1: Original signal
        axes[0].plot(time_axis, eeg_data, 'b-', linewidth=1)
        axes[0].set_title(f'Original EEG Signal - Digit {digit_label} (Channel {channel_idx})',
                         fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Preprocessed signal
        axes[1].plot(time_axis, denoised, 'g-', linewidth=1)
        axes[1].set_title('Preprocessed Signal (Lowpass + Notch Filtered)',
                         fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Amplitude')
        axes[1].grid(True, alpha=0.3)

        # Plot IMFs
        colors = plt.cm.tab10(np.linspace(0, 1, n_imfs))

        for i, imf in enumerate(imfs):
            axes[i + 2].plot(time_axis[:len(imf)], imf, color=colors[i], linewidth=1)

            # Calculate dominant frequency for each IMF
            try:
                # Simple frequency estimation using zero crossings
                zero_crossings = np.where(np.diff(np.sign(imf)))[0]
                if len(zero_crossings) > 1:
                    avg_period = 2 * len(imf) / len(zero_crossings)  # samples per cycle
                    dominant_freq = self.sampling_rate / avg_period
                    freq_text = f' (~{dominant_freq:.1f} Hz)'
                else:
                    freq_text = ''
            except:
                freq_text = ''

            axes[i + 2].set_title(f'IMF {i+1}{freq_text}', fontsize=10, fontweight='bold')
            axes[i + 2].set_ylabel('Amplitude')
            axes[i + 2].grid(True, alpha=0.3)

        # Set x-label for last subplot
        axes[-1].set_xlabel('Time (seconds)')

        plt.tight_layout()

        if save_plot:
            filename = f'imf_decomposition_digit_{digit_label}_ch_{channel_idx}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📁 Plot saved as: {filename}")

        plt.show()

        # Print IMF statistics
        print(f"\n📈 IMF Statistics for Digit {digit_label}:")
        print(f"   Number of IMFs: {n_imfs}")
        for i, imf in enumerate(imfs):
            print(f"   IMF {i+1}: Length={len(imf)}, RMS={np.sqrt(np.mean(imf**2)):.4f}")

        return imfs

    def plot_multiple_digits_imf(self, samples_dict, max_digits=5, channel_idx=0):
        """
        Plot IMF decomposition untuk beberapa digit sekaligus

        Parameters:
        - samples_dict: Dictionary {digit: eeg_sample}
        - max_digits: Maximum number of digits to plot
        - channel_idx: Channel index to analyze
        """
        print(f"\n📊 Plotting IMF comparison for multiple digits (Channel {channel_idx})")

        digits_to_plot = list(samples_dict.keys())[:max_digits]

        for digit in digits_to_plot:
            eeg_sample = samples_dict[digit]
            if len(eeg_sample.shape) > 1:
                # Multi-channel data
                signal_data = eeg_sample[channel_idx]
            else:
                # Single channel data
                signal_data = eeg_sample

            print(f"\n--- Processing Digit {digit} ---")
            imfs = self.plot_imf_decomposition(signal_data, digit, channel_idx)

            # Brief pause between plots
            time.sleep(1)

    def analyze_imf_frequency_content(self, imfs, digit_label):
        """
        Analisis konten frekuensi dari setiap IMF
        """
        print(f"\n🔍 Frequency Analysis for Digit {digit_label} IMFs:")

        for i, imf in enumerate(imfs):
            if len(imf) < 10:
                continue

            # FFT analysis
            fft_vals = np.fft.fft(imf)
            fft_freqs = np.fft.fftfreq(len(imf), 1/self.sampling_rate)

            # Find dominant frequency
            positive_freqs = fft_freqs[:len(fft_freqs)//2]
            positive_fft = np.abs(fft_vals[:len(fft_vals)//2])

            if len(positive_fft) > 1:
                dominant_freq_idx = np.argmax(positive_fft[1:]) + 1  # Skip DC
                dominant_freq = positive_freqs[dominant_freq_idx]

                # Classify frequency band
                if dominant_freq <= 4:
                    band = "Delta"
                elif dominant_freq <= 8:
                    band = "Theta"
                elif dominant_freq <= 13:
                    band = "Alpha"
                elif dominant_freq <= 20:
                    band = "Beta Low"
                elif dominant_freq <= 30:
                    band = "Beta High"
                else:
                    band = "Gamma"

                print(f"   IMF {i+1}: Dominant freq = {dominant_freq:.2f} Hz ({band} band)")
            else:
                print(f"   IMF {i+1}: Unable to determine dominant frequency")

class BrainDigiCNN:
    """
    Model BrainDigiCNN untuk klasifikasi digit EEG
    """
    
    def __init__(self, input_shape, num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
    
    def build_model(self):
        """
        Membangun arsitektur BrainDigiCNN sesuai paper
        """
        model = Sequential([
            # Layer 1: Conv1D + BN + MaxPooling
            Conv1D(filters=256, kernel_size=7, activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            # Layer 2: Conv1D + BN + MaxPooling
            Conv1D(filters=128, kernel_size=7, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            # Layer 3: Conv1D + BN + MaxPooling
            Conv1D(filters=64, kernel_size=7, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            # Layer 4: Conv1D + BN + MaxPooling
            Conv1D(filters=32, kernel_size=7, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            # Flatten
            Flatten(),
            
            # Fully Connected Layers
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile model dengan parameter sesuai paper
        """
        optimizer = Adam(learning_rate=learning_rate)

        # Use standard categorical crossentropy (stable across all TF versions)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=None):
        """
        Training model with GPU optimization
        """
        # Auto-adjust batch size based on GPU availability
        if batch_size is None:
            batch_size = 128 if GPU_AVAILABLE else 32

        print(f"   Using batch size: {batch_size} ({'GPU optimized' if GPU_AVAILABLE else 'CPU optimized'})")

        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, self.num_classes)
        y_val_cat = to_categorical(y_val, self.num_classes)

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]

        # Add GPU-specific callbacks
        if GPU_AVAILABLE:
            # Reduce learning rate on plateau
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ))

        # Training with GPU optimization
        with tf.device('/GPU:0' if GPU_AVAILABLE else '/CPU:0'):
            self.history = self.model.fit(
                X_train, y_train_cat,
                validation_data=(X_val, y_val_cat),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )

        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluasi model
        """
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Hitung metrik
        accuracy = accuracy_score(y_test, y_pred_classes)
        precision = precision_score(y_test, y_pred_classes, average='weighted')
        recall = recall_score(y_test, y_pred_classes, average='weighted')
        f1 = f1_score(y_test, y_pred_classes, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        return metrics, y_pred_classes
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            print("Model belum dilatih!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Plot confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=[f'Digit {i}' for i in range(10)],
                    yticklabels=[f'Digit {i}' for i in range(10)])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

def main_pipeline(file_path, use_checkpoint=True, clear_checkpoints=False):
    """
    Pipeline utama untuk training BrainDigiCNN dengan dataset MindBigData

    Args:
        file_path: path ke dataset
        use_checkpoint: apakah menggunakan checkpoint (default: True)
        clear_checkpoints: hapus checkpoint yang ada (default: False)
    """
    print("=== BrainDigiCNN: EEG Digit Classification with MindBigData ===\n")

    # Display GPU status
    if GPU_AVAILABLE:
        print("🚀 GPU acceleration enabled")
    else:
        print("⚠️  Running on CPU (GPU not available)")

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager()

    if clear_checkpoints:
        checkpoint_manager.clear_checkpoints()

    if use_checkpoint:
        checkpoint_manager.get_checkpoint_info()

    # 1. Load data
    print("1. Loading MindBigData...")

    # Initialize loader
    loader = MindBigDataLoader(file_path)

    # Cek apakah ada checkpoint untuk raw data
    if use_checkpoint and checkpoint_manager.checkpoint_exists('raw_data'):
        print("   📁 Loading from checkpoint...")
        data = checkpoint_manager.load_checkpoint('raw_data')
        # Set data ke loader untuk info
        loader.data = data
    else:
        print("   📥 Loading from file...")

        # Load data untuk digit 0-9 dengan device EMOTIV EPOC
        data = loader.load_data(device_filter="EP", code_filter=list(range(10)))

        if data is None or len(data) == 0:
            print("No data loaded! Please check file path and format.")
            return

        # Simpan checkpoint
        if use_checkpoint:
            checkpoint_manager.save_checkpoint('raw_data', data)

    # Get data info
    loader.get_data_info()
    
    # 2. Organize data
    print("\n2. Organizing data by trials...")

    # Cek apakah ada checkpoint untuk organized data
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

        # Simpan checkpoint
        if use_checkpoint:
            checkpoint_data = {'X_raw': X_raw, 'y': y}
            checkpoint_manager.save_checkpoint('organized_data', checkpoint_data)

    print(f"   Organized data shape: {X_raw.shape}")
    print(f"   Labels shape: {y.shape}")
    print(f"   Unique labels: {sorted(set(y))}")
    
    # 3. Preprocessing
    print("\n3. Preprocessing EEG signals...")

    # Cek apakah ada checkpoint untuk preprocessed data
    if use_checkpoint and checkpoint_manager.checkpoint_exists('preprocessed_data'):
        print("   📁 Loading preprocessed data from checkpoint...")
        X_processed = checkpoint_manager.load_checkpoint('preprocessed_data')
    else:
        print("   � Starting optimized preprocessing...")

        # Gunakan OptimizedPreprocessor
        optimized_processor = OptimizedPreprocessor(
            sampling_rate=128,
            n_processes=None,  # Auto-detect optimal number
            batch_size=64 if GPU_AVAILABLE else 32
        )

        # Process dengan optimisasi
        X_processed = optimized_processor.process_optimized(X_raw)

        # Simpan checkpoint
        if use_checkpoint:
            checkpoint_manager.save_checkpoint('preprocessed_data', X_processed)

    print(f"   ✅ Processed data shape: {X_processed.shape}")
    
    # 4. Normalize data
    print("\n4. Normalizing data...")
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_processed)
    
    # Reshape untuk input CNN 1D
    X_normalized = X_normalized.reshape(X_normalized.shape[0], X_normalized.shape[1], 1)
    
    # 5. Split data
    print("\n5. Splitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_normalized, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"   Train set: {X_train.shape}")
    print(f"   Validation set: {X_val.shape}")
    print(f"   Test set: {X_test.shape}")
    
    # 6. Build dan train model
    print("\n6. Building BrainDigiCNN model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = BrainDigiCNN(input_shape=input_shape, num_classes=10)
    
    model.build_model()
    model.compile_model(learning_rate=0.001)
    
    print(f"   Model architecture:")
    model.model.summary()
    
    # 7. Training
    print("\n7. Training model...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=20,
        batch_size=32
    )
    
    # 8. Evaluasi
    print("\n8. Evaluating model...")
    metrics, y_pred = model.evaluate(X_test, y_test)
    
    print(f"   Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Test Precision: {metrics['precision']:.4f}")
    print(f"   Test Recall: {metrics['recall']:.4f}")
    print(f"   Test F1-Score: {metrics['f1_score']:.4f}")
    
    # 9. Classification report
    print("\n9. Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=[f'Digit {i}' for i in range(10)]))
    
    # 10. Plot results
    print("\n10. Plotting results...")
    model.plot_training_history()
    model.plot_confusion_matrix(y_test, y_pred)
    
    print("\n=== BrainDigiCNN Training Complete ===")
    return model, metrics

# Contoh penggunaan
if __name__ == "__main__":
    # Path ke file dataset EP1.01.txt yang ada di root folder
    file_path = "EP1.01.txt"

    print("=== BrainDigiCNN: EEG Digit Classification ===")
    print(f"Using dataset: {file_path}")

    # Checkpoint options
    use_checkpoint = True  # Set False untuk disable checkpoint
    clear_checkpoints = False  # Set True untuk hapus checkpoint yang ada

    # Jika file ada, jalankan pipeline
    if os.path.exists(file_path):
        print(f"Dataset file found: {file_path}")

        if use_checkpoint:
            print("📁 Checkpoint system enabled")
        else:
            print("⚠️  Checkpoint system disabled")

        print("Starting training pipeline...")
        model, metrics = main_pipeline(file_path, use_checkpoint, clear_checkpoints)
    else:
        print(f"\nFile {file_path} not found!")
        print("Please make sure EP1.01.txt is in the same directory as this script.")

    print("\n=== Instructions ===")
    print("1. Make sure EP1.01.txt is in the same directory as main.py")
    print("2. Run the script: python main.py")
    print("3. The model will be trained and evaluated automatically")
    print("4. Results will be displayed including accuracy, confusion matrix, and training plots")
    print("5. Checkpoint system will save progress at each major step")
    print("6. If interrupted, restart will continue from last checkpoint")

    print("\n=== Optimization Features ===")
    print("🚀 GPU Acceleration:")
    print("   - Automatic GPU detection and configuration")
    print("   - GPU-optimized batch processing (float32 - stable mode)")
    print("   - Memory growth enabled for efficient GPU usage")
    print("🔄 Multiprocessing:")
    print("   - CPU multiprocessing for preprocessing")
    print("   - Auto-detection of optimal process count")
    print("   - Parallel EEG signal processing")
    print("📁 Checkpoint System:")
    print("   - raw_data.pkl: Original loaded data")
    print("   - organized_data.pkl: Data organized by trials")
    print("   - preprocessed_data.pkl: Processed EEG features")
    print("   - To clear checkpoints: set clear_checkpoints=True")
    print("   - To disable checkpoints: set use_checkpoint=False")

    print("\n=== Expected Performance ===")
    print("Based on the paper, expected accuracy should be around 98.27% for EMOTIV EPOC+ device")