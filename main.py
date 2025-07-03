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
import warnings
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

            # Set GPU as preferred device
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

            # Enable mixed precision for better performance
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("✅ Mixed precision enabled (float16)")

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
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta_low': (12, 16),
            'beta_high': (16, 24),
            'gamma': (24, 40)
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
    
    def lowpass_filter(self, data, cutoff_freq=45, order=5):
        """
        Lowpass filter untuk noise removal
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
    
    def empirical_mode_decomposition(self, data, max_imf=10):
        """
        Implementasi sederhana EMD (Empirical Mode Decomposition)
        """
        if len(data) < 20:
            return np.array([data])
        
        def get_spline_envelope(data, indices):
            if len(indices) < 2:
                return np.zeros_like(data)
            try:
                return np.interp(range(len(data)), indices, data[indices])
            except:
                return np.zeros_like(data)
        
        imfs = []
        residue = data.copy()
        
        for i in range(max_imf):
            if len(residue) < 10:
                break
                
            h = residue.copy()
            
            # Iterasi sifting
            for j in range(20):  # maksimal 20 iterasi
                try:
                    # Cari local maxima dan minima
                    maxima_idx = signal.argrelextrema(h, np.greater)[0]
                    minima_idx = signal.argrelextrema(h, np.less)[0]
                    
                    if len(maxima_idx) < 2 or len(minima_idx) < 2:
                        break
                    
                    # Buat envelope
                    upper_env = get_spline_envelope(h, maxima_idx)
                    lower_env = get_spline_envelope(h, minima_idx)
                    
                    mean_env = (upper_env + lower_env) / 2
                    h_new = h - mean_env
                    
                    # Cek kriteria stopping
                    if np.sum(np.abs(h_new - h)) < 1e-6:
                        break
                        
                    h = h_new
                except:
                    break
            
            imfs.append(h)
            residue = residue - h
            
            # Cek apakah residue monotonic
            try:
                if len(signal.argrelextrema(residue, np.greater)[0]) < 2:
                    break
            except:
                break
        
        return np.array(imfs) if imfs else np.array([data])
    
    def hilbert_huang_transform(self, imfs):
        """
        Hilbert-Huang Transform untuk ekstraksi fitur
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
        Pipeline lengkap untuk preprocessing EEG
        """
        # 1. Lowpass filter untuk noise removal
        denoised = self.lowpass_filter(eeg_data)
        
        # 2. Notch filter untuk power line interference
        denoised = self.notch_filter(denoised)
        
        # 3. Decompose ke sub-bands
        band_features = []
        
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            try:
                # Bandpass filter
                band_data = self.butterworth_filter(denoised, low_freq, high_freq)
                
                # EMD
                imfs = self.empirical_mode_decomposition(band_data)
                
                # HHT
                if len(imfs) > 0:
                    hht_features = self.hilbert_huang_transform(imfs)
                    band_features.append(hht_features.flatten())
                else:
                    # Fallback
                    band_features.append(band_data)
            except:
                # Fallback jika ada error
                band_features.append(denoised)
        
        # Gabungkan semua fitur dari semua band
        try:
            all_features = np.concatenate(band_features)
        except:
            all_features = denoised
        
        return all_features

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

        # Use mixed precision loss scaling if GPU is available
        if GPU_AVAILABLE:
            loss = tf.keras.losses.CategoricalCrossentropy()
        else:
            loss = 'categorical_crossentropy'

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
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

def main_pipeline(file_path):
    """
    Pipeline utama untuk training BrainDigiCNN dengan dataset MindBigData
    """
    print("=== BrainDigiCNN: EEG Digit Classification with MindBigData ===\n")

    # Display GPU status
    if GPU_AVAILABLE:
        print("🚀 GPU acceleration enabled")
    else:
        print("⚠️  Running on CPU (GPU not available)")
    
    # 1. Load data
    print("1. Loading MindBigData...")
    loader = MindBigDataLoader(file_path)
    
    # Load data untuk digit 0-9 dengan device EMOTIV EPOC
    data = loader.load_data(device_filter="EP", code_filter=list(range(10)))
    
    if data is None or len(data) == 0:
        print("No data loaded! Please check file path and format.")
        return
    
    # Get data info
    loader.get_data_info()
    
    # 2. Organize data
    print("\n2. Organizing data by trials...")
    X_raw, y = loader.organize_by_trials()
    
    if X_raw is None or len(X_raw) == 0:
        print("No organized data available!")
        return
    
    print(f"   Organized data shape: {X_raw.shape}")
    print(f"   Labels shape: {y.shape}")
    print(f"   Unique labels: {sorted(set(y))}")
    
    # 3. Preprocessing
    print("\n3. Preprocessing EEG signals...")
    processor = EEGSignalProcessor(sampling_rate=128)
    
    X_processed = []
    for i, eeg_sample in enumerate(X_raw):
        if i % 50 == 0:
            print(f"   Processing sample {i+1}/{len(X_raw)}")
        
        # Process setiap channel
        processed_channels = []
        for ch in range(eeg_sample.shape[0]):
            try:
                features = processor.process_eeg_signal(eeg_sample[ch])
                processed_channels.append(features)
            except Exception as e:
                print(f"   Error processing channel {ch}: {e}")
                # Fallback: gunakan raw signal
                processed_channels.append(eeg_sample[ch])
        
        # Pastikan semua channel memiliki panjang yang sama
        min_length = min(len(ch) for ch in processed_channels)
        processed_channels = [ch[:min_length] for ch in processed_channels]
        
        X_processed.append(np.concatenate(processed_channels))
    
    X_processed = np.array(X_processed)
    print(f"   Processed data shape: {X_processed.shape}")
    
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

    # Jika file ada, jalankan pipeline
    if os.path.exists(file_path):
        print(f"Dataset file found: {file_path}")
        print("Starting training pipeline...")
        model, metrics = main_pipeline(file_path)
    else:
        print(f"\nFile {file_path} not found!")
        print("Please make sure EP1.01.txt is in the same directory as this script.")

    print("\n=== Instructions ===")
    print("1. Make sure EP1.01.txt is in the same directory as main.py")
    print("2. Run the script: python main.py")
    print("3. The model will be trained and evaluated automatically")
    print("4. Results will be displayed including accuracy, confusion matrix, and training plots")

    print("\n=== Expected Performance ===")
    print("Based on the paper, expected accuracy should be around 98.27% for EMOTIV EPOC+ device")