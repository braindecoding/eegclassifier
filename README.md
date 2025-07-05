# eegclassifier
BrainDigiCNN based preprocessing  
BAND-Wise EMD-HHT Preprocessing  
Batch Based Processing

```sh
Raw EEG Signal (256 timepoints, 14 channels)
    ↓ Butterworth Lowpass (45Hz, Order 5, IIR)
    ↓ Notch Filter (50Hz, IIR)
    ↓ Band Decomposition:
        ├── Delta (0.5-4 Hz)     → Butterworth Bandpass Order 5 → EMD (max 10) → HHT (IA,IP,IF)
        ├── Theta (4-8 Hz)       → Butterworth Bandpass Order 5 → EMD (max 10) → HHT (IA,IP,IF)
        ├── Alpha (8-12 Hz)      → Butterworth Bandpass Order 5 → EMD (max 10) → HHT (IA,IP,IF)
        ├── Beta Low (12-16 Hz)  → Butterworth Bandpass Order 5 → EMD (max 10) → HHT (IA,IP,IF)
        ├── Beta High (16-24 Hz) → Butterworth Bandpass Order 5 → EMD (max 10) → HHT (IA,IP,IF)
        └── Gamma (24-40 Hz)     → Butterworth Bandpass Order 5 → EMD (max 10) → HHT (IA,IP,IF)
    ↓ Feature Concatenation HHT dalam bentuk time series (bukan scalar). IA(t), IP(t), IF(t) sebagai fungsi waktu
    ↓ Normalization
Final Feature Vector
```

yang jadi input CNN adalah HHT features, TAPI:

1. HHT dalam bentuk time series (bukan scalar)
2. IA(t), IP(t), IF(t) sebagai fungsi waktu
3. Dimensi ~300,000 features (bukan ~1,500)
4. Concatenation semua HHT time series dari pipeline
5. CNN belajar dari temporal evolution HHT features

## Proper EMD
Standard practices untuk EMD details:

* interpolation = "cubic_spline"  # Industry standard
* stopping_criteria = ["SD_test", "Cauchy_convergence"]
* boundary_method = "mirror_extension"
* max_iterations = 50
* tolerance = 1e-6



## Dataset Info
from datasets import load_dataset

ds = load_dataset("DavidVivancos/MindBigData2022_MNIST_EP")

# Balanced Dataset

Train/validation/test split (60/20/20)
Batch Size: 32
Epochs: 10-20

1. Stratified
2. Train test val tanpa data leakage
3. cross vall 5 cv
4. random seed 42




File	Size	Content	Usage
hf_normalized_data.pkl	200GB	Preprocessed EEG features (52,022 × 516,096)	MAIN TRAINING DATA
hf_split_indices.pkl	0.4MB	Train/val/test indices	DATA SPLITTING
hf_raw_extracted.pkl	711MB	Labels (y) for training	LABELS
