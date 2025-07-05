# Implementasi Arsitektur BrainDigiCNN untuk Klasifikasi Digit EEG

Input → 4×(Conv1D→BatchNorm→MaxPool) → Flatten → 2×FC → Output

✅ Detailed Architecture:
LayerTypeFilters/NeuronsKernel SizeActivationNotesInputInput---Features dari HHTConv1Conv1D2567ReLU+ BatchNorm + MaxPool(2)Conv2Conv1D1287ReLU+ BatchNorm + MaxPool(2)Conv3Conv1D647ReLU+ BatchNorm + MaxPool(2)Conv4Conv1D327ReLU+ BatchNorm + MaxPool(2)FlattenFlatten---Convert to 1DFC1Dense128-ReLU+ Dropout(0.5)FC2Dense64-ReLU+ Dropout(0.5)OutputDense10-SoftMax10 digits (0-9)
⚙️ HYPERPARAMETERS (Dari Paper):

✅ Training Configuration:
ParameterValueSourceOptimizerAdamPaper Table 5Learning Rate0.001Paper Table 5Loss FunctionCategorical CrossentropyPaper Table 5Batch Size32Paper Table 5Epochs10-20Paper Table 5Early StoppingYes (patience=5)Paper mentionDropout0.5Paper mention

✅ Architecture Parameters:
ComponentSpecificationPaper ReferenceConv Filters256→128→64→32Paper Table 4Kernel Size7 (all layers)Paper Table 4PoolingMaxPool, size=2Paper Table 4Activation HiddenReLUPaper explicitActivation OutputSoftMaxPaper explicitBatch NormalizationAfter each Conv1DPaper Table 4

📜 QUOTES DARI PAPER:
Architecture Quotes:

"four convolutional layers each layer having a convolutional layer followed by batch normalization (BN) layer which then further followed by max-pooling (MP) layer"
"Conv1D Filter 256/Kernel 7, Conv1D Filter 128/Kernel 7, Conv1D Filter 64/Kernel 7, Conv1D Filter 32/Kernel 7"
"Dense 128, Dense 64, Dense 10"
"The activation function ReLU and SoftMax was used at hidden layers and output layer, respectively"

Training Quotes:

"optimizer Adam, learning rate 0.001, Loss function Categorical Crossentropy, batch size 32, epochs 10-20"


Data Split:

Paper menggunakan 70:30 split, then validation from 30%
train_ratio = 0.70
val_ratio = 0.15  
test_ratio = 0.15




Training Configuration:
python# Hyperparameters sesuai paper
training_config = {
    'optimizer': 'Adam',
    'learning_rate': 0.001,
    'loss': 'categorical_crossentropy',
    'batch_size': 32,
    'epochs': 20,
    'early_stopping': True,
    'patience': 5,
    'monitor': 'val_loss',
    'restore_best_weights': True
}



1. Input Shape Calculation:
python# Setelah feature extraction dari EMD+HHT
# Estimasi: 25,000 - 40,000 features per sample
input_shape = (feature_length, 1)  # untuk Conv1D
2. Memory Optimization:
python# Untuk dataset besar
batch_size = 32  # sesuai paper
gradient_accumulation = True  # jika memory terbatas
mixed_precision = True  # untuk efisiensi
3. Monitoring Training:
pythoncallbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.1, patience=3),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]
🎯 SUMMARY:
✅ Architecture Verified:

4 Conv1D layers (256→128→64→32 filters)
Kernel size 7 untuk semua conv layers
BatchNorm + MaxPool setelah setiap conv
2 FC layers (128→64 neurons)
Dropout 0.5 pada FC layers

✅ Hyperparameters Verified:

Adam optimizer, lr=0.001
Categorical crossentropy loss
Batch size 32, epochs 10-20
Early stopping patience=5

✅ Expected Performance:

Target accuracy: 98.27% (MBD dataset)
Compatible dengan feature extraction pipeline
Scalable untuk multi-subject datasets