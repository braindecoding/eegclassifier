#!/usr/bin/env python3
"""
Script untuk plotting dan analisis IMF decomposition
Visualisasi bagaimana EMD memecah sinyal EEG untuk setiap digit
"""

import numpy as np
import matplotlib.pyplot as plt
from main import MindBigDataLoader, CheckpointManager, EEGSignalProcessor
import time

def load_sample_data():
    """
    Load sample data untuk setiap digit
    """
    print("📥 Loading sample EEG data for IMF analysis...")
    
    checkpoint_manager = CheckpointManager()
    
    # Load organized data
    if checkpoint_manager.checkpoint_exists('organized_data'):
        print("   📁 Loading from checkpoint...")
        organized_data = checkpoint_manager.load_checkpoint('organized_data')
        X_raw = organized_data['X']
        y = organized_data['y']
    else:
        print("   ❌ No organized data found! Please run main pipeline first.")
        return None
    
    print(f"   ✅ Loaded {len(X_raw)} samples")
    
    # Get one representative sample for each digit
    samples_dict = {}
    
    for digit in range(10):
        # Find indices for this digit
        digit_indices = np.where(y == digit)[0]
        
        if len(digit_indices) > 0:
            # Take the first sample for this digit
            sample_idx = digit_indices[0]
            samples_dict[digit] = X_raw[sample_idx]
            print(f"   Digit {digit}: Sample shape {X_raw[sample_idx].shape}")
        else:
            print(f"   ⚠️  No samples found for digit {digit}")
    
    return samples_dict

def plot_imf_for_all_digits():
    """
    Plot IMF decomposition untuk semua digit
    """
    print("🎯 Starting IMF Analysis for All Digits")
    print("=" * 50)
    
    # Load sample data
    samples_dict = load_sample_data()
    if samples_dict is None:
        return
    
    # Initialize EEG processor
    processor = EEGSignalProcessor(sampling_rate=128)
    
    # Plot IMFs for each digit
    print(f"\n📊 Plotting IMF decomposition for digits 0-9...")
    
    for digit in sorted(samples_dict.keys()):
        print(f"\n{'='*20} DIGIT {digit} {'='*20}")
        
        eeg_sample = samples_dict[digit]
        
        # Use first channel for analysis
        if len(eeg_sample.shape) > 1:
            signal_data = eeg_sample[0]  # First channel
            print(f"   Using channel 0, signal length: {len(signal_data)}")
        else:
            signal_data = eeg_sample
            print(f"   Single channel signal, length: {len(signal_data)}")
        
        # Plot IMF decomposition
        imfs = processor.plot_imf_decomposition(
            eeg_data=signal_data,
            digit_label=digit,
            channel_idx=0,
            save_plot=True
        )
        
        # Analyze frequency content
        processor.analyze_imf_frequency_content(imfs, digit)
        
        print(f"   ✅ Completed analysis for digit {digit}")
        
        # Small delay to prevent overwhelming
        time.sleep(0.5)

def plot_imf_comparison():
    """
    Plot perbandingan IMF untuk beberapa digit dalam satu figure
    """
    print("\n📊 Creating IMF comparison plot...")
    
    # Load sample data
    samples_dict = load_sample_data()
    if samples_dict is None:
        return
    
    # Initialize processor
    processor = EEGSignalProcessor(sampling_rate=128)
    
    # Select 3 digits for comparison
    digits_to_compare = [0, 5, 9]
    
    fig, axes = plt.subplots(len(digits_to_compare), 6, figsize=(20, 12))
    
    for row, digit in enumerate(digits_to_compare):
        if digit not in samples_dict:
            continue
            
        eeg_sample = samples_dict[digit]
        signal_data = eeg_sample[0] if len(eeg_sample.shape) > 1 else eeg_sample
        
        # Preprocessing
        denoised = processor.lowpass_filter(signal_data)
        denoised = processor.notch_filter(denoised)
        
        # EMD
        imfs = processor.empirical_mode_decomposition(denoised)
        
        # Time axis
        time_axis = np.arange(len(signal_data)) / processor.sampling_rate
        
        # Plot original signal
        axes[row, 0].plot(time_axis, signal_data, 'b-', linewidth=1)
        axes[row, 0].set_title(f'Digit {digit} - Original')
        axes[row, 0].set_ylabel('Amplitude')
        axes[row, 0].grid(True, alpha=0.3)
        
        # Plot first 5 IMFs
        for i in range(min(5, len(imfs))):
            imf = imfs[i]
            axes[row, i + 1].plot(time_axis[:len(imf)], imf, linewidth=1)
            axes[row, i + 1].set_title(f'IMF {i+1}')
            axes[row, i + 1].grid(True, alpha=0.3)
    
    # Set x-labels for bottom row
    for col in range(6):
        axes[-1, col].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig('imf_comparison_multiple_digits.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📁 Comparison plot saved as: imf_comparison_multiple_digits.png")

def analyze_imf_statistics():
    """
    Analisis statistik IMF untuk semua digit
    """
    print("\n📈 Statistical Analysis of IMFs across digits...")
    
    # Load sample data
    samples_dict = load_sample_data()
    if samples_dict is None:
        return
    
    processor = EEGSignalProcessor(sampling_rate=128)
    
    # Collect IMF statistics
    imf_stats = {}
    
    for digit in sorted(samples_dict.keys()):
        eeg_sample = samples_dict[digit]
        signal_data = eeg_sample[0] if len(eeg_sample.shape) > 1 else eeg_sample
        
        # Preprocessing
        denoised = processor.lowpass_filter(signal_data)
        denoised = processor.notch_filter(denoised)
        
        # EMD
        imfs = processor.empirical_mode_decomposition(denoised)
        
        imf_stats[digit] = {
            'num_imfs': len(imfs),
            'imf_energies': [np.sqrt(np.mean(imf**2)) for imf in imfs],
            'imf_lengths': [len(imf) for imf in imfs]
        }
    
    # Print statistics
    print(f"\n📊 IMF Statistics Summary:")
    print(f"{'Digit':<6} {'Num IMFs':<10} {'Avg Energy':<12} {'Energy Distribution'}")
    print("-" * 60)
    
    for digit in sorted(imf_stats.keys()):
        stats = imf_stats[digit]
        avg_energy = np.mean(stats['imf_energies'])
        energy_dist = [f"{e:.3f}" for e in stats['imf_energies'][:3]]  # First 3 IMFs
        
        print(f"{digit:<6} {stats['num_imfs']:<10} {avg_energy:<12.4f} {energy_dist}")
    
    # Plot energy distribution
    plt.figure(figsize=(12, 8))
    
    for digit in sorted(imf_stats.keys()):
        energies = imf_stats[digit]['imf_energies']
        plt.plot(range(1, len(energies) + 1), energies, 'o-', label=f'Digit {digit}')
    
    plt.xlabel('IMF Number')
    plt.ylabel('RMS Energy')
    plt.title('IMF Energy Distribution Across Digits')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('imf_energy_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📁 Energy distribution plot saved as: imf_energy_distribution.png")

def main():
    """
    Main function untuk IMF analysis
    """
    print("🧠 EEG IMF Analysis Tool")
    print("=" * 40)
    
    try:
        # 1. Plot individual IMF decompositions
        plot_imf_for_all_digits()
        
        # 2. Create comparison plot
        plot_imf_comparison()
        
        # 3. Statistical analysis
        analyze_imf_statistics()
        
        print(f"\n🎉 IMF Analysis completed!")
        print(f"   Check generated PNG files for visualizations")
        
    except Exception as e:
        print(f"❌ Error during IMF analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
