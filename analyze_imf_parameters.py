#!/usr/bin/env python3
"""
Script untuk menganalisis parameter IMF optimal
Menguji berbagai nilai max_imf untuk menentukan setting terbaik
"""

import numpy as np
import matplotlib.pyplot as plt
from main import MindBigDataLoader, CheckpointManager, EEGSignalProcessor
import time

def test_imf_parameters():
    """
    Test berbagai nilai max_imf untuk menentukan optimal setting
    """
    print("🔬 Testing IMF Parameters for Optimal Setting")
    print("=" * 50)
    
    # Load sample data
    checkpoint_manager = CheckpointManager()
    
    if checkpoint_manager.checkpoint_exists('organized_data'):
        print("📁 Loading organized data...")
        organized_data = checkpoint_manager.load_checkpoint('organized_data')
        X_raw = organized_data['X']
        y = organized_data['y']
    else:
        print("❌ No organized data found!")
        return
    
    # Get representative samples for each digit
    sample_signals = []
    for digit in range(10):
        digit_indices = np.where(y == digit)[0]
        if len(digit_indices) > 0:
            # Take first sample for this digit
            sample = X_raw[digit_indices[0]]
            # Use first channel
            signal_data = sample[0] if len(sample.shape) > 1 else sample
            sample_signals.append((digit, signal_data))
    
    print(f"✅ Loaded {len(sample_signals)} representative samples")
    
    # Test different max_imf values
    max_imf_values = [5, 6, 7, 8, 9, 10, 12, 15]
    
    results = {}
    
    for max_imf in max_imf_values:
        print(f"\n🧪 Testing max_imf = {max_imf}")
        
        processor = EEGSignalProcessor(sampling_rate=128)
        
        imf_counts = []
        processing_times = []
        
        for digit, signal_data in sample_signals:
            # Preprocessing
            denoised = processor.lowpass_filter(signal_data)
            denoised = processor.notch_filter(denoised)
            
            # EMD with specific max_imf
            start_time = time.time()
            imfs = processor.empirical_mode_decomposition(denoised, max_imf=max_imf)
            processing_time = time.time() - start_time
            
            imf_counts.append(len(imfs))
            processing_times.append(processing_time)
            
            print(f"   Digit {digit}: {len(imfs)} IMFs, {processing_time:.3f}s")
        
        results[max_imf] = {
            'avg_imf_count': np.mean(imf_counts),
            'std_imf_count': np.std(imf_counts),
            'avg_processing_time': np.mean(processing_times),
            'imf_counts': imf_counts
        }
        
        print(f"   Average IMFs: {np.mean(imf_counts):.1f} ± {np.std(imf_counts):.1f}")
        print(f"   Average time: {np.mean(processing_times):.3f}s")
    
    # Analysis and recommendations
    print(f"\n📊 IMF Parameter Analysis Results:")
    print(f"{'max_imf':<8} {'Avg IMFs':<10} {'Std IMFs':<10} {'Avg Time':<12} {'Efficiency'}")
    print("-" * 60)
    
    for max_imf in max_imf_values:
        result = results[max_imf]
        efficiency = result['avg_imf_count'] / result['avg_processing_time']
        
        print(f"{max_imf:<8} {result['avg_imf_count']:<10.1f} {result['std_imf_count']:<10.2f} "
              f"{result['avg_processing_time']:<12.3f} {efficiency:<10.1f}")
    
    # Plot results
    plot_imf_analysis(results, max_imf_values)
    
    # Recommendations
    make_recommendations(results, max_imf_values)

def plot_imf_analysis(results, max_imf_values):
    """
    Plot analysis results
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Average IMF count vs max_imf
    avg_imfs = [results[max_imf]['avg_imf_count'] for max_imf in max_imf_values]
    std_imfs = [results[max_imf]['std_imf_count'] for max_imf in max_imf_values]
    
    ax1.errorbar(max_imf_values, avg_imfs, yerr=std_imfs, marker='o', capsize=5)
    ax1.plot([0, max(max_imf_values)], [0, max(max_imf_values)], 'r--', alpha=0.5, label='max_imf limit')
    ax1.set_xlabel('max_imf Parameter')
    ax1.set_ylabel('Average IMFs Generated')
    ax1.set_title('IMF Count vs max_imf Parameter')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Processing time vs max_imf
    avg_times = [results[max_imf]['avg_processing_time'] for max_imf in max_imf_values]
    
    ax2.plot(max_imf_values, avg_times, 'bo-')
    ax2.set_xlabel('max_imf Parameter')
    ax2.set_ylabel('Average Processing Time (s)')
    ax2.set_title('Processing Time vs max_imf Parameter')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Efficiency (IMFs per second)
    efficiency = [results[max_imf]['avg_imf_count'] / results[max_imf]['avg_processing_time'] 
                  for max_imf in max_imf_values]
    
    ax3.plot(max_imf_values, efficiency, 'go-')
    ax3.set_xlabel('max_imf Parameter')
    ax3.set_ylabel('Efficiency (IMFs/second)')
    ax3.set_title('Processing Efficiency vs max_imf Parameter')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: IMF count distribution
    for i, max_imf in enumerate([6, 8, 10, 12]):
        if max_imf in results:
            imf_counts = results[max_imf]['imf_counts']
            ax4.hist(imf_counts, bins=range(1, max(imf_counts)+2), alpha=0.6, 
                    label=f'max_imf={max_imf}')
    
    ax4.set_xlabel('Number of IMFs Generated')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of IMF Counts')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('imf_parameter_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📁 Analysis plot saved as: imf_parameter_analysis.png")

def make_recommendations(results, max_imf_values):
    """
    Make recommendations based on analysis
    """
    print(f"\n💡 RECOMMENDATIONS:")
    print("=" * 40)
    
    # Find optimal max_imf based on different criteria
    
    # 1. Saturation point (where IMF count stops increasing significantly)
    avg_imfs = [results[max_imf]['avg_imf_count'] for max_imf in max_imf_values]
    
    saturation_point = None
    for i in range(1, len(avg_imfs)):
        if avg_imfs[i] - avg_imfs[i-1] < 0.5:  # Less than 0.5 IMF increase
            saturation_point = max_imf_values[i-1]
            break
    
    # 2. Best efficiency
    efficiency = [results[max_imf]['avg_imf_count'] / results[max_imf]['avg_processing_time'] 
                  for max_imf in max_imf_values]
    best_efficiency_idx = np.argmax(efficiency)
    best_efficiency_max_imf = max_imf_values[best_efficiency_idx]
    
    # 3. Theoretical optimal (based on EEG frequency bands)
    theoretical_optimal = 7  # 6 frequency bands + 1 residue
    
    print(f"1. 📈 SATURATION ANALYSIS:")
    if saturation_point:
        print(f"   IMF count saturates at max_imf = {saturation_point}")
        print(f"   Beyond this point, minimal additional IMFs are generated")
    else:
        print(f"   No clear saturation point found in tested range")
    
    print(f"\n2. ⚡ EFFICIENCY ANALYSIS:")
    print(f"   Best efficiency at max_imf = {best_efficiency_max_imf}")
    print(f"   Efficiency: {efficiency[best_efficiency_idx]:.1f} IMFs/second")
    
    print(f"\n3. 🧠 THEORETICAL ANALYSIS:")
    print(f"   EEG has 6 main frequency bands (Delta → Gamma)")
    print(f"   Theoretical optimal: max_imf = {theoretical_optimal}")
    print(f"   This allows for 6 IMFs + 1 residue")
    
    print(f"\n4. 📊 PRACTICAL ANALYSIS:")
    for max_imf in [6, 7, 8, 10]:
        if max_imf in results:
            result = results[max_imf]
            print(f"   max_imf = {max_imf}: {result['avg_imf_count']:.1f} IMFs, "
                  f"{result['avg_processing_time']:.3f}s")
    
    # Final recommendation
    print(f"\n🎯 FINAL RECOMMENDATION:")
    
    if saturation_point and saturation_point <= 8:
        recommended = saturation_point
        reason = "saturation analysis"
    elif theoretical_optimal in max_imf_values:
        recommended = theoretical_optimal
        reason = "theoretical optimum"
    else:
        recommended = best_efficiency_max_imf
        reason = "best efficiency"
    
    print(f"   Recommended max_imf = {recommended}")
    print(f"   Reason: {reason}")
    
    if recommended in results:
        result = results[recommended]
        print(f"   Expected: {result['avg_imf_count']:.1f} IMFs, "
              f"{result['avg_processing_time']:.3f}s processing time")
    
    print(f"\n📝 IMPLEMENTATION:")
    print(f"   Update main.py line 804:")
    print(f"   def empirical_mode_decomposition(self, data, max_imf={recommended}):")

def main():
    """
    Main function
    """
    try:
        test_imf_parameters()
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
