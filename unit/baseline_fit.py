import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.signal import medfilt
import sys
sys.path.append('./')
from preprocessing_addFeatures import data_preprocessing

def estimate_baselines(signal, window_size=201, kernel_size=101, plot=True):
    """
    Estimate multiple baselines and quantify drift severity.
    
    Parameters:
        signal (np.ndarray): Input 1D vibration signal
        window_size (int): Window size for local baseline estimation
        kernel_size (int): Kernel size for smoothing (must be odd)
        plot (bool): Whether to plot the signal and detected baselines
    
    Returns:
        tuple: (baseline_1, baseline_2, drift_severity)
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Smooth the signal to reduce high frequency noise
    smoothed = medfilt(signal, kernel_size=kernel_size)
    
    # Sliding window mean/median as local baselines
    num_windows = len(signal) // window_size
    local_baselines = []

    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        if end > len(signal):
            break
        segment = smoothed[start:end]
        median_value = np.median(segment)
        local_baselines.append(median_value)

    local_baselines = np.array(local_baselines).reshape(-1, 1)
    
    # Cluster local baselines into two groups (KMeans)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(local_baselines)
    baseline_values = kmeans.cluster_centers_.flatten()
    baseline_values.sort()  # Ensure baseline_1 < baseline_2
    baseline_1, baseline_2 = baseline_values
    drift_severity = abs(baseline_2 - baseline_1)
    
    if plot:
        time = np.arange(len(signal))
        plt.figure(figsize=(12, 5))
        plt.plot(time, signal, label="Original Signal", alpha=0.6)
        plt.plot(time, smoothed, label="Smoothed Signal", linewidth=1.5)
        for i in range(num_windows):
            plt.hlines(local_baselines[i], i*window_size, (i+1)*window_size, 
                       colors='r' if kmeans.labels_[i]==0 else 'g', linewidth=2)
        plt.title(f"Baseline Drift Detected: Δb = {drift_severity:.3f}")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return baseline_1, baseline_2, drift_severity

def compute_baseline_drift_feature(signal, num_segments=10, smoothing_kernel=51, plot=False):
    """
    Efficiently computes baseline drift severity feature (Δb) from a 1D signal.
    
    Parameters:
        signal (np.ndarray): 1D input signal.
        num_segments (int): Number of equal segments to divide the signal.
        smoothing_kernel (int): Kernel size for optional median filter (must be odd).
        plot (bool): Whether to plot the signal and baselines.
        
    Returns:
        float: Estimated baseline drift severity (Δb)
    """
    if smoothing_kernel % 2 == 0:
        smoothing_kernel += 1

    # Optional: fast smoothing via rolling median
    padded = np.pad(signal, (smoothing_kernel // 2,), mode='edge')
    smoothed = np.median(
        np.lib.stride_tricks.sliding_window_view(padded, smoothing_kernel),
        axis=1
    )
    
    # Segment the signal into equal parts and compute medians
    segment_len = len(smoothed) // num_segments
    segment_medians = [
        np.median(smoothed[i * segment_len : (i + 1) * segment_len])
        for i in range(num_segments)
    ]
    segment_medians = np.array(segment_medians)
    
    # Cluster via histogram peaks
    hist, bin_edges = np.histogram(segment_medians, bins='auto')
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    top_bins = bin_centers[np.argsort(hist)[-2:]]  # Two most frequent bin centers
    baseline_1, baseline_2 = np.sort(top_bins)
    drift_severity = abs(baseline_2 - baseline_1)

    if plot:
        time = np.arange(len(signal))
        plt.figure(figsize=(12, 5))
        plt.plot(time, signal, label='Original Signal', alpha=0.5)
        plt.plot(time, smoothed, label='Smoothed Signal', linewidth=1.5)
        plt.hlines([baseline_1, baseline_2], xmin=0, xmax=len(signal), 
                   colors=['r', 'g'], linestyles='--', label='Detected Baselines')
        plt.title(f'Detected Drift Δb = {drift_severity:.3f}')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return drift_severity

def compute_baseline_drift_feature_fast(signal, num_segments=3, plot=False):
    """
    Ultra-fast baseline drift severity computation (Δb) without smoothing.
    
    Parameters:
        signal (np.ndarray): 1D input signal.
        num_segments (int): Number of segments for median estimation.
        plot (bool): Whether to plot the signal and baselines.
        
    Returns:
        float: Estimated baseline drift severity (Δb)
    """
    # Divide signal into equal segments and compute medians
    segment_len = len(signal) // num_segments
    segment_medians = [
        np.median(signal[i * segment_len : (i + 1) * segment_len])
        for i in range(num_segments)
    ]
    segment_medians = np.array(segment_medians)
    
    # # Use histogram to identify dominant baselines
    # hist, bin_edges = np.histogram(segment_medians, bins='auto')
    # bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # top_bins = bin_centers[np.argsort(hist)[-2:]]
    # baseline_1, baseline_2 = np.sort(top_bins)
    baseline_1, baseline_2 = np.max(segment_medians), np.min(segment_medians)
    drift_severity = abs(baseline_2 - baseline_1)

    if plot:
        time = np.arange(len(signal))
        plt.figure(figsize=(12, 5))
        plt.plot(time, signal, label='Original Signal', alpha=0.6)
        plt.hlines([baseline_1, baseline_2], xmin=0, xmax=len(signal), 
                   colors=['r', 'g'], linestyles='--', label='Detected Baselines')
        plt.title(f'Drift Feature Δb = {drift_severity:.3f}')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return drift_severity

if __name__ == "__main__":
    # Example usage
    x_train, y_train, x_val, y_val, _ = data_preprocessing(balanced=False, platform='pytorch', normal_class=False, method='test')
    signal = x_val[836][:1008]
    # baseline_1, baseline_2, drift_severity = estimate_baselines(signal, window_size=201, kernel_size=101, plot=True)
    # drift_severity = compute_baseline_drift_feature(signal, num_segments=5, smoothing_kernel=51, plot=True)
    drift_severity = compute_baseline_drift_feature_fast(signal, num_segments=3, plot=True)