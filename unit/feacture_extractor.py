import numpy as np
from collections import Counter


def calculate_empty_ratio(signal):
    """
    Calculate the empty ratio of a 1D signal.
    
    Parameters:
        signal (array-like): A 1D array or list of numeric values.
        
    Returns:
        float: The ratio of zero (empty) values to total number of elements.
    """
    signal = np.asarray(signal)
    
    if signal.ndim != 1:
        raise ValueError("Input must be a 1D array.")
    
    total_elements = len(signal)
    zero_elements = np.sum(signal == 0)
    
    empty_ratio = zero_elements / total_elements if total_elements > 0 else 0.0
    return empty_ratio

def calculate_peak_intensity(signal):
    """
    Calculate the peak intensity of a 1D signal.
    
    Peak_intensity = d_0.9 / d_max
    Where:
        d_0.9 = distance between 5th and 95th percentile (central 90% range)
        d_max = distance between min and max values (total range)
    
    Parameters:
        signal (array-like): A 1D array or list of numeric values.
        
    Returns:
        float: Peak intensity value.
    """
    signal = np.asarray(signal)
    
    if signal.ndim != 1:
        raise ValueError("Input must be a 1D array.")
    
    p5 = np.percentile(signal, 5)
    p95 = np.percentile(signal, 95)
    d_09 = p95 - p5
    
    d_max = np.max(signal) - np.min(signal)
    
    # Avoid division by zero
    peak_intensity = d_09 / d_max if d_max != 0 else 0.0
    
    return peak_intensity

def calculate_1stOrderSlope_linearity(signal):
    """
    Calculate the 1st Order Slope and linearity of a 1D signal based on linear fitting residuals.
    
    Linearity = max(abs(x_i - x_pred_i)) / (x_max - x_min)
    
    Parameters:
        signal (array-like): A 1D array or list of numeric values.
        
    Returns:
        float: 1st Order Slope and Linearity score.
    """
    signal = np.asarray(signal)
    
    if signal.ndim != 1:
        raise ValueError("Input must be a 1D array.")

    n = len(signal)
    x = np.arange(n)

    # Linear fit: y = a * x + b
    coeffs = np.polyfit(x, signal, deg=1)
    linear_fit = np.polyval(coeffs, x)

    # Calculate residuals
    residuals = np.abs(signal - linear_fit)
    max_delta = np.max(residuals)

    x_min = np.min(signal)
    x_max = np.max(signal)
    range_x = x_max - x_min

    linearity = max_delta / range_x if range_x != 0 else 0.0
    return coeffs[0], linearity

def calculate_equal_value_ratio(signal):
    """
    Calculate the equal value ratio of a 1D signal.
    
    Equal_ratio = N_equal / N_total
    Where:
        N_equal = number of repeated data points (excluding singletons)
        N_total = total number of data points
    
    Parameters:
        signal (array-like): A 1D array or list of numeric values.
        
    Returns:
        float: Equal value ratio.
    """
    signal = np.asarray(signal)
    
    if signal.ndim != 1:
        raise ValueError("Input must be a 1D array.")
    
    N_total = len(signal)
    if N_total == 0:
        return 0.0

    # Count the number of occurrences for each unique value
    value_counts = Counter(signal)
    
    # Count how many elements belong to repeating values (frequency >= 2)
    N_equal = sum(count for count in value_counts.values() if count > 1)

    equal_ratio = N_equal / N_total
    return equal_ratio

def calculate_standard_deviation(signal):
    """
    Calculate the sample standard deviation (SD) of a 1D signal.
    
    SD = sqrt((1 / (n - 1)) * sum((xi - x_mean)^2))
    
    Parameters:
        signal (array-like): A 1D array or list of numeric values.
        
    Returns:
        float: Sample standard deviation of the signal.
    """
    signal = np.asarray(signal)

    if signal.ndim != 1:
        raise ValueError("Input must be a 1D array.")
    
    n = len(signal)
    if n < 2:
        return 0.0  # Standard deviation is undefined for n < 2

    mean = np.mean(signal)
    variance = np.sum((signal - mean) ** 2) / (n - 1)
    std_dev = np.sqrt(variance)

    return std_dev


def calculate_median_absolute_deviation(signal):
    """
    Calculate the Median Absolute Deviation (MAD) of a 1D signal.
    
    MAD = median(|xi - median(X)|)
    
    Parameters:
        signal (array-like): A 1D array or list of numeric values.
        
    Returns:
        float: Median Absolute Deviation of the signal.
    """
    signal = np.asarray(signal)

    if signal.ndim != 1:
        raise ValueError("Input must be a 1D array.")
    
    if len(signal) == 0:
        return 0.0

    median = np.median(signal)
    abs_deviation = np.abs(signal - median)
    mad = np.median(abs_deviation)

    return mad

def calculate_form_factor(signal):
    """
    Calculate the form factor of a 1D signal.
    
    Form_factor = rms(X) / mean(abs(X))
    
    Parameters:
        signal (array-like): A 1D array or list of numeric values.
        
    Returns:
        float: Form factor of the signal.
    """
    signal = np.asarray(signal)

    if signal.ndim != 1:
        raise ValueError("Input must be a 1D array.")
    
    if len(signal) == 0:
        return 0.0

    rms = np.sqrt(np.mean(np.square(signal)))
    abs_mean = np.mean(np.abs(signal))

    form_factor = rms / abs_mean if abs_mean != 0 else 0.0
    return form_factor

def calculate_over_average_ratio(signal):
    """
    Calculate the over-average ratio of a 1D signal.
    
    Over_ave = N_over / N_total
    Where:
        N_over = number of times the signal crosses its mean value
        N_total = total number of data points
    
    Parameters:
        signal (array-like): A 1D array or list of numeric values.
        
    Returns:
        float: Over-average ratio.
    """
    signal = np.asarray(signal)

    if signal.ndim != 1:
        raise ValueError("Input must be a 1D array.")
    
    N_total = len(signal)
    if N_total < 2:
        return 0.0  # No crossing possible

    mean_val = np.mean(signal)
    centered = signal - mean_val

    # Count zero crossings in the centered signal
    crossings = np.where(np.diff(np.sign(centered)) != 0)[0]
    N_over = len(crossings)

    over_average_ratio = N_over / N_total
    return over_average_ratio

def compute_baseline_drift_feature_fast(signal, num_segments=3):
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
    
    drift_severity = abs(np.max(segment_medians) - np.min(segment_medians))

    return drift_severity

def extractor(signal, length=10): # [1008]
    # Example 1D signal (replace with your actual signal)
    # signal = np.array([1, 2, 2, 0, 3, 3, 0, 4, 5, 0, 0, 6])

    # Compute features
    empty_ratio = calculate_empty_ratio(signal)
    peak_intensity = calculate_peak_intensity(signal)
    slope, linearity = calculate_1stOrderSlope_linearity(signal)
    equal_value_ratio = calculate_equal_value_ratio(signal)
    std_dev = calculate_standard_deviation(signal)
    mad = calculate_median_absolute_deviation(signal)
    form_factor = calculate_form_factor(signal)
    over_average_ratio = calculate_over_average_ratio(signal)
    baseline_diff = compute_baseline_drift_feature_fast(signal)

    # Combine all features into a single feature vector (NumPy array)
    features = np.array([
        empty_ratio,
        peak_intensity,
        slope,
        linearity,
        equal_value_ratio,
        std_dev,
        mad,
        form_factor,
        over_average_ratio,
        baseline_diff

        # linearity,
        # peak_intensity,
        # equal_value_ratio,
        # over_average_ratio,
        # baseline_diff,
        # slope,
        # std_dev,
        # mad,
        # form_factor,
        # empty_ratio    
    ])

    # print("Extracted Features:")
    # print(features)
    return features[:length]


if __name__ == "__main__":
    # Example usage
    signal = [0, 1.5, 0, 3.2, 0, 0, 7.1]
    ratio = calculate_empty_ratio(signal)
    print(f"Empty ratio: {ratio:.2f}")

    # Example usage
    signal = [0.2, 0.4, 1.1, 0.9, 5.6, 0.3, 10.0, 0.2, 0.5]
    pi = calculate_peak_intensity(signal)
    print(f"Peak intensity: {pi:.3f}")

    # Example usage
    signal = [1.0, 1.2, 1.5, 2.0, 2.5, 3.1, 3.6, 4.0]
    lin_score = calculate_1stOrderSlope_linearity(signal)
    print(f"Linearity: {lin_score:.3f}")

    # Example usage
    signal = [1, 2, 2, 2, 3, 3, 4, 5, 5]
    eq_ratio = calculate_equal_value_ratio(signal)
    print(f"Equal value ratio: {eq_ratio:.3f}")

    # Example usage
    signal = [1.0, 2.0, 3.0, 4.0, 5.0]
    sd = calculate_standard_deviation(signal)
    print(f"Sample Standard Deviation: {sd:.3f}")

    # Example usage
    signal = [1.0, 2.0, 2.0, 2.0, 100.0]
    mad = calculate_median_absolute_deviation(signal)
    print(f"Median Absolute Deviation: {mad:.3f}")

    # Example usage
    signal = [1.0, -2.0, 3.0, -4.0, 5.0]
    ff = calculate_form_factor(signal)
    print(f"Form Factor: {ff:.3f}")

    # Example usage
    signal = [1, 2, 0, -1, -2, 1, 2, -1]
    ratio = calculate_over_average_ratio(signal)
    print(f"Over-average ratio: {ratio:.3f}")