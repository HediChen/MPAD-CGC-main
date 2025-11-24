import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
sys.path.append('./')
import os
import pandas as pd

def plot_metric_heatmap_and_signals(metric, shape=[4,8], heatmap_title="Weight Matrix Heatmap",
                                    curve_title="Signal Curves", save_dir=None):
    """
    Plot a 3D metric (1, 32, 1008) as:
    1. A heatmap of shape (32, 1008)
    2. 32 individual signal plots in a 4x8 grid

    Parameters:
    - metric (np.ndarray): A 3D array of shape (1, 32, 1008)
    - heatmap_title (str): Title for the heatmap
    - curve_title (str): Title for the 4x8 signal subplot grid
    - save_dir (str or None): Directory to save figures; if None, does not save
    """
    # if metric.shape != (1, 32, 1008):
    #     raise ValueError(f"Expected metric shape (1, 32, 1008), got {metric.shape}")
    
    matrix = metric[0]  # shape: (32, 1008)

    # === Plot Heatmap ===
    plt.figure(figsize=(14, 6))
    sns.heatmap(matrix, cmap="Blues", cbar=True)
    plt.title(heatmap_title)
    plt.xlabel("Time Step")
    plt.ylabel("Channel")
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        heatmap_path = os.path.join(save_dir, "metric_heatmap.png")
        plt.savefig(heatmap_path, dpi=300)
        print(f"Heatmap saved to: {heatmap_path}")

    plt.show()

    # === Plot 4x8 Grid of Signal Curves ===
    fig, axes = plt.subplots(shape[0], shape[1], figsize=(shape[1] * 2, shape[0] * 2), sharex=True, sharey=True)
    axes = axes.flatten()

    for i in range(shape[0] * shape[1]):
        axes[i].plot(matrix[i], color='steelblue', linewidth=0.8)
        axes[i].set_title(f"Channel {i}", fontsize=8)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_xlim([0, matrix.shape[1]])
        axes[i].set_xlim([0, matrix.shape[1]])

        # === Save all signals to CSV ===
        if save_dir:
        # Create a DataFrame with each channel as a column
            df_dict = {f"Channel_{i}": matrix[i] for i in range(matrix.shape[0])}
            df = pd.DataFrame(df_dict)
            csv_path = os.path.join(save_dir, "extracted_features.csv")
            df.to_csv(csv_path, index=False)
            print(f"Signals saved to CSV: {csv_path}")

    # Remove unused axes (if any)
    for j in range(shape[0] * shape[1], len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(curve_title, fontsize=14)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    if save_dir:
        curves_path = os.path.join(save_dir, "extracted_features.png")
        plt.savefig(curves_path, dpi=300)
        print(f"Signal plots saved to: {curves_path}")

    plt.show()

if __name__ == "__main__":
    # Example usage
    metric = np.random.rand(1, 32, 1008)  # Replace with your actual metric data
    plot_metric_heatmap_and_signals(metric, save_dir="discussion/figures/metric_vis")