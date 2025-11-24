import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
sys.path.append('./')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
plt.rcParams['font.family'] = 'Times New Roman' 

def plot_feature_weights_heatmap(weights=None, color='Reds', title="Feature Weights Heatmap", save_path=None):
    """
    Plot a heatmap for a 1D array of 9 feature weights.

    Parameters:
    - weights (list, np.ndarray, or None): Array of shape (9,) containing the weights.
      If None, prompts user to manually input from terminal.
    - title (str): Title of the heatmap.
    - save_path (str or None): If provided, saves the heatmap to this file path.
    """
    save_path='discussion/statistical_features/feature_weights_heatmap_{}.png'.format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    feature_names = [
        "Empty Ratio",
        "Peak Intensity",
        "Slope",
        "Linearity",
        "Equal Value Ratio",
        "Standard Deviation",
        "MAD",
        "Form Factor",
        "Over Average Ratio"
    ]

    # Manual input if weights not provided
    if weights is None:
        user_input = input("Enter 9 weights (e.g., [0.18 0.09 0.07 ...]):\n")
        color = input("Enter heatmap's color (e.g., 'Reds', 'Purples', 'Greens'):\n")
        try:
            weights = np.fromstring(user_input.strip('[]'), sep=' ')
        except Exception as e:
            raise ValueError("Invalid input format. Please enter 9 space-separated numbers inside brackets.") from e

    weights = np.array(weights)

    if weights.shape != (9,):
        weights = weights.reshape(-1)
        # raise ValueError(f"weights must be a 1D array of 9 elements. Got shape {weights.shape}")

    # Plotting
    weights_2d = weights.reshape(1, -1)  # shape (1, 9) for heatmap

    plt.figure(figsize=(10, 3))  # slightly taller to fit labels
    sns.heatmap(weights_2d, annot=True, fmt=".3f", cmap=color, xticklabels=feature_names,
                yticklabels=False, cbar=True)

    plt.title(title, pad=10)
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.4)  # ⬅️ space for rotated labels

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Heatmap saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    # Example usage
    plot_feature_weights_heatmap(save_path='discussion/statistical_features/feature_weights_heatmap.png')