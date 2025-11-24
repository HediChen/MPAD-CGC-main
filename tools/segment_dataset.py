import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
import sys
sys.path.append('./')
from preprocessing_addFeatures import data_preprocessing
import  time
import matplotlib.pyplot as plt
import os

def save_data_split_indices(x_all, y_all, split_point, fold_to_save=4, save_path="data_indices_fold4.csv"):
    """
    Shuffle, split, and save the global indices of training, validation, and test data for a specific fold.

    Parameters:
    - x_all (np.ndarray): Full input features.
    - y_all (np.ndarray): Full labels.
    - split_point (int): Index at which to split training/validation and test data.
    - fold_to_save (int): The fold number to save (e.g., 4).
    - save_path (str): Path to save the resulting CSV file.

    Returns:
    - train_indices (list of int): Global training indices.
    - val_indices (list of int): Global validation indices.
    - test_indices (list of int): Global test indices.
    """

    # Shuffle the full dataset
    np.random.seed(8)
    indices = np.arange(x_all.shape[0])
    np.random.shuffle(indices)
    x_all = x_all[indices]
    y_all = y_all[indices]

    # Split into train/val and test
    x_train_val = x_all[:split_point, :]
    y_train_val = y_all[:split_point, :]
    train_val_indices = indices[:split_point]  # mapping back to global indices
    test_indices = indices[split_point:]

    # Stratify based on the first task (e.g., "Missing")
    stratify_labels = y_train_val[:, 0]
    skf = StratifiedKFold(n_splits=4, shuffle=False)

    for fold, (train_idx, val_idx) in enumerate(skf.split(x_train_val, stratify_labels), 1):
        if fold == fold_to_save:
            print(f"Saving indices for Fold {fold_to_save}...")

            # Map local fold indices back to global indices
            train_indices = train_val_indices[train_idx]
            val_indices = train_val_indices[val_idx]

            # Pad lists for CSV formatting
            max_len = max(len(train_indices), len(val_indices), len(test_indices))
            pad = lambda arr: list(arr) + [''] * (max_len - len(arr))

            df_indices = pd.DataFrame({
                "train_indices": pad(train_indices),
                "val_indices": pad(val_indices),
                "test_indices": pad(test_indices)
            })

            # Save to CSV
            df_indices.to_csv(save_path, index=False)
            print(f"Indices saved to {save_path}")

            # Return as lists
            return list(train_indices), list(val_indices), list(test_indices)

    # If fold not found
    raise ValueError(f"Fold {fold_to_save} not found.")

def plot_signal_by_index(index, x_data, y_data, save_dir=None):
    """
    Plot a 1D signal by its index along with its label. Optionally save the plot to a directory.

    Parameters:
    - index (int): The index of the sample to plot.
    - x_data (np.ndarray): 2D array of shape (N, L), where each row is a 1D signal.
    - y_data (np.ndarray): 2D array of shape (N, 3), where each row is the corresponding label.
    - save_dir (str or None): Directory to save the plot. If None, the plot is only displayed.
    """
    # Validate index
    if index < 0 or index >= x_data.shape[0]:
        raise IndexError(f"Index {index} is out of range for x_data with shape {x_data.shape}.")

    signal = x_data[index]
    label = y_data[index]
    label_str = "_".join(map(str, label.astype(int)))  # e.g., "0_1_0"

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(signal[:1008], color='royalblue')
    plt.title(f"Index: {index} | Label: {label.tolist()}")
    plt.xlabel("Time Step")
    plt.ylabel("Signal Amplitude")
    plt.grid(True)
    plt.tight_layout()

    # Save plot if save_dir is specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"signal_index_{index}_label_{label_str}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")

    # plt.show()

if __name__ == '__main__':
    save_path = 'saved_models/PLE_mode_{}'.format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    os.makedirs(save_path, exist_ok=True)
    print(f"\nModel saved to {save_path}")
    _, _, x_all, y_all, _ = data_preprocessing(balanced=False, platform='pytorch', normal_class=False, method='test')
    
    train_lst, val_lst, test_lst = save_data_split_indices(x_all, y_all, 1436, fold_to_save=4, save_path="data_indices_fold4.csv")

    save_dir = 'data/anomaly/fullData_Feb2Oct_20250621/test_data'
    for i in test_lst:
        plot_signal_by_index(i, x_all, y_all, save_dir)