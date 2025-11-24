import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('./')
plt.rcParams['font.family'] = 'Times New Roman'  # or 'Times New Roman', 'SimHei', 'Arial', etc.

def plot_loss_comparison(root_dir="Loss_function_comparision",
                         csv_filename="loss_log.csv",
                         save_filename="loss_comparison.png",
                         ymin=None,
                         ymax=None):
    """
    Plot train and test total loss curves for different loss functions.

    Parameters:
    - root_dir (str): Root directory containing subfolders for each loss function.
    - csv_filename (str): Name of the CSV file containing loss data in each subfolder.
    - save_filename (str): Filename to save the plot (stored in root_dir).
    - ymin (float or None): Minimum y-axis value. If None, it will be auto-scaled.
    - ymax (float or None): Maximum y-axis value. If None, it will be auto-scaled.
    """

    # Define loss function folders and display names
    loss_folders = {
        "0-BCEloss_PLE_Shared1_Task1_mode_2025-07-02-11-15-57": "BCE Loss",
        "1-Original_FocalLoss_PLE_Shared1_Task1_mode_2025-07-02-11-07-58": "Focal Loss",
        "2-BatchLevel_FocalLoss_PLE_Shared1_Task1_2025-07-02-11-00-45": "Batch-level Focal Loss (BLFL)"
    }

    # Colors for each loss function
    colors = {
        "BCE Loss": "blue",
        "Focal Loss": "green",
        "Batch-level Focal Loss (BLFL)": "red"
    }

    # Initialize the plot
    plt.figure(figsize=(8, 5))

    # Loop through each folder and plot its losses
    for folder, label in loss_folders.items():
        csv_path = os.path.join(root_dir, folder, csv_filename)
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found. Skipping.")
            continue

        df = pd.read_csv(csv_path)

        # Plot train and test total loss
        plt.plot(df["Epoch"], df["Train_total_loss"], label=f"{label} - Train", color=colors[label], linestyle='-')
        plt.plot(df["Epoch"], df["Test_total_loss"], label=f"{label} - Val", color=colors[label], linestyle='--')

    # Customize plot
    plt.title("Training and Validation Loss Comparison Across Loss Functions")
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")

    # Set y-axis limits if provided
    if ymin is not None or ymax is not None:
        plt.ylim(bottom=ymin, top=ymax)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(root_dir, save_filename)
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")

    # Show plot
    plt.show()


if __name__ == "__main__":
    root_dir = "saved_models/2-Loss_function_comparision"
    csv_filename = "loss_values_fold4.csv"

    # Example with Y-axis limits set
    plot_loss_comparison(
        root_dir=root_dir,
        csv_filename=csv_filename,
        save_filename="loss_comparison_gridding.png",
        ymin=None,  # You can adjust this
        ymax=None   # Or set to None for auto-scaling
    )


