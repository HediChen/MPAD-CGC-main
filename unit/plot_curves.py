from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

import os
import time

def plot_training_curves(train_loss_values, test_loss_values,
                         train_accuracy_values, test_accuracy_values, dict_classes, save_path=None):
    train_loss_values = np.array(train_loss_values)
    test_loss_values = np.array(test_loss_values)
    train_accuracy_values = np.array(train_accuracy_values)
    test_accuracy_values = np.array(test_accuracy_values)

    num_tasks = len(dict_classes)

    fig, axes = plt.subplots(num_tasks, 2, figsize=(14, 3 * num_tasks))
    fig.suptitle('Training Curves per Task', fontsize=16)

    for i, key in enumerate(dict_classes.keys()):
        task_name = dict_classes[key]

        # Loss subplot
        axes[i, 0].plot(train_loss_values[:, i], label='Train Loss')
        axes[i, 0].plot(test_loss_values[:, i], label='Val Loss')
        axes[i, 0].set_title(f'{task_name} - Loss')
        axes[i, 0].set_xlabel('Epoch')
        axes[i, 0].set_ylabel('Loss')
        axes[i, 0].legend()
        axes[i, 0].grid(True)

        # Accuracy subplot
        axes[i, 1].plot(train_accuracy_values[:, i], label='Train Accuracy')
        axes[i, 1].plot(test_accuracy_values[:, i], label='Val Accuracy')
        axes[i, 1].set_title(f'{task_name} - Accuracy')
        axes[i, 1].set_xlabel('Epoch')
        axes[i, 1].set_ylabel('Accuracy (%)')
        axes[i, 1].legend()
        axes[i, 1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path == None:
        plt.show()
    else:
        # Save the figure
        os.makedirs(save_path+'/plots', exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        fig_path = os.path.join(save_path+'/plots', f'training_curves_{timestamp}.png')
        plt.savefig(fig_path)
        print(f"Saved training curves plot to {fig_path}")
        plt.close()


def plot_combined_task_curves(train_loss_values, test_loss_values,
                              train_accuracy_values, test_accuracy_values, dict_classes, save_path=None):
    train_loss_values = np.array(train_loss_values)
    test_loss_values = np.array(test_loss_values)
    train_accuracy_values = np.array(train_accuracy_values)
    test_accuracy_values = np.array(test_accuracy_values)

    epochs = np.arange(train_loss_values.shape[0])

    # Set color palette
    colors = plt.get_cmap('tab10')

    os.makedirs('plots', exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    # Plot all loss curves
    plt.figure(figsize=(10, 6))
    for i, key in enumerate(dict_classes.keys()):
        color = colors(i)
        plt.plot(epochs, train_loss_values[:, i], linestyle='--', color=color, label=f'Train {dict_classes[key]}')
        plt.plot(epochs, test_loss_values[:, i], linestyle='-', color=color, label=f'Val {dict_classes[key]}')
    plt.title('Loss Curves for All Tasks')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path == None:
        plt.show()
    else:
        loss_path = os.path.join(save_path+'/plots', f'combined_loss_curves_{timestamp}.png')
        plt.savefig(loss_path)
        print(f"Saved combined loss curves to {loss_path}")
        # plt.show()

    # Plot all accuracy curves
    plt.figure(figsize=(10, 6))
    for i, key in enumerate(dict_classes.keys()):
        color = colors(i)
        plt.plot(epochs, train_accuracy_values[:, i], linestyle='--', color=color, label=f'Train {dict_classes[key]}')
        plt.plot(epochs, test_accuracy_values[:, i], linestyle='-', color=color, label=f'Val {dict_classes[key]}')
    plt.title('Accuracy Curves for All Tasks')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


    if save_path == None:
        plt.show()
    else:
        acc_path = os.path.join(save_path+'/plots', f'combined_accuracy_curves_{timestamp}.png')
        plt.savefig(acc_path)
        print(f"Saved combined accuracy curves to {acc_path}")
    # plt.show()

def plot_total_loss_curve(train_loss_total, test_loss_total, save_path, fold_id):
    """
    Plots and saves the total loss curves for training and testing over epochs.

    Args:
        train_loss_total (list of float): List of total training losses per epoch.
        test_loss_total (list of float): List of total testing losses per epoch.
        save_path (str): Directory where the plot will be saved.
        fold_id (int): The ID of the current cross-validation fold.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_total, label='Train Loss', color='blue', linewidth=2)
    plt.plot(test_loss_total, label='Val Loss', color='red', linewidth=2)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Total Loss', fontsize=12)
    plt.title('Training and Testing Total Loss Over Epochs', fontsize=14)
    plt.legend()
    plt.grid(True)

    os.makedirs(save_path, exist_ok=True)
    plot_file = os.path.join(save_path+'/plots', f'total_loss_curve_fold{fold_id}.png')
    plt.savefig(plot_file, dpi=300)
    plt.close()

    print(f"Total loss curve saved to {plot_file}")
