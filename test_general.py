import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
# from pytorchModels.cgc_dnn import CGC
# from pytorchModels.cgc_cnn import CGC
# from pytorchModels.cgc_Inception import CGC
# from pytorchModels.hardShare_Inception import HardShare
# from pytorchModels.mmoe_Inception import MMoe
from pytorchModels.ple_Inception_Features_analysis_train import PLE
# from pytorchModels.moe_Inception import Moe
from preprocessing_addFeatures import data_preprocessing
from torch.utils.data import DataLoader, TensorDataset



def plot_confusion_matrix(cm, task_name, class_names=['0', '1'], save_path=None):
    plt.figure(figsize=(3, 3))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {task_name}')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f"{cm[i, j]}", ha='center', va='center',
                 color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_misclassified_signal(signal, pred, true, task, index, save_path=None):
    plt.figure(figsize=(10, 3))
    plt.plot(signal, label='Signal')
    plt.title(f'Task: {task} | Index: {index} | Predicted: {pred} | True: {true}')
    plt.xlabel('Timestep')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def evaluate_predictions(y_true, y_pred, y_prob, label, x_val, val_indices, results_dir, label_prefix="original", threshold='None'):
    report = classification_report(y_true, y_pred, target_names=['0', '1'], digits=4)
    cm = confusion_matrix(y_true, y_pred)

    with open(os.path.join(results_dir, f'{label_prefix}_metrics.txt'), 'a') as f:
        f.write(f"\n=== Task: {label} ===\n")
        f.write(f"threshold: {threshold}\n")
        f.write(report + '\n')
        try:
            auc = roc_auc_score(y_true, y_prob)
            f.write(f"ROC AUC: {auc:.4f}\n")
        except:
            f.write("ROC AUC: Undefined (only one class present in y_true)\n")

    cm_path = os.path.join(results_dir, f'{label_prefix}_confusion_matrix_{label}.png')
    plot_confusion_matrix(cm, label, save_path=cm_path)

    task_dir = os.path.join(results_dir, f'misclassified_{label_prefix}_{label}')
    os.makedirs(task_dir, exist_ok=True)
    for idx, (pred, true_label) in enumerate(zip(y_pred, y_true)):
        if pred != true_label:
            raw_idx = val_indices[idx]  # recover original index
            save_path = os.path.join(task_dir, f'raw_index_{raw_idx}_pred_{pred}_true_{true_label}.png')
            plot_misclassified_signal(x_val[idx][:1008], pred, true_label, label, raw_idx, save_path)

def test(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train, y_train, x_all, y_all, all_indices = data_preprocessing(balanced=False, platform='pytorch', normal_class=False, method = 'test')
    
    np.random.seed(8)
    indices = np.arange(x_all.shape[0])
    np.random.shuffle(indices)
    x_all = x_all[indices]
    y_all = y_all[indices]
    all_indices = all_indices[indices]

    x_test = x_all[1436:,:]
    y_test = y_all[1436:,:]
    test_indices = all_indices[1436:]



    # for CGC with inception and CNN experts, HardShare, MoE, MMoE, CGC inception
    x_val_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
    y_val_tensor = torch.tensor(y_test, dtype=torch.float32)
    y_val_tensor = torch.eye(2)[y_val_tensor.long(), :]
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
    num_features = x_val_tensor[:,:,1008:].shape[-1]

    model = PLE(inputs_dim=num_features,
                 labels_dict={
                    '1_missing': 2,
                    '2_trend': 2,
                    '3_drift': 2,
                 },
                 dnn_dropout=0.2,
                 num_shared_experts=9,
                 num_task_experts=6,
                 expert_hidden_units=[128],
                 tower_hidden_units=[128, 64, 32],
                 device='cuda')

    # model = CGC(inputs_dim=num_features,
    #              labels_dict={
    #                 '1_missing': 2,
    #                 '2_trend': 2,
    #                 '3_drift': 2,
    #              },
    #              dnn_dropout=0.2,
    #              num_shared_experts=4,
    #              num_task_experts=1,
    #              expert_hidden_units=[128],
    #              tower_hidden_units=[128, 64, 32],
    #              device='cuda')

    # model = HardShare(inputs_dim=num_features,
    #             labels_dict={
    #                 '1_missing': 2,
    #                 '2_trend': 2,
    #                 '3_drift': 2,
    #             },
    #             dnn_dropout=0.2,
    #             num_shared_experts=1,
    #             expert_hidden_units=[128],
    #             tower_hidden_units=[128, 64, 32],
    #             device='cuda')

    # model = MMoe(inputs_dim=num_features,
    #             labels_dict={
    #                 '1_missing': 2,
    #                 '2_trend': 2,
    #                 '3_drift': 2,
    #             },
    #             dnn_dropout=0.2,
    #             num_shared_experts=4,
    #             expert_hidden_units=[128],
    #             tower_hidden_units=[128, 64, 32],
    #             device='cuda')

    # model = Moe(inputs_dim=num_features,
    #             labels_dict={
    #                 '1_missing': 2,
    #                 '2_trend': 2,
    #                 '3_drift': 2,
    #             },
    #             dnn_dropout=0.2,
    #             num_shared_experts=4,
    #             expert_hidden_units=[128],
    #             tower_hidden_units=[128, 64, 32],
    #             device='cuda')
    
    # for CGC DNN experts
    # x_val_tensor = torch.tensor(x_test, dtype=torch.float32)
    # y_val_tensor = torch.tensor(y_test, dtype=torch.float32)
    # y_val_tensor = torch.eye(2)[y_val_tensor.long(), :]
    # val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    # val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
    # num_features = 1008
    # model = CGC(inputs_dim=num_features,
    #             labels_dict={
    #                 '1_missing': 2,
    #                 '2_trend': 2,
    #                 '3_drift': 2,
    #             },
    #             dnn_dropout=0.2,
    #             num_shared_experts=4,
    #             num_task_experts=1,
    #             expert_hidden_units=[128],
    #             tower_hidden_units=[128, 64, 32],
    #             device='cuda')

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    dict_classes = {0: '1_missing', 1: '2_trend', 2: '3_drift'}
    y_true, y_prob, y_pred_orig = {}, {}, {}

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(x_batch)

            for i, task in dict_classes.items():
                probs = torch.softmax(outputs[task], dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(outputs[task], dim=1).cpu().numpy()
                labels = torch.argmax(y_batch[:, i], dim=1).cpu().numpy()

                y_true.setdefault(task, []).extend(labels)
                y_prob.setdefault(task, []).extend(probs)
                y_pred_orig.setdefault(task, []).extend(preds)

    original_dir = os.path.join(RESULTS_DIR, 'original_argmax')
    os.makedirs(original_dir, exist_ok=True)
    for task in dict_classes.values():
        evaluate_predictions(y_true[task], y_pred_orig[task], y_prob[task], task, x_test, test_indices, original_dir, label_prefix="original")

    print(f"\n✅ Evaluation complete - {original_dir}\n")

if __name__ == '__main__':
    model_path = 'saved_models\PLE_mode_2025-11-24-15-20-20\model_2025-11-24-15-24-31_fold_4.pth'
    # Create output folder
    RESULTS_DIR = 'saved_models\PLE_mode_2025-11-24-15-20-20\model_2025-11-24-15-24-31_fold_4'
    os.makedirs(RESULTS_DIR, exist_ok=True)
    test(model_path)