"""
Advanced Single-Task Learning DenseNet+Transformer Model for Multi-Class Anomaly Detection
Enhanced architecture for handling class imbalance

Uses pre-split datasets: x_train_fold, y_train_fold, x_val_fold, y_val_fold, x_test, y_test

References:
- Huang et al., 2017: "Densely Connected Convolutional Networks" (DenseNet)
- Vaswani et al., 2017: "Attention Is All You Need" (Transformer)
- Dosovitskiy et al., 2021: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
- Wang et al., 2019: "Calibrating Deep Neural Networks using Focal Loss"
- Lin et al., 2017: "Focal Loss for Dense Object Detection"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, f1_score, precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
import os
import  time
from preprocessing_addFeatures import data_preprocessing
from collections import defaultdict
from sklearn.model_selection import train_test_split, StratifiedKFold
from unit.summary import summary, sum_parameters_by_layer

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# ==================== DATASET CONVERSION ====================

def convert_mtl_to_stl(y_mtl):
    """
    Convert MTL dataset (3 binary labels) to STL dataset (7 multi-class labels)
    
    Args:
        y_mtl: numpy array of shape (sample_num, 3) - MTL labels
               [Missing, Trend, Drift] with values 0 or 1
    
    Returns:
        y_stl: numpy array of shape (sample_num,) - STL labels (0-6)
        label_mapping: dict mapping MTL tuples to class IDs
    """
    label_mapping = {
        (0, 0, 0): 0,  # Normal
        (1, 0, 0): 1,  # Missing
        (0, 1, 0): 2,  # Trend
        (0, 0, 1): 3,  # Drift
        (1, 1, 0): 4,  # Missing+Trend
        (1, 0, 1): 5,  # Missing+Drift
        (0, 1, 1): 6,  # Trend+Drift
    }
    
    y_stl = np.zeros(len(y_mtl), dtype=np.int64)
    
    for idx, label in enumerate(y_mtl):
        label_tuple = tuple(label)
        
        if label_tuple not in label_mapping:
            raise ValueError(f"Unknown label combination at index {idx}: {label_tuple}")
        
        y_stl[idx] = label_mapping[label_tuple]
    
    return y_stl, label_mapping


def analyze_class_distribution(y_stl, split_name="Dataset"):
    """Analyze and print class distribution"""
    class_names = [
        "Normal [0,0,0]",
        "Missing [1,0,0]",
        "Trend [0,1,0]",
        "Drift [0,0,1]",
        "Missing+Trend [1,1,0]",
        "Missing+Drift [1,0,1]",
        "Trend+Drift [0,1,1]"
    ]
    
    counter = Counter(y_stl)
    total = len(y_stl)
    
    print(f"\nCLASS DISTRIBUTION - {split_name}")
    print("-" * 70)
    print(f"{'Class':<25} {'Quantity':<12} {'Percentage':<12}")
    print("-" * 70)
    
    for class_id in range(7):
        count = counter.get(class_id, 0)
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"{class_names[class_id]:<25} {count:<12} {percentage:>10.2f}%")
    
    print("-" * 70)
    print(f"{'TOTAL':<25} {total:<12} {100.00:>10.2f}%")
    
    return counter


def compute_class_weights(y_stl, num_classes=7):
    """
    Compute class weights for imbalanced dataset
    Uses inverse of class frequency to handle imbalance
    """
    counter = Counter(y_stl)
    weights = np.zeros(num_classes)
    
    total_samples = len(y_stl)
    
    for class_id in range(num_classes):
        count = counter.get(class_id, 0)
        if count > 0:
            weights[class_id] = total_samples / (num_classes * count)
        else:
            weights[class_id] = 1.0
    
    # Normalize weights
    weights = weights / weights.sum() * num_classes
    
    print("\n" + "="*70)
    print("CLASS WEIGHTS FOR HANDLING IMBALANCE")
    print("="*70)
    class_names = [
        "Normal", "Missing", "Trend", "Drift",
        "Missing+Trend", "Missing+Drift", "Trend+Drift"
    ]
    for class_id in range(num_classes):
        print(f"Class {class_id} ({class_names[class_id]:<15}): {weights[class_id]:.4f}")
    print("="*70)
    
    return torch.from_numpy(weights).float()


# ==================== CUSTOM LOSS FUNCTIONS ====================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Reference: Lin et al., 2017 "Focal Loss for Dense Object Detection"
    """
    def __init__(self, alpha=None, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, reduction='none')
    
    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != focal_loss.device:
                self.alpha = self.alpha.to(focal_loss.device)
            focal_loss = self.alpha[targets] * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss to prevent overconfident predictions
    Reference: Szegedy et al., 2016: "Rethinking the Inception Architecture"
    """
    def __init__(self, num_classes=7, smoothing=0.1, weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.weight = weight
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            
            if self.weight is not None:
                if self.weight.device != true_dist.device:
                    self.weight = self.weight.to(true_dist.device)
                true_dist = true_dist * self.weight.unsqueeze(0)
                true_dist = true_dist / true_dist.sum(dim=1, keepdim=True)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


# ==================== DATASET & DATALOADER ====================

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series anomaly detection"""
    
    def __init__(self, X, y, transform=None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        # Add channel dimension for CNN: (1008,) -> (1, 1008)
        x = x.unsqueeze(0)
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


# ==================== DENSENET BLOCK ====================

class DenseBlock(nn.Module):
    """
    Dense Block for DenseNet
    Reference: Huang et al., 2017
    """
    def __init__(self, in_channels, growth_rate=32, num_layers=4):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.growth_rate = growth_rate
        
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm1d(in_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels + i * growth_rate, growth_rate, 
                         kernel_size=3, padding=1, stride=1)
            ))
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        return torch.cat(features, 1)


class TransitionBlock(nn.Module):
    """Transition block for DenseNet"""
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        return self.block(x)


# ==================== POSITIONAL ENCODING ====================

class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer
    Reference: Vaswani et al., 2017
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# ==================== MULTI-HEAD ATTENTION ====================

class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention"""
    def __init__(self, d_model=256, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(self.d_k)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.out_linear(context)
        
        return output, attn_weights


# ==================== TRANSFORMER ENCODER LAYER ====================

class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer"""
    def __init__(self, d_model=256, num_heads=8, ff_dim=1024, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        # Multi-head attention
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization and residual connections
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_out, attn_weights = self.attn(x, x, x, mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # Feed-forward with residual connection
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        
        return x, attn_weights


# ==================== DENSENET + TRANSFORMER MODEL ====================

class DenseNetTransformerModel(nn.Module):
    """
    Advanced DenseNet+Transformer Architecture for Time Series Classification
    
    References:
    - Huang et al., 2017: DenseNet - Densely Connected Convolutional Networks
    - Vaswani et al., 2017: Attention Is All You Need
    
    Architecture:
    1. DenseNet blocks: Extract hierarchical temporal features
    2. Positional encoding: Add position information
    3. Transformer layers: Capture long-range dependencies with attention
    4. Classification head: Multi-class classification
    
    Key advantages for handling class imbalance:
    - DenseNet feature reuse reduces gradient vanishing
    - Attention mechanisms focus on important time steps
    - Layer normalization stabilizes training
    """
    
    def __init__(self, input_channels=1, sequence_length=1008, num_classes=7,
                 growth_rate=32, num_dense_layers=4, num_dense_blocks=3,
                 d_model=256, num_heads=8, num_transformer_layers=4,
                 ff_dim=1024, dropout_rate=0.3):
        super(DenseNetTransformerModel, self).__init__()
        
        self.sequence_length = sequence_length
        self.d_model = d_model
        
        # ============== DenseNet Feature Extraction ==============
        
        # Initial convolution
        self.conv_init = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        current_channels = 64
        
        # DenseNet blocks with transitions
        self.dense_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        
        for i in range(num_dense_blocks):
            # Dense block
            dense_block = DenseBlock(current_channels, growth_rate, num_dense_layers)
            self.dense_blocks.append(dense_block)
            
            # Update channel count after dense block
            current_channels = current_channels + num_dense_layers * growth_rate
            
            # Transition block (except for the last one)
            if i < num_dense_blocks - 1:
                transition_channels = current_channels // 2
                transition = TransitionBlock(current_channels, transition_channels)
                self.transitions.append(transition)
                current_channels = transition_channels
        
        # Final batch normalization
        self.final_bn = nn.BatchNorm1d(current_channels)
        
        # Calculate feature map size after DenseNet
        # Initial: 1008 -> 504 (conv stride 2) -> 252 (pool stride 2)
        # After each transition: divided by 2
        feature_map_size = 1008 // 4  # After initial conv and pool
        for i in range(num_dense_blocks - 1):
            feature_map_size = feature_map_size // 2  # Each transition halves
        
        # Projection to d_model
        self.feature_projection = nn.Conv1d(current_channels, d_model, kernel_size=1)
        
        # ============== Positional Encoding ==============
        self.pos_encoding = PositionalEncoding(d_model, max_len=feature_map_size + 10)
        
        # ============== Transformer Encoder ==============
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, ff_dim, dropout_rate)
            for _ in range(num_transformer_layers)
        ])
        
        # ============== Classification Head ==============
        self.dropout = nn.Dropout(dropout_rate)
        
        # Global average pooling will be done in forward
        self.fc1 = nn.Linear(d_model, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Store attention weights for visualization
        self.attention_weights = None
    
    def forward(self, x):
        # x shape: (batch_size, 1, 1008)
        
        # ============== DenseNet Feature Extraction ==============
        x = self.conv_init(x)  # (batch_size, 64, 252)
        
        for i, dense_block in enumerate(self.dense_blocks):
            x = dense_block(x)
            
            if i < len(self.transitions):
                x = self.transitions[i](x)
        
        x = self.final_bn(x)  # (batch_size, channels, feature_map_size)
        
        # ============== Projection to Transformer Dimension ==============
        x = self.feature_projection(x)  # (batch_size, d_model, feature_map_size)
        
        # Reshape for transformer: (batch_size, feature_map_size, d_model)
        x = x.transpose(1, 2)
        
        # ============== Positional Encoding ==============
        x = self.pos_encoding(x)
        
        # ============== Transformer Layers ==============
        attn_weights_list = []
        for transformer_layer in self.transformer_layers:
            x, attn_weights = transformer_layer(x)
            attn_weights_list.append(attn_weights)
        
        # Store attention weights for visualization
        self.attention_weights = attn_weights_list
        
        # ============== Global Average Pooling ==============
        x = torch.mean(x, dim=1)  # (batch_size, d_model)
        
        # ============== Classification Head ==============
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


# ==================== TRAINING & EVALUATION ====================

class ModelTrainer:
    """Handle model training, validation, and evaluation"""
    
    def __init__(self, model, device, class_weights=None, use_label_smoothing=True):
        self.model = model
        self.device = device
        self.class_weights = class_weights
        
        # Define loss functions
        if use_label_smoothing:
            self.criterion_primary = LabelSmoothingLoss(
                num_classes=7, smoothing=0.1, weight=class_weights
            )
        else:
            self.criterion_primary = nn.CrossEntropyLoss(weight=class_weights)
        
        self.criterion_focal = FocalLoss(gamma=2.0, alpha=class_weights, weight=None)
        
        self.optimizer = optim.AdamW(
            model.parameters(), lr=0.0005, weight_decay=1e-4, betas=(0.9, 0.999)
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=0.00001
        )
    
    def train_epoch(self, train_loader, use_focal_loss=False):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(x)
            
            # Compute loss
            if use_focal_loss:
                loss = self.criterion_focal(outputs, y)
            else:
                loss = self.criterion_primary(outputs, y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                outputs = self.model(x)
                loss = self.criterion_primary(outputs, y)
                
                total_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        return avg_loss, accuracy, f1, all_preds, all_labels
    
    def test(self, test_loader):
        """Test model and return comprehensive metrics"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                outputs = self.model(x)
                probs = torch.softmax(outputs, dim=1)
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        return all_preds, all_labels, all_probs


# ==================== MTL CONVERSION & EVALUATION ====================

def convert_stl_predictions_to_mtl(y_pred_stl):
    """Convert STL 7-class predictions back to MTL 3-binary predictions"""
    mapping = {
        0: [0, 0, 0],
        1: [1, 0, 0],
        2: [0, 1, 0],
        3: [0, 0, 1],
        4: [1, 1, 0],
        5: [1, 0, 1],
        6: [0, 1, 1]
    }
    
    y_pred_mtl = np.array([mapping[pred] for pred in y_pred_stl])
    return y_pred_mtl


def evaluate_mtl_metrics(y_true_mtl, y_pred_mtl):
    """Evaluate MTL-style metrics from converted predictions"""
    task_names = ["1_missing", "2_trend", "3_drift"]
    results = {}
    
    for task_idx, task_name in enumerate(task_names):
        y_true_task = y_true_mtl[:, task_idx]
        y_pred_task = y_pred_mtl[:, task_idx]
        
        class_report = classification_report(
            y_true_task, y_pred_task,
            target_names=['0', '1'],
            output_dict=True
        )
        
        if len(np.unique(y_true_task)) > 1:
            roc_auc = roc_auc_score(y_true_task, y_pred_task)
        else:
            roc_auc = np.nan
        
        results[task_name] = {
            'classification_report': class_report,
            'roc_auc': roc_auc,
            'accuracy': accuracy_score(y_true_task, y_pred_task),
            'precision': precision_score(y_true_task, y_pred_task, zero_division=0),
            'recall': recall_score(y_true_task, y_pred_task, zero_division=0),
            'f1': f1_score(y_true_task, y_pred_task, zero_division=0)
        }
    
    return results


def print_stl_metrics(y_true_stl, y_pred_stl):
    """Print STL evaluation metrics"""
    print("\n" + "="*80)
    print("STL MULTI-CLASS EVALUATION METRICS (7 Classes)")
    print("="*80)
    
    class_names = [
        "Normal", "Missing", "Trend", "Drift",
        "Missing+Trend", "Missing+Drift", "Trend+Drift"
    ]
    
    print(classification_report(
        y_true_stl, y_pred_stl,
        target_names=class_names
    ))
    
    accuracy = accuracy_score(y_true_stl, y_pred_stl)
    f1_weighted = f1_score(y_true_stl, y_pred_stl, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true_stl, y_pred_stl, average='macro', zero_division=0)
    
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    print("="*80)


def print_mtl_metrics(mtl_results):
    """Print MTL evaluation metrics in readable format"""
    print("\n" + "="*80)
    print("MTL-STYLE EVALUATION METRICS (for comparison with original MTL model)")
    print("="*80)
    
    for task_name, metrics in mtl_results.items():
        print(f"\nClassification Report for Task: {task_name}")
        print("-" * 70)
        report_dict = metrics['classification_report']
        
        print(f"{'':20} {'precision':>12} {'recall':>12} {'f1-score':>12} {'support':>10}")
        print("-" * 70)
        for class_label in ['0', '1']:
            if class_label in report_dict:
                stats = report_dict[class_label]
                print(f"{class_label:>20} {stats['precision']:>12.4f} {stats['recall']:>12.4f} "
                      f"{stats['f1-score']:>12.4f} {int(stats['support']):>10}")
        
        print("-" * 70)
        accuracy = metrics['accuracy']
        print(f"{'Accuracy':20} {accuracy:>12.4f}")
        
        if '0' in report_dict and '1' in report_dict:
            macro_precision = (report_dict['0']['precision'] + report_dict['1']['precision']) / 2
            macro_recall = (report_dict['0']['recall'] + report_dict['1']['recall']) / 2
            macro_f1 = (report_dict['0']['f1-score'] + report_dict['1']['f1-score']) / 2
            
            print(f"{'macro avg':20} {macro_precision:>12.4f} {macro_recall:>12.4f} "
                  f"{macro_f1:>12.4f}")
        
        if not np.isnan(metrics['roc_auc']):
            print(f"\nROC AUC: {metrics['roc_auc']:.4f}")
        else:
            print("\nROC AUC: N/A (only one class present)")
    
    print("="*80)


# ==================== VISUALIZATION ====================

def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('DenseNet+Transformer - Training History - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[1].plot(history['val_f1'], label='Val F1-Score', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('DenseNet+Transformer - Training History - Accuracy & F1-Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('densenet_transformer_training_history.png', dpi=300, bbox_inches='tight')
    print("\n✓ Training history plot saved as 'densenet_transformer_training_history.png'")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    class_names = [
        "Normal", "Missing", "Trend", "Drift",
        "Missing+Trend", "Missing+Drift", "Trend+Drift"
    ]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names, ax=ax, cbar_kws={'label': 'Count'})
    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix plot saved as '{title.lower().replace(' ', '_')}.png'")
    plt.close()


# ==================== MAIN EXECUTION ====================

def main(x_train_fold, y_train_fold, x_val_fold, y_val_fold, x_test, y_test,
         num_epochs=100, batch_size=32, use_label_smoothing=True):
    """
    Main training pipeline using pre-split datasets with DenseNet+Transformer model
    
    Args:
        x_train_fold: numpy array of shape (train_num, 1008)
        y_train_fold: numpy array of shape (train_num, 3)
        x_val_fold: numpy array of shape (val_num, 1008)
        y_val_fold: numpy array of shape (val_num, 3)
        x_test: numpy array of shape (test_num, 1008)
        y_test: numpy array of shape (test_num, 3)
        num_epochs: number of training epochs
        batch_size: batch size for training
        use_label_smoothing: whether to use label smoothing loss
    """
    
    print("\n" + "="*80)
    print("ADVANCED DENSENET+TRANSFORMER MODEL FOR ANOMALY DETECTION")
    print("Using Pre-Split Datasets")
    print("="*80)
    
    # ============== STEP 1: DATA CONVERSION ==============
    print("\n[STEP 1] Converting MTL Labels to STL Labels...")
    
    y_train_stl, label_mapping = convert_mtl_to_stl(y_train_fold)
    y_val_stl, _ = convert_mtl_to_stl(y_val_fold)
    y_test_stl, _ = convert_mtl_to_stl(y_test)
    
    print(f"✓ Conversion successful!")
    print(f"  - Label mapping: {label_mapping}")
    
    # ============== STEP 2: ANALYZE CLASS DISTRIBUTION ==============
    print("\n[STEP 2] Analyzing Class Distribution...")
    print("="*70)
    
    analyze_class_distribution(y_train_stl, "Training Set")
    analyze_class_distribution(y_val_stl, "Validation Set")
    analyze_class_distribution(y_test_stl, "Test Set")
    
    # ============== STEP 3: DATA PREPROCESSING ==============
    print("\n[STEP 3] Data Preprocessing...")
    
    scaler = StandardScaler()
    x_train_fold = scaler.fit_transform(x_train_fold)
    x_val_fold = scaler.transform(x_val_fold)
    x_test = scaler.transform(x_test)
    
    print("✓ Input signals standardized (zero mean, unit variance)")
    print(f"  - Training set: {x_train_fold.shape}")
    print(f"  - Validation set: {x_val_fold.shape}")
    print(f"  - Test set: {x_test.shape}")
    
    # Create dataloaders
    train_dataset = TimeSeriesDataset(x_train_fold, y_train_stl)
    val_dataset = TimeSeriesDataset(x_val_fold, y_val_stl)
    test_dataset = TimeSeriesDataset(x_test, y_test_stl)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✓ Dataloaders created")
    
    # ============== STEP 4: COMPUTE CLASS WEIGHTS ==============
    print("\n[STEP 4] Computing Class Weights for Imbalance Handling...")
    class_weights = compute_class_weights(y_train_stl, num_classes=7)
    
    # ============== STEP 5: CREATE MODEL ==============
    print("\n[STEP 5] Creating Advanced DenseNet+Transformer Model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    model = DenseNetTransformerModel(
        input_channels=1,
        sequence_length=1008,
        num_classes=7,
        growth_rate=32,
        num_dense_layers=4,
        num_dense_blocks=3,
        d_model=256,
        num_heads=8,
        num_transformer_layers=4,
        ff_dim=1024,
        dropout_rate=0.3
    )
    model = model.to(device)
    
    # Print model summary
    sum_parameters_by_layer(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"\nModel Architecture Highlights:")
    print(f"  - DenseNet Blocks: 3 (with growth rate {32})")
    print(f"  - Dense Layers per Block: 4")
    print(f"  - Transformer Encoder Layers: 4")
    print(f"  - Attention Heads: 8")
    print(f"  - Model Dimension (d_model): 256")
    
    # ============== STEP 6: TRAINING ==============
    print("\n[STEP 6] Training Model with Label Smoothing and Focal Loss...")
    print("="*70)
    
    trainer = ModelTrainer(model, device, class_weights=class_weights,
                          use_label_smoothing=use_label_smoothing)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': []
    }
    
    best_val_f1 = 0.0
    patience = 500
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Train - using label smoothing
        train_loss = trainer.train_epoch(train_loader, use_focal_loss=False)
        
        # Validate
        val_loss, val_acc, val_f1, _, _ = trainer.validate(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Learning rate scheduling
        trainer.scheduler.step()
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), 'best_densenet_transformer_model.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:>3}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_acc:.4f}, "
                  f"Val F1: {val_f1:.4f}")
        
        if patience_counter >= patience:
            print(f"\n✓ Early stopping triggered at epoch {epoch+1}")
            break
    
    print("✓ Training completed!")
    
    # ============== STEP 7: TESTING ==============
    print("\n[STEP 7] Testing Model...")
    
    model.load_state_dict(torch.load('best_densenet_transformer_model.pth'))
    
    y_pred_stl, y_true_stl_test, y_probs = trainer.test(test_loader)
    
    print("✓ Test predictions generated")
    
    # ============== STEP 8: STL EVALUATION ==============
    print_stl_metrics(y_true_stl_test, y_pred_stl)
    
    # ============== STEP 9: MTL CONVERSION & EVALUATION ==============
    print("\n[STEP 9] MTL-Style Evaluation Metrics")
    
    y_pred_mtl = convert_stl_predictions_to_mtl(y_pred_stl)
    
    mtl_results = evaluate_mtl_metrics(y_test, y_pred_mtl)
    print_mtl_metrics(mtl_results)
    
    # ============== STEP 10: VISUALIZATION ==============
    print("\n[STEP 10] Generating Visualizations...")
    
    plot_training_history(history)
    plot_confusion_matrix(y_true_stl_test, y_pred_stl, 
                         title='DenseNet+Transformer Confusion Matrix (7 Classes)')
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated files:")
    print("  - best_densenet_transformer_model.pth: Trained model weights")
    print("  - densenet_transformer_training_history.png: Training curves")
    print("  - densenet+transformer_confusion_matrix.png: Confusion matrix")
    print("="*80 + "\n")
    
    return model, history, mtl_results, y_pred_stl, y_true_stl_test


if __name__ == "__main__":
    # ============== LOAD YOUR PRE-SPLIT DATASETS ==============
    print("\n" + "="*80)
    print("LOADING PRE-SPLIT DATASETS")
    print("="*80)
    
    # # Load your pre-split MTL datasets
    # x_train_fold = np.load('x_train_fold.npy')      # (train_num, 1008)
    # y_train_fold = np.load('y_train_fold.npy')      # (train_num, 3)
    
    # x_val_fold = np.load('x_val_fold.npy')          # (val_num, 1008)
    # y_val_fold = np.load('y_val_fold.npy')          # (val_num, 3)
    
    # x_test = np.load('x_test.npy')                  # (test_num, 1008)
    # y_test = np.load('y_test.npy')                  # (test_num, 3)
    


    # 生成模型保存目录（使用当前时间戳区分不同实验）
    save_path = 'saved_models/PLE_mode_{}'.format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    os.makedirs(save_path, exist_ok=True)
    print(f"\nModel saved to {save_path}")

    # 数据预处理：返回 x,y 等，其中此处只取全部样本与标签
    # balanced=False 不做类平衡；normal_class=False 不单独指定正常类；method='test' 指定预处理模式
    _, _, x_all, y_all, _ = data_preprocessing(balanced=False, platform='pytorch', normal_class=False, method='test')
    
    # 数据打乱（确保随机性）
    np.random.seed(8)
    indices = np.arange(x_all.shape[0])
    np.random.shuffle(indices)
    x_all = x_all[indices]
    y_all = y_all[indices]

    # 划分训练验证整体集合与测试集合（此处测试集后面未用到，只训练验证折）
    x_train_val = x_all[:1436,:] # train+val共占比80%， 1436/1794=0.8
    y_train_val = y_all[:1436,:] # 80%
    x_test = x_all[1436:,:] # test占比20%
    y_test = y_all[1436:,:] # 20%

    # 使用第一个任务的标签作为分层依据（保证折划分类别均衡）
    stratify_labels = y_train_val[:, 0]
    skf = StratifiedKFold(n_splits=4, shuffle=False)

    # 存储所有折的预测与真实标签（用于最终汇总）
    all_y_true = defaultdict(list)
    all_y_pred = defaultdict(list)
    all_y_prod = defaultdict(list)
    report_logs = []  # 保存各折分类报告的文本

    # 交叉验证循环（这里只训练第4折，其他折直接跳过）
    for fold, (train_idx, val_idx) in enumerate(skf.split(x_train_val, stratify_labels), 1):
        if fold == 4:
            print(f"\n=== Fold {fold} ===")
            # 当前折的训练与验证数据
            x_train_fold = x_train_val[train_idx]
            y_train_fold = y_train_val[train_idx]
            x_val_fold = x_train_val[val_idx]
            y_val_fold = y_train_val[val_idx]

            # 训练当前折模型并返回该折的预测结果（真实标签、类别预测、概率预测）
            print(f"✓ Data loaded successfully!")
            print(f"  - x_train_fold shape: {x_train_fold.shape}")
            print(f"  - y_train_fold shape: {y_train_fold.shape}")
            print(f"  - x_val_fold shape: {x_val_fold.shape}")
            print(f"  - y_val_fold shape: {y_val_fold.shape}")
            print(f"  - x_test shape: {x_test.shape}")
            print(f"  - y_test shape: {y_test.shape}")
            print("="*80)
            
            # Run main training pipeline
            model, history, mtl_results, y_pred, y_true = main(
                x_train_fold=x_train_fold[:, :1008],  # 只取信号部分作为输入
                y_train_fold=y_train_fold,
                x_val_fold=x_val_fold[:, :1008],
                y_val_fold=y_val_fold,
                x_test=x_test[:, :1008],
                y_test=y_test,
                num_epochs=1000,
                batch_size=128
            )
        else:
            continue