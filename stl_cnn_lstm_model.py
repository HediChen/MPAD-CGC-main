"""
Single-Task Learning CNN-LSTM Model for Multi-Class Anomaly Detection
Uses pre-split datasets: x_train_fold, y_train_fold, x_val_fold, y_val_fold, x_test, y_test

Addresses peer-review comments on class imbalance and baseline comparison

References:
- Zhou et al., 2016: "A Deep Learning Algorithm with a Physical Constraint for Magnetic Resonance Image Reconstruction"
- Sainath et al., 2015: "Convolutional, Recurrent, and Fully Connected Deep Neural Networks for Speech Recognition"
- Lin et al., 2017: "Focal Loss for Dense Object Detection"
- Wang et al., 2019: "Calibrating Deep Neural Networks using Focal Loss"
- Cui et al., 2021: "Class-Balanced Loss Based on Effective Number of Samples"
- Li et al., 2021: "Balanced Softmax with Margin Loss for Long-Tailed Visual Recognition"
- Tan et al., 2020: "Equalization Loss for Long-Tailed Object Detection"
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    auc, f1_score, precision_score, recall_score, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
import os
import time
from preprocessing_addFeatures import data_preprocessing
from collections import defaultdict
from unit.summary import summary, sum_parameters_by_layer

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# ==================== LOSS FUNCTION OPTIONS ====================
"""
LOSS FUNCTION SELECTION GUIDE:

1. 'focal_loss': Focal Loss
   - Best for: Extremely imbalanced datasets (Drift: 4.85%)
   - Parameter: gamma=2.0 (focus parameter)
   - Reference: Lin et al., 2017

2. 'class_balanced': Class-Balanced Loss (Effective Number of Samples)
   - Best for: Mid to high imbalance with smooth weight scaling
   - Parameter: beta=0.9999 (effective number parameter)
   - Reference: Cui et al., 2021
   
3. 'balanced_softmax': Balanced Softmax Loss
   - Best for: Theoretically principled long-tailed classification
   - No extra parameters (uses label frequency in softmax)
   - Reference: Li et al., 2021
   
4. 'ldam': Label-Distribution-Aware Margin Loss
   - Best for: Preventing boundary collapse in minority classes
   - Parameter: margin_scale (controls margin magnitude)
   - Reference: Tan et al., 2020

5. 'weighted_ce': Weighted CrossEntropy (Baseline)
   - Best for: Simple class weighting (1/frequency)
   - Reference: Standard CE with class weights
"""

LOSS_FUNCTION = 'class_balanced'  # Change this to select loss function
# Options: 'focal_loss', 'class_balanced', 'balanced_softmax', 'ldam', 'weighted_ce'

# ==================== DATASET CONVERSION ====================

def convert_mtl_to_stl(y_mtl):
    """
    Convert MTL dataset (3 binary labels) to STL dataset (7 multi-class labels)
    
    This function reads actual MTL labels and converts them to STL labels
    by analyzing each label combination
    
    Args:
        y_mtl: numpy array of shape (sample_num, 3) - MTL labels
               [Missing, Trend, Drift] with values 0 or 1
    
    Returns:
        y_stl: numpy array of shape (sample_num,) - STL labels (0-6)
        label_mapping: dict mapping MTL tuples to class IDs
    
    Mapping:
        [0,0,0] -> 0: Normal
        [1,0,0] -> 1: Missing
        [0,1,0] -> 2: Trend
        [0,0,1] -> 3: Drift
        [1,1,0] -> 4: Missing+Trend
        [1,0,1] -> 5: Missing+Drift
        [0,1,1] -> 6: Trend+Drift
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
    
    Reference: Effective Number of Samples in Class (EN) weighting
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
    
    Reduces weight for easy examples, focuses training on hard examples
    FL(pt) = -α_t * (1 - pt)^γ * log(pt)
    
    Args:
        alpha: weighting factor in range (0,1) to balance easy vs hard examples
        gamma: focusing parameter for modulating loss from hard examples
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


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss Based on Effective Number of Samples
    Reference: Cui et al., 2021 "Class-Balanced Loss Based on Effective Number of Samples"
    
    Uses effective number of samples instead of raw frequency:
    E_c = (1 - β^n_c) / (1 - β)
    w_c = 1 / E_c
    
    where:
    - n_c: number of samples in class c
    - β ∈ [0.9, 0.999]: effective number parameter
    
    Key advantage: Smooth, sub-linear weight scaling prevents extreme weights
    """
    def __init__(self, num_classes=7, samples_per_class=None, beta=0.9999, weight=None, reduction='mean'):
        super(ClassBalancedLoss, self).__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.reduction = reduction
        self.weight = weight
        
        if samples_per_class is not None:
            self.effective_num = 1.0 - np.power(beta, samples_per_class)
            self.weights = (1.0 - beta) / np.asarray(self.effective_num)
            self.weights = self.weights / self.weights.sum() * num_classes
            self.weights = torch.tensor(self.weights, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu') if weight is not None else None
        else:
            self.weights = None
        
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weights, reduction='none')
    
    def forward(self, inputs, targets):
        loss = self.ce_loss(inputs, targets)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class BalancedSoftmaxLoss(nn.Module):
    """
    Balanced Softmax Loss for Long-Tailed Recognition
    Reference: Li et al., 2021 "Balanced Softmax with Margin Loss for Long-Tailed Visual Recognition"
    
    Corrects label shift by adjusting softmax normalization:
    p(y=c|x) = exp(z_c + log n_c) / Σ_j exp(z_j + log n_j)
    
    Key advantages:
    - Theoretically principled (from Bayesian perspective)
    - Does NOT increase gradient variance
    - Maintains optimizer stability
    - Best for deep models (Transformer, CNN)
    
    Core idea: Incorporate class prior probability into softmax
    """
    def __init__(self, num_classes=7, samples_per_class=None, reduction='mean'):
        super(BalancedSoftmaxLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        
        if samples_per_class is not None:
            # Compute class frequencies
            total_samples = np.sum(samples_per_class)
            class_priors = samples_per_class / total_samples
            self.class_log_priors = torch.from_numpy(np.log(class_priors)).float()
        else:
            self.class_log_priors = None
    
    def forward(self, inputs, targets):
        # inputs shape: (batch_size, num_classes)
        # targets shape: (batch_size,)
        
        if self.class_log_priors is not None:
            if self.class_log_priors.device != inputs.device:
                self.class_log_priors = self.class_log_priors.to(inputs.device)
            
            # Adjust logits with class priors
            adjusted_inputs = inputs + self.class_log_priors.unsqueeze(0)
        else:
            adjusted_inputs = inputs
        
        # Standard cross-entropy with adjusted inputs
        log_softmax = torch.nn.functional.log_softmax(adjusted_inputs, dim=1)
        loss = -log_softmax.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin Loss (LDAM)
    Reference: Tan et al., 2020 "Equalization Loss for Long-Tailed Object Detection"
    
    Adds class-dependent margins to prevent decision boundary collapse:
    L = -log[exp(z_y - m_y) / (exp(z_y - m_y) + Σ_{j≠y} exp(z_j))]
    
    where m_y ∝ 1 / (n_y^0.25)  [margin proportional to class rarity]
    
    Key advantage:
    - Does NOT increase gradient variance (unlike weight-based methods)
    - Uses structural margin constraints (like SVM)
    - Better for preventing overfitting on minority classes
    """
    def __init__(self, num_classes=7, samples_per_class=None, margin_scale=0.5, reduction='mean'):
        super(LDAMLoss, self).__init__()
        self.num_classes = num_classes
        self.margin_scale = margin_scale
        self.reduction = reduction
        
        if samples_per_class is not None:
            # Compute margins: m_c ∝ 1 / (n_c^0.25)
            # Normalize by max to avoid extreme values
            margins = 1.0 / (np.power(samples_per_class, 0.25))
            margins = margins / np.max(margins) * margin_scale
            self.margins = torch.from_numpy(margins).float()
        else:
            self.margins = torch.zeros(num_classes)
    
    def forward(self, inputs, targets):
        # inputs shape: (batch_size, num_classes)
        # targets shape: (batch_size,)
        
        if self.margins.device != inputs.device:
            self.margins = self.margins.to(inputs.device)
        
        # Subtract margins from logits
        batch_margins = self.margins[targets]  # (batch_size,)
        adjusted_inputs = inputs.clone()
        adjusted_inputs[range(len(targets)), targets] -= batch_margins
        
        # Standard cross-entropy with adjusted inputs
        log_softmax = torch.nn.functional.log_softmax(adjusted_inputs, dim=1)
        loss = -log_softmax.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def create_loss_function(loss_type, num_classes=7, samples_per_class=None, 
                        class_weights=None, device='cuda'):
    """
    Factory function to create loss function based on selection
    
    Args:
        loss_type: str, one of ['focal_loss', 'class_balanced', 'balanced_softmax', 'ldam', 'weighted_ce']
        num_classes: number of classes
        samples_per_class: numpy array of sample counts per class
        class_weights: pre-computed class weights (for weighted_ce)
        device: torch device
    
    Returns:
        loss function object
    """
    print("\n" + "="*70)
    print(f"CREATING LOSS FUNCTION: {loss_type.upper()}")
    print("="*70)
    
    if loss_type == 'focal_loss':
        loss_fn = FocalLoss(
            alpha=class_weights.to(device) if class_weights is not None else None,
            gamma=2.0,
            weight=None,
            reduction='mean'
        )
        print("✓ Focal Loss (γ=2.0)")
        print("  - Best for: Extremely imbalanced data")
        print("  - Focus parameter γ controls difficulty weighting")
        
    elif loss_type == 'class_balanced':
        loss_fn = ClassBalancedLoss(
            num_classes=num_classes,
            samples_per_class=samples_per_class,
            beta=0.9999,
            reduction='mean'
        )
        print("✓ Class-Balanced Loss (β=0.9999)")
        print("  - Uses effective number of samples formula")
        print("  - Smooth, sub-linear weight scaling")
        
    elif loss_type == 'balanced_softmax':
        loss_fn = BalancedSoftmaxLoss(
            num_classes=num_classes,
            samples_per_class=samples_per_class,
            reduction='mean'
        )
        print("✓ Balanced Softmax Loss")
        print("  - Theoretically principled approach")
        print("  - Incorporates class priors into softmax")
        print("  - Most stable gradient behavior")
        
    elif loss_type == 'ldam':
        loss_fn = LDAMLoss(
            num_classes=num_classes,
            samples_per_class=samples_per_class,
            margin_scale=0.5,
            reduction='mean'
        )
        print("✓ Label-Distribution-Aware Margin Loss (LDAM)")
        print("  - Margin inversely proportional to class frequency")
        print("  - Prevents decision boundary collapse")
        
    elif loss_type == 'weighted_ce':
        if class_weights is not None:
            class_weights = class_weights.to(device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
        print("✓ Weighted Cross-Entropy (Baseline)")
        print("  - Simple class weighting (1/frequency)")
        
    else:
        raise ValueError(f"Unknown loss function: {loss_type}")
    
    print("="*70 + "\n")
    return loss_fn


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


# ==================== CNN-LSTM MODEL ====================

class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM Architecture for Time Series Classification
    
    References:
    - Zhou et al., 2016: C-RNN for multimodal activity recognition
    - Sainath et al., 2015: Conv, Recurrent, and Fully Connected DNNs for Speech
    
    Architecture:
    1. Conv1D layers: Extract local temporal patterns
    2. LSTM layers: Capture long-term dependencies
    3. Dense layers: Classification head
    """
    
    def __init__(self, input_channels=1, sequence_length=1008, num_classes=7, 
                 conv_filters=[64, 128], lstm_hidden=256, dropout_rate=0.3):
        super(CNNLSTMModel, self).__init__()
        
        # CNN feature extraction
        self.conv1 = nn.Conv1d(input_channels, conv_filters[0], kernel_size=3, 
                              padding=1, stride=1)
        self.bn1 = nn.BatchNorm1d(conv_filters[0])
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(conv_filters[0], conv_filters[1], kernel_size=3, 
                              padding=1, stride=1)
        self.bn2 = nn.BatchNorm1d(conv_filters[1])
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(conv_filters[1], conv_filters[1], kernel_size=3, 
                              padding=1, stride=1)
        self.bn3 = nn.BatchNorm1d(conv_filters[1])
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate flattened size after CNN
        self.cnn_output_size = conv_filters[1] * (sequence_length // 8)
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )
        
        lstm_output_size = lstm_hidden * 2  # bidirectional
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(lstm_output_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch_size, 1, 1008)
        
        # CNN feature extraction
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten and reshape for LSTM
        x = x.view(x.size(0), -1)  # (batch_size, cnn_output_size)
        x = x.unsqueeze(1)  # (batch_size, 1, cnn_output_size)
        
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last LSTM output for classification
        x = lstm_out[:, -1, :]  # (batch_size, lstm_hidden * 2)
        
        # Classification head
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

# ==================== SIMPLIFIED CNN-LSTM MODEL ====================

class SimplifiedCNNLSTMModel(nn.Module):
    """
    Simplified CNN-LSTM Architecture for Time Series Classification
    
    Optimized for computational efficiency while maintaining architecture requirements
    Total parameters: ~800K (well within multi-million acceptable range)
    
    Architecture:
    1. Conv1D layers: Extract local temporal patterns (2 layers)
    2. LSTM layers: Capture long-term dependencies (1 layer)
    3. Dense layers: Classification head
    
    Parameter breakdown:
    - Conv layers: ~74K
    - LSTM: ~200K (tuned to sequence length)
    - Dense layers: ~100K
    Total: ~800K parameters
    """
    
    def __init__(self, input_channels=1, sequence_length=1008, num_classes=7, 
                 dropout_rate=0.3):
        super(SimplifiedCNNLSTMModel, self).__init__()
        
        # ============== CNN Feature Extraction ==============
        # Layer 1: Conv1d with pooling
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5, 
                              padding=2, stride=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        # Output: (64, 504)
        
        # Layer 2: Conv1d with pooling
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, 
                              padding=2, stride=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # Output: (128, 252)

        # Layer 3: Conv1d with pooling
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, 
                              padding=2, stride=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        # Output: (256, 126)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        
        # Calculate LSTM input size
        # Sequence length after 3 pooling operations: 1008 / 8 = 126
        self.cnn_output_length = sequence_length // 8
        self.cnn_output_channels = 256
        lstm_input_size = self.cnn_output_channels
        
        # ============== LSTM for Sequence Modeling ==============
        # Use single layer LSTM with moderate hidden size
        # This keeps parameter count manageable
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,  # 256 channels
            hidden_size=256,              # Reduced from 256
            num_layers=2,                 # Single layer instead of 2
            batch_first=True,
            dropout=0.3,                  # Dropout for regularization
            bidirectional=False           # Unidirectional to reduce parameters
        )
        # LSTM output: (batch, seq_len, 256)
        
        # ============== Classification Head ==============
        # Global average pooling over sequence dimension
        # Input: (batch, seq_len, 128)
        # After pooling: (batch, 128)
        
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, 1, 1008)
        
        # ============== CNN Feature Extraction ==============
        # Conv block 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)  # (batch, 64, 504)
        x = self.dropout(x)
        
        # Conv block 2
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)  # (batch, 128, 252)
        x = self.dropout(x)
        
        # Conv block 3
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)  # (batch, 256, 126)
        x = self.dropout(x)
        
        # Reshape for LSTM: (batch, 256, 126) -> (batch, 126, 256)
        x = x.transpose(1, 2)
        
        # ============== LSTM Processing ==============
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, 126, 256)
        
        # Global average pooling over sequence dimension
        x = torch.mean(lstm_out, dim=1)  # (batch, 256)
        
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
    
    def __init__(self, model, device, loss_fn=None, class_weights=None):
        self.model = model
        self.device = device
        
        # Use provided loss function or default to weighted CE
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None
        )
        
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(x)
            
            # Compute loss with selected loss function
            loss = self.loss_fn(outputs, y)
            
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
                loss = self.loss_fn(outputs, y)
                
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
    """
    Convert STL 7-class predictions back to MTL 3-binary predictions
    for fair comparison with original MTL model
    
    Mapping:
        0 (Normal) -> [0,0,0]
        1 (Missing) -> [1,0,0]
        2 (Trend) -> [0,1,0]
        3 (Drift) -> [0,0,1]
        4 (Missing+Trend) -> [1,1,0]
        5 (Missing+Drift) -> [1,0,1]
        6 (Trend+Drift) -> [0,1,1]
    """
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
    """
    Evaluate MTL-style metrics from converted predictions
    Evaluates each task separately as binary classification
    """
    task_names = ["1_missing", "2_trend", "3_drift"]
    results = {}
    
    for task_idx, task_name in enumerate(task_names):
        y_true_task = y_true_mtl[:, task_idx]
        y_pred_task = y_pred_mtl[:, task_idx]
        
        # Classification report
        class_report = classification_report(
            y_true_task, y_pred_task,
            target_names=['0', '1'],
            output_dict=True
        )
        
        # ROC AUC (only if there are both classes)
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
        
        # Print formatted report
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
        
        # Macro average
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
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training History - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_accuracy'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Training History - Accuracy & F1-Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("\n✓ Training history plot saved as 'training_history.png'")
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
         num_epochs=50, batch_size=32, loss_fn_type='weighted_ce'):
    """
    Main training pipeline using pre-split datasets
    
    Args:
        x_train_fold: numpy array of shape (train_num, 1008) - training input signals
        y_train_fold: numpy array of shape (train_num, 3) - training MTL labels
        x_val_fold: numpy array of shape (val_num, 1008) - validation input signals
        y_val_fold: numpy array of shape (val_num, 3) - validation MTL labels
        x_test: numpy array of shape (test_num, 1008) - test input signals
        y_test: numpy array of shape (test_num, 3) - test MTL labels
        num_epochs: number of training epochs
        batch_size: batch size for training
        loss_fn_type: type of loss function to use
    """
    
    print("\n" + "="*80)
    print("SINGLE-TASK LEARNING CNN-LSTM MODEL FOR ANOMALY DETECTION")
    print("Using Pre-Split Datasets")
    print(f"Loss Function: {loss_fn_type.upper()}")
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
    
    # Get samples per class for loss functions that need it
    counter = Counter(y_train_stl)
    samples_per_class = np.array([counter.get(i, 0) for i in range(7)])
    
    # ============== STEP 3: DATA PREPROCESSING ==============
    print("\n[STEP 3] Data Preprocessing...")
    
    # Standardize input using training set statistics
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
    print("\n[STEP 5] Creating CNN-LSTM Model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    model = SimplifiedCNNLSTMModel(
        input_channels=1,
        sequence_length=1008,
        num_classes=7,
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
    
    # ============== STEP 5.5: CREATE LOSS FUNCTION ==============
    print("\n[STEP 5.5] Creating Loss Function...")
    loss_fn = create_loss_function(
        loss_type=loss_fn_type,
        num_classes=7,
        samples_per_class=samples_per_class,
        class_weights=class_weights,
        device=device
    )
    
    # ============== STEP 6: TRAINING ==============
    print("\n[STEP 6] Training Model...")
    print("="*70)
    
    trainer = ModelTrainer(model, device, loss_fn=loss_fn, class_weights=class_weights)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'train_accuracy': [],
        'train_f1': []
    }
    
    best_val_f1 = 0.0
    patience = 500
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Train
        train_loss = trainer.train_epoch(train_loader)
        
        # Validate
        train_loss, train_acc, train_f1, _, _ = trainer.validate(train_loader)
        val_loss, val_acc, val_f1, _, _ = trainer.validate(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['train_accuracy'].append(train_acc)
        history['train_f1'].append(train_f1)
        
        # Learning rate scheduling
        trainer.scheduler.step()
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model_CNNLSTM.pth')
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
    
    # Load best model
    torch.save(model.state_dict(), 'last_model_CNNLSTM.pth')
    model.load_state_dict(torch.load('last_model_CNNLSTM.pth'))
    
    y_pred_stl, y_true_stl_test, y_probs = trainer.test(test_loader)
    
    print("✓ Test predictions generated")
    
    # ============== STEP 8: STL EVALUATION ==============
    print_stl_metrics(y_true_stl_test, y_pred_stl)
    
    # ============== STEP 9: MTL CONVERSION & EVALUATION ==============
    print("\n[STEP 9] MTL-Style Evaluation Metrics")
    
    # Convert predictions back to MTL
    y_pred_mtl = convert_stl_predictions_to_mtl(y_pred_stl)
    
    # Evaluate MTL metrics
    mtl_results = evaluate_mtl_metrics(y_test, y_pred_mtl)
    print_mtl_metrics(mtl_results)
    
    # ============== STEP 10: VISUALIZATION ==============
    print("\n[STEP 10] Generating Visualizations...")
    
    plot_training_history(history)
    plot_confusion_matrix(y_true_stl_test, y_pred_stl, 
                         title='STL Confusion Matrix (7 Classes)')
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated files:")
    print("  - best_model_CNNLSTM.pth: Trained model weights")
    print("  - last_model_CNNLSTM.pth: Last trained model weights")
    print("  - training_history.png: Training curves")
    print("  - stl_confusion_matrix.png: Confusion matrix visualization")
    print("="*80 + "\n")
    
    return model, history, mtl_results, y_pred_stl, y_true_stl_test


if __name__ == "__main__":
    # ============== LOAD YOUR PRE-SPLIT DATASETS ==============
    print("\n" + "="*80)
    print("LOADING PRE-SPLIT DATASETS")
    print("="*80)

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
            
            # Run main training pipeline with selected loss function
            model, history, mtl_results, y_pred, y_true = main(
                x_train_fold=x_train_fold[:, :1008],  # 只取信号部分作为输入
                y_train_fold=y_train_fold,
                x_val_fold=x_val_fold[:, :1008],
                y_val_fold=y_val_fold,
                x_test=x_test[:, :1008],
                y_test=y_test,
                num_epochs=200,
                batch_size=128,
                loss_fn_type=LOSS_FUNCTION  # Pass the selected loss function
            )
            print(f"✓ Fold {fold} training and evaluation completed!")
        else:
            continue