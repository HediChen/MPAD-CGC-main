import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List, Dict, Tuple, Union, Optional

# Dilated Convolution Expert
class DilatedConvExpert(nn.Module):
    def __init__(self, input_channels=1, output_dim=256):
        super(DilatedConvExpert, self).__init__()

        # Define dilated convolution layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=7, stride=1, padding=3, dilation=1),  # (batch, 16, 1008)
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2, dilation=2),  # (batch, 32, 1008)
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, dilation=4),  # (batch, 64, 1008)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # (batch, 64, 1)
            nn.Flatten(),  # (batch, 64)
        )
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x

# CNN Expert
class CNNExpert(nn.Module):
    def __init__(self, input_channels=1, output_dim=256):
        super(CNNExpert, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=7, stride=2, padding=3),  # (batch, 16, 504)
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),  # (batch, 32, 252)
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # (batch, 64, 126)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # (batch, 64, 1)
            nn.Flatten(),  # (batch, 64)
        )
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x

# Multi-head Attention Layer
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        assert self.head_dim * num_heads == input_dim, "Input dimension must be divisible by the number of heads"

        # Linear transformations for query, key, and value
        self.q_linear = nn.Linear(input_dim, input_dim)
        self.k_linear = nn.Linear(input_dim, input_dim)
        self.v_linear = nn.Linear(input_dim, input_dim)

        # Output linear transformation
        self.out_linear = nn.Linear(input_dim, input_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        # Apply linear transformations to query, key, and value
        query = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.head_dim ** 0.5  # (batch_size, num_heads, seq_len, seq_len)
        attention_weights = torch.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attention_weights = self.dropout(attention_weights)

        # Weighted sum of values
        output = torch.matmul(attention_weights, value)  # (batch_size, num_heads, seq_len, head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        # Final output transformation
        output = self.out_linear(output)

        return output

# Modified MMoe model with Attention
class MMoeWithAttention(nn.Module):
    def __init__(self,
                 inputs_dim: int,
                 labels_dict: Dict[str, int],
                 num_experts: int,
                 expert_hidden_units: Union[List[int], Tuple[int]],
                 tower_hidden_units: Union[List[int], Tuple[int]] = (256, 128),
                 l2_reg_dnn: float = 0.,
                 dnn_dropout: float = 0.,
                 dnn_activation: Optional[str] = 'relu',
                 dnn_use_bn: bool = False,
                 device: str = 'cpu'):
        super(MMoeWithAttention, self).__init__()

        self.labels_dict = labels_dict
        output_dim = expert_hidden_units[-1]

        # CNN-based Experts
        self.experts = nn.ModuleList([CNNExpert(input_channels=1, output_dim=output_dim)
                                      for _ in range(num_experts)])

        # Gating Networks still take flattened input
        self.gate_dnn = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(1008, num_experts)
            ) for _ in labels_dict
        ])

        # Task Towers and final output layers with Attention added
        self.task_tower = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_dim, tower_hidden_units[0]),
                nn.ReLU(),
                MultiHeadAttention(tower_hidden_units[0], num_heads=4),  # Attention layer added
                nn.Linear(tower_hidden_units[0], tower_hidden_units[1]),
                nn.ReLU()
            ) for _ in labels_dict
        ])

        self.task_dense = nn.ModuleList([
            nn.Linear(tower_hidden_units[-1], labels_dict[name]) for name in labels_dict
        ])

        self.l2_reg_dnn = l2_reg_dnn
        self.device = device
        self.to(device)

    @property
    def l2_reg_loss(self):
        reg_loss = torch.zeros((1,), device=self.device)
        if self.l2_reg_dnn and self.l2_reg_dnn > 0.:
            for name, parameter in self.named_parameters():
                if 'weight' in name:
                    reg_loss += torch.sum(self.l2_reg_dnn * torch.square(parameter))
        return reg_loss

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = OrderedDict()

        # Each CNN expert processes the 1D input
        experts_output = [expert(x) for expert in self.experts]

        for idx, name in enumerate(self.labels_dict):
            gate = self.gate_dnn[idx](x)
            tower_input = self._merge_experts_with_gate(experts_output, gate)
            tower_output = self.task_tower[idx](tower_input)
            task_output = self.task_dense[idx](tower_output)
            outputs[name] = torch.sigmoid(task_output)

        return outputs

    def _merge_experts_with_gate(self, experts: List[torch.Tensor],
                                 gate: torch.Tensor):
        experts = torch.stack(experts, dim=1)  # (batch_size, num_experts, feature_dim)
        gate_weight = torch.softmax(gate, dim=-1).unsqueeze(2)  # (batch_size, num_experts, 1)
        return torch.sum(experts * gate_weight, dim=1)  # Weighted sum

# Example usage
if __name__ == '__main__':
    model = MMoeWithAttention(inputs_dim=1008,
                              labels_dict={"task1": 2, "task2": 3, "task3": 4, "task4": 2, "task5": 5},
                              num_experts=3,
                              expert_hidden_units=[256])  # Output dim from CNN expert

    dummy_input = torch.randn(4, 1, 1008)  # (batch_size, channels, length)
    outputs = model(dummy_input)

    for name in outputs:
        print(f"{name}: {outputs[name].shape}")
