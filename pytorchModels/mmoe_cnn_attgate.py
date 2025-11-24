import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List, Dict, Tuple, Union, Optional


class DilatedConvExpert(nn.Module):
    def __init__(self, input_channels=1, output_dim=256):
        super(DilatedConvExpert, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=7, stride=1, padding=3, dilation=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2, dilation=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, dilation=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x


class CNNExpert(nn.Module):
    def __init__(self, input_channels=1, output_dim=256):
        super(CNNExpert, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x


class MultiHeadAttentionGate(nn.Module):
    def __init__(self, num_experts: int, expert_dim: int, num_heads: int = 4):
        super(MultiHeadAttentionGate, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=expert_dim, num_heads=num_heads, batch_first=True)
        self.query_proj = nn.Linear(1008, expert_dim)

    def forward(self, experts: torch.Tensor, x: torch.Tensor):
        query = self.query_proj(x.view(x.size(0), -1)).unsqueeze(1)  # (batch_size, 1, expert_dim)
        attn_output, _ = self.attn(query, experts, experts)  # (batch_size, 1, expert_dim)
        return attn_output.squeeze(1)  # (batch_size, expert_dim)


class MMoe(nn.Module):
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
        super(MMoe, self).__init__()

        self.labels_dict = labels_dict
        output_dim = expert_hidden_units[-1]

        # Experts
        self.experts = nn.ModuleList([CNNExpert(input_channels=1, output_dim=output_dim)
                                      for _ in range(num_experts)])
        
        # Multi-Head Attention per task
        self.attn_modules = nn.ModuleList([
            MultiHeadAttentionGate(num_experts=num_experts, expert_dim=output_dim)
            for _ in labels_dict
        ])

        # Task Towers
        self.task_tower = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_dim, tower_hidden_units[0]),
                nn.ReLU(),
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

        # Experts
        experts_output = [expert(x) for expert in self.experts]  # List of (batch, expert_dim)
        experts_output = torch.stack(experts_output, dim=1)  # (batch, num_experts, expert_dim)

        # Task-specific attention & tower
        for idx, name in enumerate(self.labels_dict):
            attn_output = self.attn_modules[idx](experts_output, x)  # (batch, expert_dim)
            tower_output = self.task_tower[idx](attn_output)
            task_output = self.task_dense[idx](tower_output)
            outputs[name] = torch.sigmoid(task_output)

        return outputs


# === Test ===
if __name__ == '__main__':
    model = MMoe(inputs_dim=1008,
                 labels_dict={"task1": 2, "task2": 3, "task3": 4, "task4": 2, "task5": 5},
                 num_experts=3,
                 expert_hidden_units=[256])

    dummy_input = torch.randn(4, 1, 1008)  # (batch_size, channels, length)
    outputs = model(dummy_input)

    for name in outputs:
        print(f"{name}: {outputs[name].shape}")
