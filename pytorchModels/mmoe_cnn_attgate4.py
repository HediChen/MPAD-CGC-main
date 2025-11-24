import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List, Dict, Tuple, Union, Optional

# ----- Expert Definition -----
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
            nn.Flatten()
        )
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc(x)

# ----- Cross-Attention Gate with x as Query -----
class InputCrossAttentionGate(nn.Module):
    def __init__(self, input_dim: int, expert_dim: int):
        super(InputCrossAttentionGate, self).__init__()
        self.q_proj = nn.Linear(input_dim, expert_dim)
        self.k_proj = nn.Linear(expert_dim, expert_dim)
        self.v_proj = nn.Linear(expert_dim, expert_dim)
        self.scale = expert_dim ** 0.5

    def forward(self, experts: torch.Tensor, x_input: torch.Tensor):
        B, N, D = experts.size()
        x_flat = x_input.view(B, -1)  # (B, L)
        Q = self.q_proj(x_flat).unsqueeze(1)  # (B, 1, D)
        K = self.k_proj(experts)              # (B, N, D)
        V = self.v_proj(experts)              # (B, N, D)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, 1, N)
        attn_weights = torch.softmax(attn_scores, dim=-1)                # (B, 1, N)
        context = torch.matmul(attn_weights, V)                          # (B, 1, D)
        return context.squeeze(1)  # (B, D)

# ----- MMOE with Input-based Cross-Attention -----
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

        self.experts = nn.ModuleList([
            CNNExpert(input_channels=1, output_dim=output_dim)
            for _ in range(num_experts)
        ])

        self.attn_module = InputCrossAttentionGate(input_dim=inputs_dim, expert_dim=output_dim)

        self.task_tower = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_dim, tower_hidden_units[0]),
                nn.ReLU(),
                nn.Linear(tower_hidden_units[0], tower_hidden_units[1]),
                nn.ReLU()
            ) for _ in labels_dict
        ])

        self.task_dense = nn.ModuleList([
            nn.Linear(tower_hidden_units[-1], labels_dict[name])
            for name in labels_dict
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

        expert_outputs = [expert(x) for expert in self.experts]
        experts = torch.stack(expert_outputs, dim=1)  # (B, N, D)

        for idx, name in enumerate(self.labels_dict):
            attn_output = self.attn_module(experts, x_input=x)  # (B, D)
            tower_output = self.task_tower[idx](attn_output)
            task_output = self.task_dense[idx](tower_output)
            outputs[name] = torch.sigmoid(task_output)

        return outputs

# ----- Example -----
if __name__ == '__main__':
    model = MMoe(
        inputs_dim=1008,
        labels_dict={"task1": 2, "task2": 3, "task3": 4, "task4": 2, "task5": 5},
        num_experts=3,
        expert_hidden_units=[256]
    )

    dummy_input = torch.randn(4, 1, 1008)
    outputs = model(dummy_input)
    for name, out in outputs.items():
        print(f"{name}: {out.shape}")
