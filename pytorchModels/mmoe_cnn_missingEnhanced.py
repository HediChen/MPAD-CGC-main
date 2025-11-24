import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List, Dict, Tuple, Union, Optional

class DilatedConvExpert(nn.Module):
    def __init__(self, input_channels=1, output_dim=256):
        super(DilatedConvExpert, self).__init__()

        # Define dilated convolution layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=7, stride=1, padding=3, dilation=1),  # (batch, 16, 1008)
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2, dilation=1),  # (batch, 32, 1008)
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, dilation=1),  # (batch, 64, 1008)
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

class CNNExpert(nn.Module):
    def __init__(self, input_channels=1, output_dim=256, kernel_list=[7, 5, 3]):
        super(CNNExpert, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=kernel_list[0], stride=2, padding=kernel_list[0]//2),  # (batch, 16, 504)
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=kernel_list[1], stride=2, padding=kernel_list[1]//2),  # (batch, 32, 252)
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=kernel_list[2], stride=2, padding=kernel_list[2]//2),  # (batch, 64, 126)
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

class ResCNNExpert(nn.Module):
    def __init__(self, input_channels=1, output_dim=256):
        super(ResCNNExpert, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.res1 = nn.Conv1d(input_channels, 16, kernel_size=1, stride=2)  # For residual

        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.res2 = nn.Conv1d(16, 32, kernel_size=1, stride=2)

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.res3 = nn.Conv1d(32, 64, kernel_size=1, stride=2)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        out1 = self.layer1(x)
        res1 = self.res1(x)
        out1 = out1 + res1  # Residual connection 1

        out2 = self.layer2(out1)
        res2 = self.res2(out1)
        out2 = out2 + res2  # Residual connection 2

        out3 = self.layer3(out2)
        res3 = self.res3(out2)
        out3 = out3 + res3  # Residual connection 3

        pooled = self.global_pool(out3)
        flat = self.flatten(pooled)
        output = self.fc(flat)
        return output

class DCNNExpert(nn.Module):
    def __init__(self, input_dim=1008, intermediate_dim=768, input_channels=1, output_dim=256):
        super(DCNNExpert, self).__init__()
        self.fc_input = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.Dropout(p=0.3)
            )  # Fully connected layer from 1008 to 1024

        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=7, stride=2, padding=3),  # (batch, 16, 512)
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),  # (batch, 32, 256)
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # (batch, 64, 128)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # (batch, 64, 1)
            nn.Flatten(),  # (batch, 64)
        )
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 1008)
        x = self.fc_input(x)       # (batch_size, 1024)
        x = x.unsqueeze(1)         # Reshape to (batch_size, 1, 1024) for Conv1d
        x = self.conv_layers(x)
        x = self.fc(x)
        return x

class LSTMCNNExpert(nn.Module):
    def __init__(self, input_channels=1, input_len=1008, lstm_hidden_size=64, lstm_layers=1, output_dim=256):
        super(LSTMCNNExpert, self).__init__()
        self.lstm = nn.LSTM(input_size=input_channels,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=False)  # Unidirectional

        # Project LSTM output back to (B, C, T) format for Conv1D
        self.proj = nn.Linear(lstm_hidden_size, input_channels)

        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=7, stride=2, padding=3),  # (B, 16, L//2)
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),  # (B, 32, L//4)
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # (B, 64, L//8)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # (B, 64, 1)
            nn.Flatten()
        )
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):  # x: (B, 1, 1008)
        x = x.squeeze(1)         # (B, 1008)
        x = x.unsqueeze(-1)      # (B, 1008, 1)
        lstm_out, _ = self.lstm(x)  # (B, 1008, H)
        x = self.proj(lstm_out)     # (B, 1008, 1)
        x = x.permute(0, 2, 1)      # (B, 1, 1008) → for Conv1d

        x = self.conv_layers(x)     # (B, 64)
        x = self.fc(x)              # (B, output_dim)
        return x



class MMoe(nn.Module):
    def __init__(self,
                 inputs_dim: int,  # unused, but kept for compatibility
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

        # CNN-based Experts
        self.experts = nn.ModuleList([CNNExpert(input_channels=1, output_dim=output_dim)
                                      for _ in range(num_experts)])

        self.gate_dnn = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(1008, 128),
                nn.ReLU(),
                nn.Linear(128, num_experts)
                ) for _ in labels_dict
                ])


        # Task Towers and final output layers
        self.task_tower = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_dim, tower_hidden_units[0]),
                # nn.BatchNorm1d(tower_hidden_units[0]),
                nn.ReLU(),
                nn.Dropout(dnn_dropout),
                nn.Linear(tower_hidden_units[0], tower_hidden_units[1]),
                # nn.BatchNorm1d(tower_hidden_units[1]),
                nn.ReLU(),
                nn.Dropout(dnn_dropout),
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
            tower_input = _merge_experts_with_gate(experts_output, gate)

            # experts = torch.stack(experts_output, dim=1)
            # tower_input = torch.sum(experts, dim=1)

            tower_output = self.task_tower[idx](tower_input)
            task_output = self.task_dense[idx](tower_output)
            outputs[name] = torch.sigmoid(task_output)

            # outputs[name] = torch.softmax(task_output,dim=-1)

        return outputs


def _merge_experts_with_gate(experts: List[torch.Tensor],
                             gate: torch.Tensor):
    experts = torch.stack(experts, dim=1)  # (batch_size, num_experts, feature_dim)
    gate_weight = torch.softmax(gate, dim=-1).unsqueeze(2)  # (batch_size, num_experts, 1)
    return torch.sum(experts * gate_weight, dim=1)  # Weighted sum


# Example usage
if __name__ == '__main__':
    import numpy as np

    model = MMoe(inputs_dim=1008,
                 labels_dict={"task1": 2, "task2": 3, "task3": 4, "task4": 2, "task5": 5},
                 num_experts=3,
                 expert_hidden_units=[256])  # Output dim from CNN expert

    dummy_input = torch.randn(4, 1, 1008)  # (batch_size, channels, length)
    outputs = model(dummy_input)

    for name in outputs:
        print(f"{name}: {outputs[name].shape}")
