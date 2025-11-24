import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List, Dict, Union, Tuple, Optional
import sys
sys.path.append('./pytorchModels')
from utils import DNN

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNExpert(nn.Module):
    def __init__(self, input_channels=1, num_channels=[16, 32, 64], kernel_size=3, dropout=0.2, output_dim=256):
        super(TCNExpert, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = input_channels if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                     stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 2) * dilation_size,
                                     dropout=dropout)]
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(num_channels[-1], output_dim)

    def forward(self, x):
        out = self.tcn(x)
        out = self.pool(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out


class FeatureAttention(nn.Module):
    def __init__(self, input_dim):
        super(FeatureAttention, self).__init__()
        self.attn_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim),  # (B, 9) → (B, 9)
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),  # (B, 9) → (B, 9)
        )
        self.softmax = nn.Softmax(dim=1)  # Attention across the 9 features

    def forward(self, x):
        # x: (B, 9)
        scores = self.attn_layer(x)  # (B, 9)
        weights = self.softmax(scores)  # (B, 9)
        attended = x * weights  # Element-wise weighting
        return attended

class GateNetwork(nn.Module):
    def __init__(self, inputs_dim, output_dim, num_experts, shared_experts_cnn):
        super(GateNetwork, self).__init__()
        # Replace DNN experts with CNN experts
        self.task_experts_dnn = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(),
                FeatureAttention(input_dim=9),
                nn.Linear(9, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                # nn.Dropout(0.2),
                nn.Linear(16, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                # nn.Dropout(0.2),
                nn.Linear(32, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                # nn.Dropout(0.2),
                nn.Linear(64, output_dim)
                ) for _ in range(num_experts)
                ])
        self.shared_experts_cnn = shared_experts_cnn

        # self.gates = nn.Linear(inputs_dim, num_experts + len(shared_experts_cnn), bias=False)

        self.gates = nn.Sequential(
                nn.Flatten(),
                nn.Linear(1008, 128),
                nn.ReLU(),
                nn.Linear(128, num_experts + len(shared_experts_cnn))
                )

    def forward(self, inputs):
        # inputs: (batch_size, 1, 1008)
        experts_output = [expert(inputs[:,:,:1008]) for expert in self.shared_experts_cnn]
        experts_output += [expert(inputs[:,:,1008:]) for expert in self.task_experts_dnn]

        experts_output = torch.stack(experts_output, dim=1)  # (B, num_experts_total, D)

        # gate_input = experts_output[:, 0, :]  # Use one expert's output for gate input (or use avg)
        gate_weight = torch.softmax(self.gates(inputs[:,:,:1008]), dim=-1).squeeze(1).unsqueeze(2)  # (B, num_experts_total)

        return torch.sum(experts_output * gate_weight, dim=1)


class PLE(nn.Module):
    """One Level PLE.

    :param inputs_dim: Dimension of the inputs.
    :param labels_dict: dict. The number of Labels
    :param num_shared_experts: int. The number of Shared Experts
    :param num_task_experts: int. The number of every task Specific Experts
    :param expert_hidden_units: list of positive integer, the layer number and units in each expert layer.
    :param tower_hidden_units: list of positive integer, the layer number and units in each tower layer.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether to use BatchNormalization before activation or not in DNN
    :return: A PyTorch model instance.
    """

    def __init__(self,
                 inputs_dim: int,
                 labels_dict: Dict[str, int],
                 num_shared_experts: int,
                 num_task_experts: int,
                 expert_hidden_units: Union[List[int], Tuple[int]],
                 tower_hidden_units=(256, 128),
                 l2_reg_dnn: float = 0.,
                 dnn_dropout: float = 0.,
                 dnn_activation: Optional[str] = 'relu',
                 dnn_use_bn: bool = False,
                 device: str = 'cpu'):
        super(PLE, self).__init__()

        self.labels_dict = labels_dict

        # Replace DNN experts with CNN experts
        output_dim = expert_hidden_units[-1]
        self.shared_experts_cnn = nn.ModuleList([TCNExpert(output_dim=output_dim) for _ in range(num_shared_experts)])

        self.gate_network = nn.ModuleList([
            GateNetwork(inputs_dim, output_dim, num_task_experts, self.shared_experts_cnn)
            for _ in labels_dict
        ])

        # Task Towers and final output layers
        self.task_tower = nn.ModuleList([
            nn.Sequential(
                # FeatureAttention(input_dim=output_dim),
                nn.Linear(output_dim, tower_hidden_units[0]),
                nn.BatchNorm1d(tower_hidden_units[0]),
                nn.ReLU(),
                nn.Dropout(dnn_dropout),
                nn.Linear(tower_hidden_units[0], tower_hidden_units[1]),
                nn.BatchNorm1d(tower_hidden_units[1]),
                nn.ReLU(),
                nn.Dropout(dnn_dropout),
                nn.Linear(tower_hidden_units[1], tower_hidden_units[1]),
                nn.BatchNorm1d(tower_hidden_units[1]),
                nn.ReLU(),
                nn.Dropout(dnn_dropout),
            ) for _ in labels_dict
        ])

        self.task_dense = nn.ModuleList(
            [DNN(tower_hidden_units[-1], [labels_dict[name]], activation=None, bias=False) for name in labels_dict]
        )

        self.l2_reg_dnn = l2_reg_dnn
        self.device = device
        self.to(device)

    @property
    def l2_reg_loss(self):
        """L2 Regularization Loss"""
        reg_loss = torch.zeros((1,), device=self.device)
        if self.l2_reg_dnn and self.l2_reg_dnn > 0.:
            for name, parameter in self.named_parameters():
                if 'weight' in name:
                    reg_loss += torch.sum(self.l2_reg_dnn * torch.square(parameter))
        return reg_loss

    def forward(self, dnn_inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = OrderedDict()
        for index, name in enumerate(self.labels_dict):
            tower_inputs = self.gate_network[index](dnn_inputs)
            tower_output = self.task_tower[index](tower_inputs)
            task_output = self.task_dense[index](tower_output)

            # outputs[name] = torch.softmax(task_output, dim=-1)
            outputs[name] = torch.sigmoid(task_output)

        return outputs


if __name__ == '__main__':
    import numpy as np

    # Define the model with the modified architecture
    model = PLE(inputs_dim=1008,
                labels_dict={"click": 2, "like": 2},
                num_shared_experts=2,
                num_task_experts=2,
                expert_hidden_units=[256])

    # Test the model
    inputs = torch.FloatTensor(np.random.random([4, 1, 1008]))  # Example input shape (4, 1, 1008)
    outputs = model(inputs)

    print(outputs)
    for name in outputs:
        print(name, outputs[name].shape)
