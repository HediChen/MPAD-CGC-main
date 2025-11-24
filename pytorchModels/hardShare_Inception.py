import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List, Dict, Union, Tuple, Optional
import sys
sys.path.append('./pytorchModels')
from utils import DNN


class LocalTemporalAttention(nn.Module):
    """
    Local Temporal Attention (LTA) module.
    Applies lightweight depthwise convolution-based attention across time.
    Input: (B, C, T)
    Output: (B, C, T)
    """
    def __init__(self, channels, kernel_size=9):
        super(LocalTemporalAttention, self).__init__()
        padding = kernel_size // 2
        self.attention = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels),  # Depthwise
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1),  # Pointwise
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.attention(x)  # (B, C, T)
        return x * attn           # Element-wise scaling

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()

        # Dilated convolutions with increasing dilation rates for broader receptive fields
        self.branch1 = nn.Conv1d(in_channels, out_channels, kernel_size=9, padding=16, dilation=4)  # RF = 33 receptive fields
        self.branch2 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=4, dilation=2)   # RF = 9
        self.branch3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)   # RF = 3
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

        self.bn = nn.BatchNorm1d(out_channels * 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.branch1(x)  # (B, out_channels, T)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        out = torch.cat([x1, x2, x3, x4], dim=1)  # (B, out_channels * 4, T)
        out = self.bn(out)
        return self.relu(out)


class InceptionExpert(nn.Module):
    def __init__(self, input_channels=1, output_dim=256):
        super(InceptionExpert, self).__init__()
        self.inception_layers = nn.Sequential(
            InceptionBlock(input_channels, 8),   # (batch, 32, 1008)
            # InceptionBlock(32, 8),               # (batch, 32, 1008)

            LocalTemporalAttention(32),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),  # (batch, 64, 504)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2),  # (batch, 32, 252)
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # # First depthwise separable conv (replaces 7x7 Conv)
            # nn.Conv1d(32, 32, kernel_size=7, stride=2, padding=3, groups=32),  # depthwise
            # nn.Conv1d(32, 64, kernel_size=1),  # pointwise
            # nn.BatchNorm1d(64),
            # nn.ReLU(),

            # # Second depthwise separable conv (replaces 5x5 Conv)
            # nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2, groups=64),  # depthwise
            # nn.Conv1d(64, 64, kernel_size=1),  # pointwise
            # nn.BatchNorm1d(64),
            # nn.ReLU(),


            # nn.Conv1d(32, 32, kernel_size=7, stride=2, padding=3),  # (batch, 64, 504)
            # nn.BatchNorm1d(32),
            # nn.ReLU(),
            # nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2),  # (batch, 32, 252)
            # nn.BatchNorm1d(32),
            # nn.ReLU(),
            # nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # (batch, 32, 252)
            # nn.BatchNorm1d(64),
            # nn.ReLU(),

            # nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2),  # (batch, 64, 504)
            # nn.BatchNorm1d(32),
            # nn.ReLU(),
            # nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # (batch, 32, 252)
            # nn.BatchNorm1d(64),
            # nn.ReLU(),
            # GlobalTemporalAttention(64),
            # ChannelAttention1D(64), #CAM

            nn.AdaptiveAvgPool1d(1),             # (batch, 64, 1)
            nn.Flatten()                         # (batch, 64)
        )
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.inception_layers(x)
        x = self.fc(x)
        return x


class HardShare(nn.Module):
    """One Level PLE.

    :param inputs_dim: Dimension of the inputs.
    :param labels_dict: dict. The number of Labels
    :param num_shared_experts: int. The number of Shared Experts
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
                 expert_hidden_units: Union[List[int], Tuple[int]],
                 tower_hidden_units=(256, 128),
                 l2_reg_dnn: float = 0.,
                 dnn_dropout: float = 0.,
                 dnn_activation: Optional[str] = 'relu',
                 dnn_use_bn: bool = False,
                 device: str = 'cpu'):
        super(HardShare, self).__init__()

        self.labels_dict = labels_dict

        output_dim = expert_hidden_units[-1]
        self.shared_experts_cnn = nn.ModuleList([InceptionExpert(output_dim=output_dim) for _ in range(num_shared_experts)])

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
                nn.Linear(tower_hidden_units[1], tower_hidden_units[2]),
                nn.BatchNorm1d(tower_hidden_units[2]),
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

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = OrderedDict()
        experts_output = [expert(inputs[:,:,:1008]) for expert in self.shared_experts_cnn]
        tower_inputs = torch.stack(experts_output, dim=1) # (B, num_experts, D)
        tower_inputs_avg = torch.mean(tower_inputs, dim=1)  # shape (B, D)
        for index, name in enumerate(self.labels_dict):

            tower_output = self.task_tower[index](tower_inputs_avg)
            task_output = self.task_dense[index](tower_output)

            outputs[name] = torch.sigmoid(task_output)

        return outputs


if __name__ == '__main__':
    import numpy as np

    # Define the model with the modified architecture
    model = HardShare(inputs_dim=1008,
                labels_dict={"click": 2, "like": 2},
                num_shared_experts=1,
                expert_hidden_units=[128],
                tower_hidden_units=[128, 64, 32])

    # Test the model
    inputs = torch.FloatTensor(np.random.random([4, 1, 1008]))  # Example input shape (4, 1, 1008)
    outputs = model(inputs)

    print(outputs)
    for name in outputs:
        print(name, outputs[name].shape)
