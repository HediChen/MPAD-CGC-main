import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List, Dict, Union, Tuple, Optional
import sys
sys.path.append('./pytorchModels')
from utils import DNN


class CNNExpert(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, kernel_size=3, final_fc_out=128):
        super(CNNExpert, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.MaxPool1d(2),

            nn.Conv1d(out_channels, out_channels, kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(out_channels, final_fc_out)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class GateNetwork(nn.Module):
    def __init__(self, num_experts, shared_experts_cnn):
        super(GateNetwork, self).__init__()
        # Replace DNN experts with CNN experts
        self.task_experts_cnn = nn.ModuleList([CNNExpert() for _ in range(num_experts)])
        self.shared_experts_cnn = shared_experts_cnn

        self.gates = nn.Linear(128, num_experts + len(shared_experts_cnn), bias=False)

    def forward(self, inputs):
        # inputs: (batch_size, 1, 1008)
        experts_output = [expert(inputs) for expert in self.shared_experts_cnn]
        experts_output += [expert(inputs) for expert in self.task_experts_cnn]

        experts_output = torch.stack(experts_output, dim=1)  # (B, num_experts_total, D)

        gate_input = experts_output[:, 0, :]  # Use one expert's output for gate input (or use avg)
        gate_weight = torch.softmax(self.gates(gate_input), dim=-1).unsqueeze(2)

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
        self.shared_experts_cnn = nn.ModuleList([CNNExpert() for _ in range(num_shared_experts)])

        self.gate_network = nn.ModuleList([
            GateNetwork(num_task_experts, self.shared_experts_cnn)
            for _ in labels_dict
        ])

        self.task_tower = nn.ModuleList([
            DNN(expert_hidden_units[-1], tower_hidden_units, activation=dnn_activation,
                 dropout_rate=dnn_dropout, use_bn=dnn_use_bn)
            for _ in labels_dict
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
