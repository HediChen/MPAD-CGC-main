import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List, Dict, Tuple, Union, Optional

class TransformerExpert(nn.Module):
    def __init__(self, input_channels=1, output_dim=256, nhead=4, num_layers=2, d_model=64):
        super(TransformerExpert, self).__init__()
        
        self.input_channels = input_channels
        self.output_dim = output_dim
        
        # Define the transformer architecture
        self.embedding = nn.Linear(input_channels, d_model)  # Embed input to d_model
        
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, output_dim)  # Output layer to project to the required dimension

    def forward(self, x):
        # x is of shape (batch, channels, length), so we permute to (length, batch, channels)
        x = x.permute(2, 0, 1)  # (length, batch, input_channels)
        
        # Apply embedding
        x = self.embedding(x)
        
        # Pass through Transformer Encoder
        x = self.transformer_encoder(x)
        
        # Take the last output of the transformer (or average pooling over all outputs)
        x = x[-1, :, :]  # Take the last token's representation
        
        # Pass through the fully connected layer
        x = self.fc(x)
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

        # Transformer-based Experts
        self.experts = nn.ModuleList([TransformerExpert(input_channels=1, output_dim=output_dim)
                                      for _ in range(num_experts)])

        # Gating Networks still take flattened input
        self.gate_dnn = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(1008, num_experts)
            ) for _ in labels_dict
        ])

        # Task Towers and final output layers
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

        # Each Transformer expert processes the 1D input
        experts_output = [expert(x) for expert in self.experts]

        for idx, name in enumerate(self.labels_dict):
            gate = self.gate_dnn[idx](x)
            tower_input = _merge_experts_with_gate(experts_output, gate)
            tower_output = self.task_tower[idx](tower_input)
            task_output = self.task_dense[idx](tower_output)
            outputs[name] = torch.sigmoid(task_output)

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
                 expert_hidden_units=[256])  # Output dim from Transformer expert

    dummy_input = torch.randn(4, 1, 1008)  # (batch_size, channels, length)
    outputs = model(dummy_input)

    for name in outputs:
        print(f"{name}: {outputs[name].shape}")
