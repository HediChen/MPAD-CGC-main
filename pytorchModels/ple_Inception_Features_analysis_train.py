'''
这是MPAD-CGC模型的核心实现脚本，包含多尺度Inception模块、局部时间注意力机制、
任务特定专家和Gate网络等组件。该模型设计用于多任务时间序列异常检测，
能够有效捕捉不同时间尺度的异常模式，并通过任务特定的专家网络和Gate机制实现共享与专属特征的融合。
模型结构包括：
- 多尺度空洞卷积Inception模块：提取不同时间尺度的特征表示。
- 局部时间注意力机制：强调关键的局部时序片段。
- 任务特定专家网络：处理统计特征并进行非线性映射。
- Gate网络：融合共享专家和任务专家的输出，生成任务专属表示。
- 任务塔和输出层：进一步精炼特征并进行二分类预测。
该脚本还包含模型的前向传播逻辑，支持多任务输出，并提供L2正则化损失计算。
'''

import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List, Dict, Union, Tuple, Optional
import numpy as np
import sys
sys.path.append('./pytorchModels')
from utils import DNN
from tools.plot_featureWeight import plot_feature_weights_heatmap
from tools.plot_multiscaleFeatures import plot_metric_heatmap_and_signals

class LocalTemporalAttention(nn.Module):
    """
    Local Temporal Attention (LTA) module.
    Applies lightweight depthwise convolution-based attention across time.
    Input: (B, C, T)
    Output: (B, C, T)
    """
    # 本模块对应论文中共享专家的局部时间注意力组件（LTA），用于在时间维度上自适应强调具有判别力的局部时序片段，缓解仅用统计特征无法刻画局部混合异常的问题。
    def __init__(self, channels, kernel_size=9):
        super(LocalTemporalAttention, self).__init__()
        padding = kernel_size // 2
        # 使用 depthwise + pointwise 轻量卷积结构生成时间注意力权重图，降低参数量并保持通道独立性。
        self.attention = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels),  # Depthwise
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1),  # Pointwise
            nn.Sigmoid()  # 映射到 (0,1)，形成对时间步的加权系数
        )

    def forward(self, x):
        attn = self.attention(x)  # (B, C, T)
        return x * attn           # 将注意力系数与原特征逐元素融合，突出关键局部时序片段

class InceptionBlock(nn.Module):
    # 多尺度空洞卷积 Inception 分支，提取不同感受野的时序模式，对应论文“multiscale dilated inception”部分。
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()

        # 四个分支：不同膨胀率和卷积核，增加感受野多样性，捕获短期/中期/长期异常模式。
        self.branch1 = nn.Conv1d(in_channels, out_channels, kernel_size=9, padding=16, dilation=4)  # RF = 33 receptive fields
        self.branch2 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=4, dilation=2)   # RF = 9
        self.branch3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)   # RF = 3
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),  # 池化增强平滑与鲁棒性
            nn.Conv1d(in_channels, out_channels, kernel_size=1)  # 线性调整通道
        )

        self.bn = nn.BatchNorm1d(out_channels * 4)  # 统一归一化所有拼接后的多尺度特征
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.branch1(x)  # (B, out_channels, T)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        out = torch.cat([x1, x2, x3, x4], dim=1)  # (B, out_channels * 4, T) 多尺度特征融合
        out = self.bn(out)
        return self.relu(out)

class InceptionExpert(nn.Module):
    # 共享专家结构：顺序包含 InceptionBlock → LTA → 编码器 → 全局池化 + 线性层。
    # 对应论文中的“Multiscale dilated inception enhanced shared expert”，用于跨任务共享的多尺度时序表示抽取。
    def __init__(self, input_channels=1, output_dim=256):
        super(InceptionExpert, self).__init__()
        self.inception_layers = nn.Sequential(
            InceptionBlock(input_channels, 8),   # 输出通道四倍 → 32，多尺度初步聚合
        )

        self.LTA = nn.Sequential(
            LocalTemporalAttention(32)  # 局部时间注意力强化关键波段
        )

        # 分层编码模块：进一步下采样与通道扩展，形成更抽象的时间表示。
        self.encoder = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),  # (batch, 64, 504)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2),  # (batch, 64, 252)
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # 全局时间聚合 + 映射到共享特征向量（任务通用 embedding）
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),             # (batch, 64, 1) 全局时间压缩
            nn.Flatten(),
            nn.Linear(64, output_dim)            # 输出共享表示维度
            )

    def forward(self, x):
        x = self.inception_layers(x)  # 多尺度卷积
        x = self.LTA(x)               # 局部注意力
        x = self.encoder(x)           # 分层编码
        x = self.fc(x)                # 全局聚合 + 映射
        return x

class FeatureAttention(nn.Module):
    # 任务特定专家的统计特征自适应加权模块，对应论文中“feature analysis + task-specific refinement”。
    def __init__(self, input_dim):
        super(FeatureAttention, self).__init__()
        self.attn_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
        )
        self.softmax = nn.Softmax(dim=1)  # 针对 9 个统计特征生成归一化权重

    def forward(self, x):
        # x: (B, 9) 输入统计指标（如均值、方差、峰度等）
        scores = self.attn_layer(x)      # 学习相关性打分
        weights = self.softmax(scores)   # 归一化成注意力权重
        attended = x * weights           # 按权重重标定特征
        return attended

class GateNetwork(nn.Module):
    # GateNetwork 实现任务级的特征融合控制：将共享专家输出 + 任务特定专家输出进行加权组合。
    # 对应论文 “Feature selection and fusion via gate control mechanism”。
    def __init__(self, inputs_dim, output_dim, num_experts, shared_experts_cnn):
        super(GateNetwork, self).__init__()
        # 任务特定专家：处理统计特征（假设输入后半部分为统计特征区域），多层非线性提升判别表达。
        self.task_experts_dnn = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(),
                FeatureAttention(input_dim=inputs_dim),  # 自适应加权统计特征（这里 inputs_dim=1008，后续被切片区分）
                nn.Linear(inputs_dim, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, output_dim)  # 与共享专家输出维度对齐，便于融合
                ) for _ in range(num_experts)
                ])

        # 引用共享专家集合（多尺度时序特征抽取器）
        self.shared_experts_cnn = shared_experts_cnn

        # Gate：根据输入原始序列前段（这里用 inputs[:,:,:1008]）学习各专家权重，输出总专家数的权重向量。
        self.gates = nn.Sequential(
                nn.Flatten(),
                nn.Linear(1008, 128),
                nn.ReLU(),
                nn.Linear(128, num_experts + len(shared_experts_cnn))  # 对所有共享 + 任务专家打分
                )

    def forward(self, inputs):
        # inputs: (batch_size, 1, 1008)，假设前部分用于共享专家卷积处理，后部分可包含统计特征（代码中统一长度切片）。
        # 共享专家只看原始序列前 1008（此处与数据预处理约定相关）
        experts_output = [expert(inputs[:,:,:1008]) for expert in self.shared_experts_cnn]
        # 任务特定专家处理 inputs[:,:,1008:]（当前示例中不存在后半部分数据，逻辑仍保留用于扩展）
        experts_output += [expert(inputs[:,:,1008:]) for expert in self.task_experts_dnn]

        experts_output = torch.stack(experts_output, dim=1)  # (B, num_experts_total, D)
        # Gate 网络基于输入生成权重分布（softmax），决定每个专家对该任务的贡献
        gate_weight = torch.softmax(self.gates(inputs[:,:,:1008]), dim=-1).squeeze(1).unsqueeze(2)  # (B, num_experts_total, 1)

        # 按权重融合专家输出形成任务专属的融合表示
        return torch.sum(experts_output * gate_weight, dim=1), gate_weight

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
    # 本类实现论文中 MPAD-ICGC 的核心结构（单层版 PLE）：包括
    # 1. 共享专家（多尺度 + 注意力）
    # 2. 任务特定专家（统计特征加权 + 非线性映射）
    # 3. Gate 融合机制
    # 4. 任务塔（进一步降维与分类表示）
    # 5. 输出层（多任务二分类：是否出现对应异常模式）
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

        output_dim = expert_hidden_units[-1]  # 专家输出维度（共享与任务对齐）
        # 构建多个共享专家（多尺度时序公共表示抽取）
        self.shared_experts_cnn = nn.ModuleList([InceptionExpert(output_dim=output_dim) for _ in range(num_shared_experts)])

        # 每个任务拥有一组 GateNetwork：融合共享专家 + 任务专家输出
        self.gate_network = nn.ModuleList([
            GateNetwork(inputs_dim, output_dim, num_task_experts, self.shared_experts_cnn)
            for _ in labels_dict
        ])

        # 任务塔：对融合后的任务表示进行层次映射和特征压缩，提高分类可分性
        self.task_tower = nn.ModuleList([
            nn.Sequential(
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

        # 最终任务输出层：二分类（是否存在该模式异常），对应论文的“anomaly detection + output”
        self.task_dense = nn.ModuleList(
            [DNN(tower_hidden_units[-1], [labels_dict[name]], activation=None, bias=False) for name in labels_dict]
        )

        self.l2_reg_dnn = l2_reg_dnn  # 正则项（与论文中训练阶段的约束相关，可用于抑制过拟合）
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
        # dnn_inputs: (B, 1, 1008) 输入时间序列（可包含原始信号 + 衍生统计特征）
        outputs = OrderedDict()
        gate_weights = OrderedDict()
        for index, name in enumerate(self.labels_dict):
            # 通过对应任务的 GateNetwork 融合共享与任务特定专家表示
            tower_inputs, gate_weight = self.gate_network[index](dnn_inputs)
            # 送入任务塔进行特征精炼
            tower_output = self.task_tower[index](tower_inputs)
            # 最终分类（sigmoid 二分类概率：是否出现该类型异常）
            task_output = self.task_dense[index](tower_output)

            # outputs[name] = torch.softmax(task_output, dim=-1)    # 若为多类可切换 softmax
            outputs[name] = torch.sigmoid(task_output)  # 当前为二分类任务输出
            gate_weights[name] = gate_weight             # 可用于后续可解释分析（不同专家贡献度）

        # 返回任务输出（可扩展返回 gate_weights 以做可视化分析）
        return outputs

if __name__ == '__main__':
    import numpy as np

    # 示例：构建模型（对应两种异常检测任务：click 与 like），验证前向流程。
    model = PLE(inputs_dim=1008,
                labels_dict={"click": 2, "like": 2},
                num_shared_experts=2,
                num_task_experts=2,
                expert_hidden_units=[256])

    # 随机模拟输入：batch=4，单通道，长度=1008
    inputs = torch.FloatTensor(np.random.random([4, 1, 1008]))
    outputs = model(inputs)

    print(outputs)
    for name in outputs:
        print(name, outputs[name].shape)  # 每个任务的输出张量形状（概率或 logits）
