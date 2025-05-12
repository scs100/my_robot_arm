#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field

from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode


@PreTrainedConfig.register_subclass("act")
@dataclass
class ACTConfig(PreTrainedConfig):
    """Configuration class for the Action Chunking Transformers policy.

    Defaults are configured for training on bimanual Aloha tasks like "insertion" or "transfer".

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_shapes` and 'output_shapes`.

    Notes on the inputs and outputs:
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.images." they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - May optionally work without an "observation.state" key for the proprioceptive robot state.
        - "action" is required as an output key.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        chunk_size: The size of the action prediction "chunks" in units of environment steps.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
            This should be no greater than the chunk size. For example, if the chunk size size 100, you may
            set this to 50. This would mean that the model predicts 100 steps worth of actions, runs 50 in the
            environment, and throws the other 50 out.
        input_shapes: A dictionary defining the shapes of the input data for the policy. The key represents
            the input data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "observation.image" refers to an input from a camera with dimensions [3, 96, 96],
            indicating it has three color channels and 96x96 resolution. Importantly, `input_shapes` doesn't
            include batch dimension or temporal dimension.
        output_shapes: A dictionary defining the shapes of the output data for the policy. The key represents
            the output data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "action" refers to an output shape of [14], indicating 14-dimensional actions.
            Importantly, `output_shapes` doesn't include batch dimension or temporal dimension.
        input_normalization_modes: A dictionary with key representing the modality (e.g. "observation.state"),
            and the value specifies the normalization mode to apply. The two available modes are "mean_std"
            which subtracts the mean and divides by the standard deviation and "min_max" which rescale in a
            [-1, 1] range.
        output_normalization_modes: Similar dictionary as `normalize_input_modes`, but to unnormalize to the
            original scale. Note that this is also used for normalizing the training targets.
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
            `None` means no pretrained weights.
        replace_final_stride_with_dilation: Whether to replace the ResNet's final 2x2 stride with a dilated
            convolution.
        pre_norm: Whether to use "pre-norm" in the transformer blocks.
        dim_model: The transformer blocks' main hidden dimension.
        n_heads: The number of heads to use in the transformer blocks' multi-head attention.
        dim_feedforward: The dimension to expand the transformer's hidden dimension to in the feed-forward
            layers.
        feedforward_activation: The activation to use in the transformer block's feed-forward layers.
        n_encoder_layers: The number of transformer layers to use for the transformer encoder.
        n_decoder_layers: The number of transformer layers to use for the transformer decoder.
        use_vae: Whether to use a variational objective during training. This introduces another transformer
            which is used as the VAE's encoder (not to be confused with the transformer encoder - see
            documentation in the policy class).
        latent_dim: The VAE's latent dimension.
        n_vae_encoder_layers: The number of transformer layers to use for the VAE's encoder.
        temporal_ensemble_coeff: Coefficient for the exponential weighting scheme to apply for temporal
            ensembling. Defaults to None which means temporal ensembling is not used. `n_action_steps` must be
            1 when using this feature, as inference needs to happen at every step to form an ensemble. For
            more information on how ensembling works, please see `ACTTemporalEnsembler`.
        dropout: Dropout to use in the transformer layers (see code for details).
        kl_weight: The weight to use for the KL-divergence component of the loss if the variational objective
            is enabled. Loss is then calculated as: `reconstruction_loss + kl_weight * kld_loss`.
    """

    # Input / output structure.
    # 观察步数：传递给策略的环境观察步数（当前步骤和额外的历史步骤）
    # 设为1表示只使用当前观察，不使用历史观察
    n_obs_steps: int = 1

    # 动作预测块大小：每次预测的环境步骤数量
    # 较大的块大小允许模型预测更长序列的动作，提高长期规划能力
    chunk_size: int = 50

    # 执行动作步数：每次策略调用在环境中执行的动作步数
    # 不应大于chunk_size，例如可以预测100步但只执行其中的50步
    n_action_steps: int = 50

    # 归一化映射：定义不同模态数据的归一化方式
    # MEAN_STD：减去均值并除以标准差
    # MIN_MAX：将数据缩放到[-1,1]范围
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,  # 视觉数据归一化方式
            "STATE": NormalizationMode.MEAN_STD,   # 状态数据归一化方式
            "ACTION": NormalizationMode.MEAN_STD,  # 动作数据归一化方式
        }
    )

    # Architecture.
    # Vision backbone.
    # 视觉骨干网络：用于处理图像输入的ResNet变体名称
    # 使用ResNet作为特征提取器可以有效处理视觉信息
    vision_backbone: str = "resnet18"

    # 预训练骨干网络权重：用于初始化视觉骨干网络的预训练权重
    # 使用预训练权重可以加速训练并提高性能，特别是在数据有限的情况下
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"

    # 替换最终步长为空洞卷积：是否用空洞卷积替换ResNet的最终2x2步长
    # 空洞卷积可以增加感受野而不减少特征图分辨率
    replace_final_stride_with_dilation: int = False

    # Transformer layers.
    # 预归一化：是否在Transformer块中使用"pre-norm"结构
    # pre-norm结构在某些情况下可以提高训练稳定性
    pre_norm: bool = False

    # 模型维度：Transformer块的主要隐藏维度
    # 较大的维度可以增加模型容量，但也会增加计算成本
    dim_model: int = 512

    # 注意力头数：Transformer块中多头注意力的头数
    # 多头注意力允许模型关注不同的表示子空间
    n_heads: int = 8

    # 前馈网络维度：Transformer中前馈层扩展隐藏维度的大小
    # 较大的前馈维度可以增强模型的非线性表达能力
    dim_feedforward: int = 3200

    # 前馈激活函数：Transformer块前馈层中使用的激活函数
    feedforward_activation: str = "relu"

    # 编码器层数：用于Transformer编码器的层数
    # 更多的层可以处理更复杂的特征，但也增加了过拟合风险
    n_encoder_layers: int = 4

    # Note: Although the original ACT implementation has 7 for `n_decoder_layers`, there is a bug in the code
    # that means only the first layer is used. Here we match the original implementation by setting this to 1.
    # See this issue https://github.com/tonyzhaozh/act/issues/25#issue-2258740521.
    # 解码器层数：用于Transformer解码器的层数
    # 设为1是为了匹配原始ACT实现中的bug（原本设为7但只使用了第一层）
    n_decoder_layers: int = 1

    # VAE.
    # 使用VAE：是否在训练期间使用变分目标
    # VAE可以学习更好的潜在表示，有助于生成多样化的动作
    use_vae: bool = True

    # 潜在维度：VAE的潜在空间维度
    # 较小的潜在维度可以强制模型学习更紧凑的表示
    latent_dim: int = 32

    # VAE编码器层数：用于VAE编码器的Transformer层数
    n_vae_encoder_layers: int = 4

    # Inference.
    # Note: the value used in ACT when temporal ensembling is enabled is 0.01.
    # 时间集成系数：用于时间集成的指数加权方案的系数
    # 时间集成可以提高预测稳定性，通过组合多个时间步的预测
    # 设为None表示不使用时间集成，使用时通常设为0.01
    temporal_ensemble_coeff: float | None = None

    # Training and loss computation.
    # Dropout比率：在Transformer层中使用的dropout比率
    # 用于防止过拟合，较高的值会增加正则化强度
    dropout: float = 0.1

    # KL损失权重：如果启用变分目标，用于KL散度损失分量的权重
    # 控制重建损失和KL散度之间的平衡，影响VAE的正则化强度
    # 损失计算为：reconstruction_loss + kl_weight * kld_loss
    kl_weight: float = 10.0

    # Training preset
    # 优化器学习率：主要模型参数的学习率
    # 较小的学习率有助于稳定训练，但可能导致收敛较慢
    optimizer_lr: float = 1e-5

    # 优化器权重衰减：用于L2正则化的权重衰减系数
    # 有助于防止过拟合，特别是在参数较多的模型中
    optimizer_weight_decay: float = 1e-4

    # 骨干网络学习率：视觉骨干网络的学习率
    # 通常与主学习率相同或更小，因为预训练的骨干网络可能需要较小的更新
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
