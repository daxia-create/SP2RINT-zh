# 中文学习注释：保持原始实现不变，仅添加阅读辅助说明。
# 模块路径：core/models/layers/quantized_linear.py；请结合调用处核对输入/输出形状、设备、数据类型和梯度流。

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.registry import MODELS
from pyutils.general import logger
# from pyutils.quant.lsq import ActQuantizer_LSQ
from torch import Tensor, nn
from torch.nn import Parameter
from torch.types import Device

from .quantized_base_layer import QBaseLayer
from .utils import (
    WeightQuantizer_LSQ,
    ActQuantizer_LSQ,
)

__all__ = [
    "QLinear",
]

# MODELS.register_module(name="Linear", module=nn.Linear)


@MODELS.register_module()
# [中文注释] 定义 QLinear 类，封装模块状态、参数和计算方法。
class QLinear(QBaseLayer):
    """
    Quantized Linear layer.
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor
    __annotations__ = {"bias": Optional[Tensor]}

    # [中文注释] 初始化模块参数和内部状态，检查默认值及张量维度。
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        w_bit: int = 8,
        in_bit: int = 8,
        out_bit: int = 8,
        device: Device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    ):
        super().__init__(device=device)
        self.in_features = in_features
        self.out_features = out_features

        self.w_bit = w_bit
        self.in_bit = in_bit
        self.out_bit = out_bit


        ### build trainable parameters
        self.build_parameters()
        ## quantization tool
        self.input_quantizer = ActQuantizer_LSQ(
            None,
            device=self.device,
            nbits=self.in_bit,
            offset=True,
            signed=False,
            mode="tensor_wise",
        )
        self.weight_quantizer = WeightQuantizer_LSQ(
            None,
            device=self.device,
            nbits=self.w_bit,
            offset=False,
            signed=True,
            mode="tensor_wise",
        )
        self.output_quantizer = ActQuantizer_LSQ(
            None,
            device=self.device,
            nbits=self.out_bit,
            offset=True,
            signed=False,
            mode="tensor_wise",
        )

        ### default set to no noise
        self.set_input_noise(0)
        self.set_weight_noise(0)
        self.set_output_noise(0)
        
        self.set_input_bitwidth(self.in_bit)
        self.set_weight_bitwidth(self.w_bit)
        self.set_output_bitwidth(self.out_bit)

        if bias:
            self.bias = Parameter(torch.Tensor(out_features).to(self.device))
        else:
            self.register_parameter("bias", None)

        self.prune_mask = None
        self.reset_parameters()

    # [中文注释] 实现 build_parameters；结合调用处确认功能、输入输出及状态变化。
    def build_parameters(self):
        self.weight = nn.Linear(
            self.in_features, 
            self.out_features, 
            bias=False,
            device=self.device,
        ).weight
        

    # [中文注释] 实现 reset_parameters；结合调用处确认功能、输入输出及状态变化。
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    # [中文注释] 实现 _forward_impl；结合调用处确认功能、输入输出及状态变化。
    def _forward_impl(self, x: Tensor, weight:Tensor) -> Tensor:
        # import pdb; pdb.set_trace()
        x = F.linear(
            x,
            weight,
            bias=self.bias,
        )
        return x

    # [中文注释] 实现 extra_repr；结合调用处确认功能、输入输出及状态变化。
    def extra_repr(self):
        s = f"{self.in_features}, {self.out_features}"
        # s += f", bias={1 if self.bias != 0 else 0}"
        s += f", in_bits={self.in_bit}, w_bits={self.w_bit}, out_bits={self.out_bit}"
        s += f", input_noise_std={self.input_noise_std}, weight_noise_std={self.weight_noise_std}, output_noise_std={self.output_noise_std}"
        return s
