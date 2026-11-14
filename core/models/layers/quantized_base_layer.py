# 中文学习注释：保持原始实现不变，仅添加阅读辅助说明。
# 模块路径：core/models/layers/quantized_base_layer.py；请结合调用处核对输入/输出形状、设备、数据类型和梯度流。

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from pyutils.compute import add_gaussian_noise
from torch import Tensor, nn
# from torch.nn import Parameter, init
from torch.types import Device
from pyutils.quant.lsq import ActQuantizer_LSQ
from .utils import (
    WeightQuantizer_LSQ,
)

# from .utils import STE, mzi_out_diff_to_phase, mzi_phase_to_out_diff, partition_chunks

__all__ = ["QBaseLayer"]


# [中文注释] 定义 QBaseLayer 类，封装模块状态、参数和计算方法。
class QBaseLayer(nn.Module):
    # [中文注释] 初始化模块参数和内部状态，检查默认值及张量维度。
    def __init__(self, *args, device: Device = torch.device("cpu"), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # cuda or cpu, defaults to cpu
        self.device = device

    # [中文注释] 实现 set_input_noise；结合调用处确认功能、输入输出及状态变化。
    def set_input_noise(self, noise_std: float = 0.0) -> None:
        self.input_noise_std = noise_std
    
    # [中文注释] 实现 set_weight_noise；结合调用处确认功能、输入输出及状态变化。
    def set_weight_noise(self, noise_std: float = 0.0) -> None:
        self.weight_noise_std = noise_std

    # [中文注释] 实现 set_output_noise；结合调用处确认功能、输入输出及状态变化。
    def set_output_noise(self, noise_std: float = 0.0) -> None:
        self.output_noise_std = noise_std



    # [中文注释] 实现 set_input_bitwidth；结合调用处确认功能、输入输出及状态变化。
    def set_input_bitwidth(self, in_bit: int) -> None:
        self.in_bit = in_bit
        self.input_quantizer.set_bit(in_bit)

    # [中文注释] 实现 set_weight_bitwidth；结合调用处确认功能、输入输出及状态变化。
    def set_weight_bitwidth(self, w_bit: int) -> None:
        self.w_bit = w_bit
        self.weight_quantizer.set_bit(w_bit)
        
    # [中文注释] 实现 set_output_bitwidth；结合调用处确认功能、输入输出及状态变化。
    def set_output_bitwidth(self, out_bit: int) -> None:
        self.out_bit = out_bit
        self.output_quantizer.set_bit(out_bit)

    # def _add_output_noise(self, x) -> None:
    #     if self.output_noise_std > 1e-6:
    #         # vector_len = np.prod(self.weight.shape[1::2])  # q*c*k2
    #         noise = gen_gaussian_noise(
    #             x,
    #             noise_mean=0,
    #             noise_std=np.sqrt(vector_len) * self.output_noise_std,
    #         )

    #         x = x + noise
    #     return x
    # [中文注释] 实现 weight_transform；结合调用处确认功能、输入输出及状态变化。
    def weight_transform(self, weight: Tensor) -> Tensor:
        # if self.w_bit < 16:
        #     weight = self.weight_quantizer(weight)  # [-alpha, alpha]
        #     alpha = self.weight_quantizer.alpha.data
        # else:
        #     alpha = weight.data.abs().max()
        # self.weight_scale = alpha.item()

        # if self.weight_noise_std > 0:
        #     ## Warning: noise need to be added to normalized input
        #     weight = add_gaussian_noise(
        #         weight, noise_std=self.weight_noise_std * self.weight_scale
        #     )
        return weight
    
    # [中文注释] 实现 input_transform；结合调用处确认功能、输入输出及状态变化。
    def input_transform(self, x: Tensor) -> Tensor:
        # if self.in_bit < 16:
        #     x = self.input_quantizer(x)  # [-alpha, alpha] or [0, alpha]
        #     alpha = self.input_quantizer.alpha.data
        # else:
        #     alpha = x.data.abs().max()
        # self.input_scale = alpha.item()

        # if self.input_noise_std > 0:
        #     x = add_gaussian_noise(x, noise_std=self.input_noise_std * self.input_scale)

        return x
    
    # [中文注释] 实现 output_transform；结合调用处确认功能、输入输出及状态变化。
    def output_transform(self, x: Tensor) -> Tensor:
        # if self.output_noise_std > 0:
        #     if hasattr(self, 'in_features'):
        #         in_dim = self.in_features
        #     elif hasattr(self, 'in_channels'):
        #         in_dim = self.in_channels
        #     else:
        #         in_dim = x.size(-1)  # Number of columns
        #     x = add_gaussian_noise(
        #         x,
        #         noise_std=self.output_noise_std
        #         * self.input_scale
        #         * self.weight_scale
        #         * (in_dim ** 0.5)
        #     )

        # if self.out_bit < 16:
        #     x = self.output_quantizer(x)

        return x

    # [中文注释] 前向计算：跟踪输入输出张量形状及可微运算。
    def forward(self, x:Tensor) -> Tensor:
        
        # try:
        # ## preprocess input
        #     x = self.input_transform(x)
        # except:
        #     import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        x = self.input_transform(x)
        ## preprocess weights
        weight = self.weight_transform(self.weight)

        # import pdb; pdb.set_trace()
        ## forward pass
        out = self._forward_impl(x, weight)

        ## postprocess output
        out = self.output_transform(out)
        
        ## add bias
        if self.bias is not None:
            out = out + self.bias

        return out
    
    # [中文注释] 实现 _forward_impl；结合调用处确认功能、输入输出及状态变化。
    def _forward_impl(self, x: Tensor, weights: Dict[str, Tensor]) -> Tensor:
        raise NotImplementedError 
       
    # [中文注释] 实现 extra_repr；结合调用处确认功能、输入输出及状态变化。
    def extra_repr(self) -> str:
        return ""
