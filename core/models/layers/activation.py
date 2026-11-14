# 中文学习注释：保持原始实现不变，仅添加阅读辅助说明。
# 模块路径：core/models/layers/activation.py；请结合调用处核对输入/输出形状、设备、数据类型和梯度流。

import torch
import torch.nn as nn
import torch.nn.functional as F

# [中文注释] 定义 GeLU 类，封装模块状态、参数和计算方法。
class GeLU(nn.Module):
    # [中文注释] 前向计算：跟踪输入输出张量形状及可微运算。
    def forward(self, x):
        return F.gelu(x)