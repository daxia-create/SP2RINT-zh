# 中文学习注释：实验/单元测试辅助脚本；仅增加说明，不修改原始执行逻辑。
# 脚本用途：unitest/phase_mask_plot.py。运行前核对数据路径、配置、设备及输出目录。
# 重点检查：实验变量、扫描范围、评价指标、绘图坐标/单位和结果保存位置。

import torch
import matplotlib.pyplot as plt
import numpy as np
import random

if __name__ == "__main__":
    # Create a 2D tensor with random values
    tm = torch.zeros(32, 32)
    for idx in range(4):
        for i in range(32):
            random_phaes = random.uniform(0, 2 * np.pi)
            tm[i, i] = random_phaes
        plt.figure()
        plt.imshow(tm.numpy())
        plt.savefig(f"./figs/phase_mask_{idx}.png")
        plt.close()