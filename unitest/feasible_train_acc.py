# 中文学习注释：实验/单元测试辅助脚本；仅增加说明，不修改原始执行逻辑。
# 脚本用途：unitest/feasible_train_acc.py。运行前核对数据路径、配置、设备及输出目录。
# 重点检查：实验变量、扫描范围、评价指标、绘图坐标/单位和结果保存位置。

import re
import numpy as np

# Specify the exact path to your log file
log_file_path = "/home/pingchua/projects/MetaONN/log/fmnist/meta_cnn/Benchmark/run-31_meta_cnn_fmnist_lr-0.002_pd-2_enc-phase_lam- 0.850_dz- 4.000_ps- 0.300_c-Exp4_layerwise_only.log"

# Regular expression pattern to extract CE loss and Accuracy
pattern = r'Feasible train set: Average loss: ([\deE\.-]+), Accuracy: (\d+)/(\d+) \(([\d\.]+)%\)'

# Lists to store loss and accuracy
loss_list = []
accuracy_list = []

# Process the specified log file
with open(log_file_path, 'r') as file:
    for line in file:
        if 'Feasible train set: Average loss' in line:
            match = re.search(pattern, line)
            if match:
                loss = float(match.group(1))
                accuracy_percent = float(match.group(4))

                loss_list.append(loss)
                accuracy_list.append(accuracy_percent)

# Convert to numpy arrays
loss_array = np.array(loss_list)
accuracy_array = np.array(accuracy_list)

loss_array = loss_array[::2]
accuracy_array = accuracy_array[::2]

print("this is the loss_array:" , loss_array)
print("this is the accuracy_array:" , accuracy_array)

# Calculate means
mean_loss = np.mean(loss_array)
mean_accuracy = np.mean(accuracy_array)

# Print results
print(f'Mean CE Loss: {mean_loss:.4e}')
print(f'Mean Accuracy: {mean_accuracy:.2f}%')