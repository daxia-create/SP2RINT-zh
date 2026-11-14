# 中文学习注释：实验/单元测试辅助脚本；仅增加说明，不修改原始执行逻辑。
# 脚本用途：unitest/test_fft_component.py。运行前核对数据路径、配置、设备及输出目录。
# 重点检查：实验变量、扫描范围、评价指标、绘图坐标/单位和结果保存位置。

import torch
import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 480  # Signal length
sampling_rate = 480  # Sampling rate (assume 480 Hz)

# Compute FFT frequency axis
freqs = torch.fft.fftfreq(L, d=1/sampling_rate)

# print("this is freqs", freqs)
# quit()

# Select some frequency indices (mix of low and high frequencies)
selected_indices = torch.tensor([-1, -2, -3, -4, -5])  

# Generate time axis
t = np.arange(L) / sampling_rate  # Normalize time axis (seconds)

# Plot setup
plt.figure(figsize=(10, 5))

# Process each selected frequency separately
for idx in selected_indices:
    # Initialize frequency-domain signal (all components zero)
    fft_signal = torch.zeros(L, dtype=torch.complex64)
    fft_signal[idx] = 1 + 0j  # Set only one frequency component to 1

    # Compute IFFT to get time-domain signal
    time_signal = torch.fft.ifft(fft_signal).real  # Take only real part

    # Plot current frequency component waveform
    plt.plot(t, time_signal.numpy(), label=f"Freq {freqs[idx]:.1f} Hz")

# Add legend and labels
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("IFFT of Selected Frequency Components (Separated)")
plt.legend()
plt.grid()

# Save the image
plt.savefig("./figs/ifft_waveform_separated.png", dpi=300)

