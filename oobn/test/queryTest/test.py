import matplotlib.pyplot as plt
import numpy as np

# 模拟数据
x = np.linspace(0, 500, 500)
y1 = np.sin(x / 50) + np.random.normal(0, 0.1, 500)  # 数据1
y2 = np.cos(x / 50) + np.random.normal(0, 0.1, 500)  # 数据2
y3 = np.sin(x / 100) + np.random.normal(0, 0.1, 500)  # 数据3

# 假设误差范围
error1 = np.random.uniform(0.05, 0.1, 500)
error2 = np.random.uniform(0.05, 0.1, 500)
error3 = np.random.uniform(0.05, 0.1, 500)

# 创建分组子图
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# 第一个子图：Global inference
axes[0].plot(x, y1, label="Global inference", color="blue", linewidth=1.5)
axes[0].fill_between(x, y1 - error1, y1 + error1, color="blue", alpha=0.2)  # 添加误差带
axes[0].set_title("Global inference", fontsize=12)
axes[0].set_ylabel("Inference time (ms)", fontsize=10)
axes[0].legend(loc="upper right")

# 第二个子图：Pressure_rise_limiter
axes[1].plot(x, y2, label="Pressure_rise_limiter", color="orange", linewidth=1.5, linestyle="--")
axes[1].fill_between(x, y2 - error2, y2 + error2, color="orange", alpha=0.2)  # 添加误差带
axes[1].set_title("Pressure rise limiter", fontsize=12)
axes[1].set_ylabel("Inference time (ms)", fontsize=10)
axes[1].legend(loc="upper right")

# 第三个子图：Low_protection_value
axes[2].plot(x, y3, label="Low_protection_value", color="red", linewidth=1.5, linestyle="-.")
axes[2].fill_between(x, y3 - error3, y3 + error3, color="red", alpha=0.2)  # 添加误差带
axes[2].set_title("Low protection value", fontsize=12)
axes[2].set_xlabel("The round of inference", fontsize=10)
axes[2].set_ylabel("Inference time (ms)", fontsize=10)
axes[2].legend(loc="upper right")

# 调整布局
plt.tight_layout()
plt.show()
