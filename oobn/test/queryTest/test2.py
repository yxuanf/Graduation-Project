import matplotlib.pyplot as plt
import numpy as np

# 生成 20 组随机数据
data = [np.random.normal(loc=np.random.randint(10, 100), scale=10, size=100) for _ in range(20)]

# 绘制箱型图
plt.figure(figsize=(12, 6))  # 设置图形大小
plt.boxplot(data, labels=[f"Group {i+1}" for i in range(20)], showfliers=True)

# 添加标题和坐标轴标签
plt.title("Box Plot for 20 Groups")
plt.xlabel("Groups")
plt.ylabel("Values")

# 旋转 X 轴标签避免重叠
plt.xticks(rotation=45)

# 显示图形
plt.show()
