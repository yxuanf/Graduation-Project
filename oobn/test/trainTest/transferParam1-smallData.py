import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_excel("./data/excel/TransferLearn-size100.xlsx")
# 显示数据的前几行，查看数据结构
print(df.head())
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei']
# 绘制折线图
# 绘制 y1 列数据的折线图
plt.plot(df['dataSize'], df['KL-MLE'], marker='^', label='MLE', color='b', linestyle='-')
# 绘制 y2 列数据的折线图
plt.plot(df['dataSize'], df['KL-Transfer'], marker='x', label='OURs', color='r', linestyle='-')
# plt.axvline(x=712, linestyle='--', color='y', label=f'触发边界 data size={648}')
# 添加标题和标签
# plt.title('折线图示例：x 与 y1, y2 的关系')
plt.xlabel('目标域样本数量')
plt.ylabel('KL散度')
plt.title("")
# 显示图例
plt.legend()

# 显示图形
plt.show()
