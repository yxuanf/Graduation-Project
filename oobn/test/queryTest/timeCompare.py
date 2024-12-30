import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def timeCompare(path: str):
    """

    Args:
        path:
    """
    df = pd.read_excel(path)
    x = df["nums"].to_numpy()
    yCTP = df["CTP"].to_numpy()
    y_new = df["PIA-GBS"].to_numpy()
    y_VE = df["VE"].to_numpy()

    plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei']
    fig, ax1 = plt.subplots()
    ax1.plot(x, y_new, label='PIA-GBP', marker='*',color="#23BAC5")  # 蓝色实线
    ax1.plot(x, y_VE, label='VE', marker='x',color="#EECA40")  # 红色实线
    ax1.set_xlabel('Number of inference nodes', fontsize=12)
    ax1.set_ylabel('Time of PIA-GBP and VE (ms)', fontsize=12)  # 设置右侧 Y 轴标签

    # 第二个坐标轴（右侧，公用同一个 X 轴）
    ax2 = ax1.twinx()
    ax2.plot(x, yCTP, label='CTP', marker='^',color="#FD763F")
    ax2.set_xlabel('Number of inference nodes', fontsize=12)
    ax2.set_ylabel('Time of CTP (ms)', fontsize=12)
    ax2.set_ylim(1, np.max(yCTP) + 10)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.legend()
    # 设置布局
    fig.legend(loc='upper left', bbox_to_anchor=(1, 0), ncol=1)
    # fig.suptitle('Algorithm time-consuming comparison', fontsize=14)
    # 调整图形的边界，以留出空间给图例
    plt.subplots_adjust(right=0.8)


if __name__ == '__main__':
    timeCompare("./data/excel/model-1推理时间随节点的变化.xlsx")
    plt.show()
