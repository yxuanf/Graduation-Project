import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

if __name__ == '__main__':
    df = pd.read_excel("./data/excel/TransferLearningTest.xlsx")
    plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei']
    fig, ax = plt.subplots()
    x = df['dataSize']
    y1 = df['KL-MLE']
    y2 = df['KL-Transfer']
    ax.plot(x, y1, color='#7DAEE0', linestyle='-')
    ax.plot(x, y2, color='#EA8379', linestyle='-')
    ax.plot(x[19:], y1[19:], label='MLE', marker='^', color='#7DAEE0')
    ax.plot(x[19:], y2[19:], label='The proposed method', marker='x', color='#EA8379')
    ax.axvline(x=712, linestyle='--', label=f'Target sample threshold：{712}', color='#9EC4BE')
    ax_inset = fig.add_axes([0.488, 0.36, 0.4, 0.3])
    x_zoom = x[:20]
    y_zoom1 = y1[:20]
    y_zoom2 = y2[:20]
    ax_inset.plot(x_zoom, y_zoom1, label='MLE', color='#7DAEE0', linestyle='-', marker='^')
    ax_inset.plot(x_zoom, y_zoom2, label='the proposed method', color='#EA8379', linestyle='-', marker='x')
    # 在主图和放大图之间加框和连接
    ax.set_xlabel('目标域样本数量', fontsize=12)
    ax.set_ylabel('KL散度', fontsize=12)
    ax.legend()
    plt.show()
