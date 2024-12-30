import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

if __name__ == '__main__':
    df = pd.read_excel("./chapter3/data/excel/消融实验1.xlsx")
    plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei']
    fig, ax = plt.subplots()
    x = df['datasize']
    y1 = df['noS']
    y2 = df['noP']
    y3 = df['both']

    ax.plot(x, y1, label='仅考虑参数相似性', marker='^', color='#2C91E0')
    ax.plot(x, y2,  label='仅考虑结构相似性', marker='o',color='#3ABF99')
    ax.plot(x, y3, label='综合考虑', marker='x', color='#EA8379')
    # 在主图和放大图之间加框和连接
    ax.set_xlabel('目标域样本数量', fontsize=12)
    ax.set_ylabel('KL散度', fontsize=12)
    ax.legend()
    plt.show()
