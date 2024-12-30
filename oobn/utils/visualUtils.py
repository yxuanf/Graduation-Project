import matplotlib.pyplot as plt
from matplotlib import rcParams


def timeComparision(tVE: list, tCTP: list, tNew: list):
    """
    Args:
        title:
        tVE:
        tCTP:
        tNew:
    """
    plt.figure()
    # 设置中文字体
    plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei']
    # rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题
    t = list(range(1, len(tVE) + 1))
    # tVE = [a + 4 for a in tVE]
    plt.plot(t, tVE, label="VE", color="#EECA40")  # 第一条曲线
    plt.plot(t, tCTP, label="CTP", color="#FD763F")  # 第二条曲线
    plt.plot(t, tNew, label="PIA-GBP", color="#23BAC5")  # 第三条曲线
    plt.yscale('log')  # 对数坐标轴：y轴
    plt.xlabel("The round of inference")
    plt.ylabel("inference time (ms)")
    # 添加图例
    plt.legend()


def detailTime(tGLO: list, *args):
    """

    Args:
        tGLO:
        **args:
    """
    n = len(args[0]) if len(args[0]) <= 3 else 3
    fig, axs = plt.subplots(n + 1, 1, figsize=(6, 8))
    # 设置中文字体
    plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei']
    t = list(range(1, len(tGLO) + 1))
    axs[0].plot(t, tGLO, label="Global Belief Propagation", color="red")
    i = 0
    for k, v in args[0].items():
        if i >= 3:
            break
        axs[i + 1].plot(t, v, label=k)
        i += 1
    # 添加 x 轴和 y 轴标签
    for ax in axs:
        ax.set_xlabel('The round of inference')
        ax.set_ylabel('inference time (ms)')

    # 在每个子图中添加图例
    for ax in axs:
        ax.legend()
    plt.tight_layout()


if __name__ == '__main__':
    pass
