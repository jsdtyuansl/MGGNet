import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_axes_limits(ax, x_min, x_max, y_min, y_max):
    """设置坐标轴范围"""
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])


def plot_scatter_with_regression(x, y, color, label, ax, x_seq, regression_params):
    """绘制散点图和回归线"""
    ax.scatter(x, y, s=40, alpha=0.7, color=color, edgecolors="k", linewidth=0.5, label=label)
    b, a = regression_params
    ax.plot(x_seq, a + b * x_seq, color=color, lw=0.75)


if __name__ == '__main__':
    fontname = 'Times New Roman'


    # 读取数据
    data1 = pd.read_csv("./picture/test_core2013.csv")
    data2 = pd.read_csv("./picture/test_core2016.csv")
    data3 = pd.read_csv("./picture/test_train2016.csv")
    data4 = pd.read_csv("./picture/test_val2016.csv")

    # 提取数据列
    x1, y1 = data1['label'], data1['pred']
    x2, y2 = data2['label'], data2['pred']
    x3, y3 = data3['label'], data3['pred']
    x4, y4 = data4['label'], data4['pred']

    # 计算所有数据集的坐标范围
    x_min = min(0, min(x1), min(x2), min(x3), min(x4))
    x_max = max(14, max(x1), max(x2), max(x3), max(x4))
    y_min = min(0, min(y1), min(y2), min(y3), min(y4))
    y_max = max(14, max(y1), max(y2), max(y3), max(y4))

    # 创建图表1
    fig, ax = plt.subplots(figsize=(6, 6))
    x_seq = np.linspace(x_min, x_max, num=100)
    plot_scatter_with_regression(x1, y1, 'blue', 'core_v2013', ax, x_seq, np.polyfit(x1, y1, deg=1))
    plot_scatter_with_regression(x2, y2, 'lightblue', 'core_v2016', ax, x_seq, np.polyfit(x2, y2, deg=1))
    ax.set_xlabel('True', fontsize=12,fontname=fontname)
    ax.set_ylabel('Prediction', fontsize=12,fontname=fontname)
    plt.setp(ax.get_yticklabels(), fontsize=12, fontname=fontname)
    plt.setp(ax.get_xticklabels(), fontsize=12, fontname=fontname)
    ax.tick_params(width=0.5, size=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    set_axes_limits(ax, x_min, x_max, y_min, y_max)
    ax.legend(loc='upper left',fontsize=10, prop={'family': 'Times New Roman'})
    plt.savefig("./picture/scatter_plot_combined.tif", dpi=600, bbox_inches='tight')
    plt.show()

    # 创建图表2
    fig, ax = plt.subplots(figsize=(6, 6))
    x_seq = np.linspace(x_min, x_max, num=100)
    plot_scatter_with_regression(x3, y3, 'lightblue', 'training set', ax, x_seq, np.polyfit(x3, y3, deg=1))
    ax.set_xlabel('True', fontsize=12,fontname=fontname)
    ax.set_ylabel('Prediction', fontsize=12,fontname=fontname)
    plt.setp(ax.get_yticklabels(), fontsize=12, fontname=fontname)
    plt.setp(ax.get_xticklabels(), fontsize=12, fontname=fontname)
    ax.tick_params(width=0.5, size=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    set_axes_limits(ax, x_min, x_max, y_min, y_max)
    ax.legend(loc='upper left',fontsize=10, prop={'family': 'Times New Roman'})
    plt.savefig("./picture/scatter_plot_train2016.tif", dpi=600, bbox_inches='tight')
    plt.show()

    # 创建图表3
    fig, ax = plt.subplots(figsize=(6, 6))
    x_seq = np.linspace(x_min, x_max, num=100)
    plot_scatter_with_regression(x4, y4, 'lightblue', 'validation set', ax, x_seq, np.polyfit(x4, y4, deg=1))
    ax.set_xlabel('True', fontsize=12,fontname=fontname)
    ax.set_ylabel('Prediction', fontsize=12,fontname=fontname)
    plt.setp(ax.get_yticklabels(), fontsize=12, fontname=fontname)
    plt.setp(ax.get_xticklabels(), fontsize=12, fontname=fontname)
    ax.tick_params(width=0.5, size=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    set_axes_limits(ax, x_min, x_max, y_min, y_max)
    ax.legend(loc='upper left',fontsize=10, prop={'family': 'Times New Roman'})
    plt.savefig("./picture/scatter_plot_val2016.tif", dpi=600, bbox_inches='tight')
    plt.show()