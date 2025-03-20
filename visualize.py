import numpy as np
from openstl.utils import show_video_line

# show the given frames from an example
inputs = np.load('./work_dirs/SPI-6-TAU/saved/inputs.npy')
preds = np.load('./work_dirs/SPI-6-TAU/saved/preds.npy')
trues = np.load('./work_dirs/SPI-6-TAU/saved/trues.npy')

import matplotlib.pyplot as plt
import numpy as np


def show_model_prediction(inputs, pred, true, vmax=0.6, vmin=0.0, cmap='RdYlBu', norm=None, cbar=False, format='png',
                          out_path=None, dpi=500):
    """ 显示 4 个输入帧、1 个预测帧 和 1 个真实值帧，共 6 张子图，高分辨率 """

    ncols = 6  # 6 张子图
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(25, 5))  # 增加 figsize

    # 确保数据格式正确
    inputs = np.array(inputs)  # (4, H, W)
    pred = np.array(pred)  # (H, W)
    true = np.array(true)  # (H, W)

    # 合并数据，形成 (6, H, W) 格式
    all_frames = np.concatenate([inputs, [pred], [true]], axis=0)

    # 设置子图标题
    titles = ["Input t=0", "Input t=1", "Input t=2", "Input t=3", "Prediction t=4", "True t=4"]

    images = []
    for t, ax in enumerate(axes.flat):
        im = ax.imshow(all_frames[t], cmap=cmap)

        # 设置标题（减小字体）
        ax.set_title(titles[t])

        # 设置坐标轴间隔
        ax.set_xticks(np.arange(0, all_frames[t].shape[1], 100))  # 横坐标间隔 100
        ax.set_yticks(np.arange(0, all_frames[t].shape[0], 50))   # 纵坐标间隔 50

        # **让坐标轴更细**
        ax.spines['top'].set_linewidth(0.3)
        ax.spines['bottom'].set_linewidth(0.3)
        ax.spines['left'].set_linewidth(0.3)
        ax.spines['right'].set_linewidth(0.3)

        # **设置坐标轴字体大小**
        # ax.tick_params(axis='both', labelsize=2, width=0.3, length=2)

        images.append(im)

    # 可选地添加颜色条
    if cbar:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # 颜色条放右侧
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, cax=cbar_ax)

    # # 自动优化布局
    # plt.tight_layout()

    # 显示图片
    plt.show()

    # 保存高分辨率图片
    if out_path is not None:
        fig.savefig(out_path, format=format, dpi=dpi, pad_inches=0, bbox_inches='tight')

    plt.close()


# 打印数据形状，确认维度
print("Inputs shape:", inputs.shape)  # 预计是 (B, T, C, H, W)
print("Preds shape:", preds.shape)    # 预计是 (B, T, C, H, W)
print("Trues shape:", trues.shape)    # 预计是 (B, T, C, H, W)

# 取 batch 内第一组数据
inputs = inputs[0]  # (T, C, H, W)
pred = preds[0, 0]  # (C, H, W)
true = trues[0, 0]  # (C, H, W)

# 可能需要 squeeze() 以去掉通道维度
inputs = inputs.squeeze(1)  # (T, H, W)
pred = pred.squeeze(0)  # (H, W)
true = true.squeeze(0)  # (H, W)

# 进行可视化
show_model_prediction(inputs, pred, true)
