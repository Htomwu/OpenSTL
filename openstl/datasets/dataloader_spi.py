import os
import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import zoom
from openstl.datasets.utils import create_loader
from sklearn.model_selection import train_test_split


class SpiDataset(Dataset):
    """Taxibj <https://arxiv.org/abs/1610.00081>`_ Dataset"""

    def __init__(self, data, pre_seq_length, aft_seq_length, data_name='spi'):
        """
        Args:
            data: 预处理后的 SPI 数据，形状为 (time_steps, 1, lat, lon)
            pre_seq_length: 预测所需的历史时间步长
            aft_seq_length: 需要预测的未来时间步长
        """
        super(SpiDataset, self).__init__()
        self.data = data
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.data_name = data_name
        self.mean = torch.tensor(self.data).mean().item()
        self.std = torch.tensor(self.data).std().item()

        # 确保数据长度足够
        assert self.data.shape[0] > self.pre_seq_length + self.aft_seq_length, "数据集太短，无法构造样本"


    def __len__(self):
        return self.data.shape[0] - self.pre_seq_length - self.aft_seq_length + 1

    def __getitem__(self, index):
        """返回 (历史数据, 未来数据)"""
        data = self.data[index:index + self.pre_seq_length]  # 输入数据
        labels = self.data[index + self.pre_seq_length: index + self.pre_seq_length + self.aft_seq_length]  # 目标输出

        return torch.tensor(data).float(), torch.tensor(labels).float()


def load_data(batch_size, val_batch_size, data_root, num_workers=4,
              pre_seq_length=None, aft_seq_length=None, in_shape=None,
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False):

    # 加载数据
    file_path = os.path.join(data_root, "era5", "Monthly_SPI-3_from_ERA5_195901-202212.nc")
    data = xr.open_dataset(file_path)['spi'].values  # 假设 'spi' 是变量名

    # 步骤 1: 删除时间维度上全为 NaN 的值
    data = data[~np.isnan(data).all(axis=(1, 2))]

    # 步骤 2: 用均值填充剩余的 NaN 值
    data = np.where(np.isnan(data), np.nanmean(data, axis=(0, 1), keepdims=True), data)

    # 步骤 3: 插值 - 将数据从 (360, 720) 缩小到 (180, 360)
    time_steps, lat, lon = data.shape

    # 设置目标分辨率
    new_lat = 180
    new_lon = 360

    # 创建原始的经纬度网格
    latitudes = np.linspace(-90, 90, lat)
    longitudes = np.linspace(-180, 180, lon)

    # 创建新的目标经纬度网格
    new_latitudes = np.linspace(-90, 90, new_lat)
    new_longitudes = np.linspace(-180, 180, new_lon)

    # 新数组用于存储调整后的数据
    resized_data = np.zeros((time_steps, new_lat, new_lon))  # 新数据数组形状为 (time_steps, 180, 360)

    # 使用 zoom 或插值方法将每个时间步的数据调整到新分辨率
    for t in range(time_steps):

        # 获取当前时间步的二维数据
        current_data = data[t, :, :]

        # 进行下采样
        interpolated_data = zoom(current_data, (new_lat / lat, new_lon / lon), order=3)

        # 确保插值后的数据形状为 (180, 360)
        assert interpolated_data.shape == (new_lat, new_lon), f"插值后的数据形状不匹配: {interpolated_data.shape}"

        # 将插值后的数据存入新的数组
        resized_data[t] = interpolated_data  # 直接替换当前时间步的数据

    # 步骤 4: 归一化 - 使用 MinMaxScaler 对数据进行归一化
    scaler = MinMaxScaler()
    time_steps, lat, lon = resized_data.shape  # 更新尺寸

    # 扁平化数据进行归一化
    resized_data = scaler.fit_transform(resized_data.reshape(time_steps, -1)).reshape(time_steps, lat, lon)

    # 步骤 5: 扩展数据维度以适应 ConvLSTM 输入 (batch, time, channels, height, width)
    resized_data = np.expand_dims(resized_data, axis=1)  # 增加 channel 维度，使得数据形状变为 (time_steps, 1, lat, lon)

    # 步骤 6: 划分数据集
    train_data, val_data = train_test_split(resized_data, test_size=0.2, shuffle=False)
    test_data = val_data

    train_set = SpiDataset(train_data, pre_seq_length, aft_seq_length)
    val_set = SpiDataset(val_data, pre_seq_length, aft_seq_length)
    test_set = SpiDataset(test_data, pre_seq_length, aft_seq_length)

    # 创建数据加载器
    dataloader_train = create_loader(train_set, batch_size=batch_size, shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True, num_workers=num_workers,
                                     distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_val = create_loader(val_set, batch_size=val_batch_size, shuffle=False, is_training=False,
                                   pin_memory=True, drop_last=drop_last, num_workers=num_workers,
                                   distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_test = create_loader(test_set, batch_size=val_batch_size, shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last, num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)

    return dataloader_train, dataloader_val, dataloader_test


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torch
    import numpy as np

    dataloader_train, _, dataloader_test = \
        load_data(batch_size=4,
                  val_batch_size=4,
                  data_root='../../data/',
                  num_workers=4,
                  pre_seq_length=4, aft_seq_length=1)

    print(f"Train batches: {len(dataloader_train)}, Test batches: {len(dataloader_test)}")

    # 获取测试数据的真实值
    test_data_true = []

    for batch in dataloader_test:
        _, labels = batch  # 假设 dataloader 返回 (输入数据, 真实标签)
        test_data_true.append(labels.numpy())  # 转换为 NumPy 数组

    # 合并所有批次的数据
    test_data_true = np.concatenate(test_data_true, axis=0)  # 确保维度对齐

    # 展平数据
    flattened_data = test_data_true.flatten()

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(flattened_data, bins=100, alpha=0.75, edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of data on dataloader')
    plt.grid(True)
    plt.show()

    print(dataloader_test.dataname)
