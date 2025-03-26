import os
from torch.utils.data import Dataset
from openstl.datasets.utils import create_loader
import numpy as np
import torch

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
    train_data = np.load(os.path.join(data_root, "processed_data", "SPI-6", "train_data_180x360.npy"))
    val_data = np.load(os.path.join(data_root, "processed_data", "SPI-6", "val_data_180x360.npy"))
    test_data = np.load(os.path.join(data_root, "processed_data", "SPI-6", "test_data_180x360.npy"))

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
