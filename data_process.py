import os
import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # 加载数据
    output_dir = "data/processed_data/SPI-6"
    file_path = os.path.join("data", "era5", "Monthly_SPI-6_from_ERA5_195901-202212.nc")
    data = xr.open_dataset(file_path)['spi'].values  # 假设 'spei' 是变量名

    # 步骤 1: 删除时间维度上全为 NaN 的值
    data = data[~np.isnan(data).all(axis=(1, 2))]

    # 步骤 2: 用均值填充剩余的 NaN 值
    data = np.where(np.isnan(data), np.nanmean(data, axis=(0, 1), keepdims=True), data)

    # 步骤 3: 插值 - 将数据从 (360, 720) 缩放到 (256, 256)
    time_steps, lat, lon = data.shape

    print("预处理完成")

    # 设置目标分辨率
    new_lat = 256
    new_lon = 256

    # 新数组用于存储调整后的数据
    resized_data = np.zeros((time_steps, new_lat, new_lon))

    # 使用 zoom 进行缩放
    for t in range(time_steps):
        current_data = data[t, :, :]
        interpolated_data = zoom(current_data, (new_lat / lat, new_lon / lon), order=3)
        assert interpolated_data.shape == (new_lat, new_lon), f"插值后的数据形状不匹配: {interpolated_data.shape}"
        resized_data[t] = interpolated_data

    print("插值完成")

    # 步骤 4: 归一化
    scaler = MinMaxScaler()
    resized_data = scaler.fit_transform(resized_data.reshape(time_steps, -1)).reshape(time_steps, new_lat, new_lon)

    # 步骤 5: 扩展维度为 ConvLSTM 格式
    resized_data = np.expand_dims(resized_data, axis=1)  # (time_steps, 1, lat, lon)

    # 步骤 6: 划分数据集
    train_data, val_data = train_test_split(resized_data, test_size=0.2, shuffle=False)
    test_data = val_data

    # 保存数据
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "train_data_256x256.npy"), train_data)
    np.save(os.path.join(output_dir, "val_data_256x256.npy"), val_data)
    np.save(os.path.join(output_dir, "test_data_256x256.npy"), test_data)

    print("done")
