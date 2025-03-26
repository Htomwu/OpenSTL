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
    output_dir = "data/processed_data/SPEI-6"
    file_path = os.path.join("data", "era5", "Monthly_SPEI-6_from_ERA5_195901-202212.nc")
    data = xr.open_dataset(file_path)['spei'].values  # 假设 'spi' 是变量名

    # 步骤 1: 删除时间维度上全为 NaN 的值
    data = data[~np.isnan(data).all(axis=(1, 2))]

    # 步骤 2: 用均值填充剩余的 NaN 值
    data = np.where(np.isnan(data), np.nanmean(data, axis=(0, 1), keepdims=True), data)

    # 步骤 3: 插值 - 将数据从 (360, 720) 缩小到 (180, 360)
    time_steps, lat, lon = data.shape

    print("预处理完成")

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

    print("插值完成")

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


    np.save(os.path.join(output_dir, "train_data_180x360.npy"), train_data)
    np.save(os.path.join(output_dir, "val_data_180x360.npy"), val_data)
    np.save(os.path.join(output_dir, "test_data_180x360.npy"), test_data)

    print("done")