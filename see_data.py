import numpy as np
import os
import matplotlib.pyplot as plt

train_data = np.load(os.path.join("data", "processed_data","SPI-6", "train_data_256x256.npy"))
test_data = np.load(os.path.join("data", "processed_data", "SPI-6","test_data_256x256.npy"))
valid_data = np.load(os.path.join("data", "processed_data", "SPI-6","val_data_256x256.npy"))

print(train_data.shape)

# 可视化处理后的第一张图（第一个时间步）
plt.figure(figsize=(10, 10))
plt.imshow(train_data[0, 0], origin='lower', extent=[-180, 180, -90, 90], cmap='coolwarm')
plt.colorbar(label='Normalized SPI-3')
plt.title('First Time Step of Resized and Normalized SPI-3 Data')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(False)
plt.show()
