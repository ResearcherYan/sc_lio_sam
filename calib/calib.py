import numpy as np

# roll = 179.048905 / 180 * np.pi
# pitch = 1.046874 / 180 * np.pi
# yaw = 2.264938 / 180 * np.pi

roll = 3.13306
pitch = 0.0176589
yaw = -0.0708897

Rx = np.array([[1, 0, 0],
               [0, np.cos(roll), -np.sin(roll)],
               [0, np.sin(roll), np.cos(roll)]])
Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
               [0, 1, 0],
               [-np.sin(pitch), 0, np.cos(pitch)]])
Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
               [np.sin(yaw), np.cos(yaw), 0],
               [0, 0, 1]])


# lidar 坐标系到 imu 坐标系的旋转变换（可以用于把 imu 下的点/向量变换到 lidar 坐标系下的矩阵）
R_lidar2imu = Rz @ Ry @ Rx
R_imu2lidar = R_lidar2imu.T
# R_lidar2imu = Rx @ Ry @ Rz

trans_imu2lidar = np.array([[0.155, 0, -0.1297]]).T  # shape: 3*1

trans_lidar2imu = - R_lidar2imu @ trans_imu2lidar

# 用 numpy 计算精度应该更高，而且位移量也是用这个矩阵算的，所以最终代码中还是用这里算出来的这个旋转矩阵，不用标定得到的 txt 文件中的旋转矩阵
print(R_lidar2imu)
print(trans_lidar2imu)
