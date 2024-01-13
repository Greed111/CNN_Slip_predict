import numpy as np
import xarray as xr

# 数组的坐标是纬度从负到正，经度从小到大

data_address_u = xr.open_dataset('E:/BaiduNetDOWNLORD/ORAS3/ORAS3taux.nc')
data_address_v = xr.open_dataset('E:/BaiduNetDOWNLORD/ORAS3/ORAS3tauy.nc')
data_address_ssh = xr.open_dataset('E:/BaiduNetDOWNLORD/ORAS3/ORAS3SSH.nc')
data_address_t05 = xr.open_dataset('E:/BaiduNetDOWNLORD/ORAS3/ORAS3temp05m.nc')
data_address_t25 = xr.open_dataset('E:/BaiduNetDOWNLORD/ORAS3/ORAS3temp25m.nc')
data_address_t45 = xr.open_dataset('E:/BaiduNetDOWNLORD/ORAS3/ORAS3temp45m.nc')


# 读取数据，xarray中的矩阵
windu = data_address_u['TAUX'].values
windv = data_address_v['TAUY'].values
sealevel = data_address_ssh['SSH'].values
t05 = data_address_t05['TEMP'].values
t05 = np.squeeze(t05)
t25 = data_address_t25['TEMP'].values
t25 = np.squeeze(t25)
t45 = data_address_t45['TEMP'].values
t45 = np.squeeze(t45)

mask = np.isnan(windu)
windu[mask] = 0
windv[mask] = 0
sealevel[mask] = 0
t05[mask] = 0
t25[mask] = 0
t45[mask] = 0


sealevel_mean = np.mean(sealevel, axis=0)
print(sealevel_mean[0:2,0],
      sealevel[0,0:2,0])
for i in range(612):
    sealevel[i, :, :] = np.squeeze(sealevel[i, :, :]) - sealevel_mean
print(sealevel[0,0:2,0])

num_tr = 43*12  # 43年训练
num_ts = 9*12-1  # 9年检验,没最后一年，所以需要后10年数据
para_input_train = np.zeros((num_tr, 72, 31, 156), dtype=np.float32)
para_input_test = np.zeros((num_ts+1, 72, 31, 156), dtype=np.float32)
ssh_pre_train = np.zeros((num_tr, 6, 31, 156), dtype=np.float32)
ssh_pre_test = np.zeros((num_ts+1, 6, 31, 156), dtype=np.float32)

# 保存所有train部分的x和y数据
m = 0
for i in range(num_tr):

    # 储存目标因变量数据
    ssh_pre_train[i, 0, :, :] = windu[i+12, :, :]
    ssh_pre_train[i, 1, :, :] = windv[i+12, :, :]
    ssh_pre_train[i, 2, :, :] = sealevel[i+12, :, :]
    ssh_pre_train[i, 3, :, :] = t05[i+12, :, :]
    ssh_pre_train[i, 4, :, :] = t25[i+12, :, :]
    ssh_pre_train[i, 5, :, :] = t45[i+12, :, :]

    # 储存样本自变量数据
    for j in range(12):
        para_input_train[i, m, :, :] = windu[i + j, :, :]
        para_input_train[i, m + 1, :, :] = windv[i + j, :, :]
        para_input_train[i, m + 2, :, :] = sealevel[i + j, :, :]
        para_input_train[i, m + 3, :, :] = t05[i + j, :, :]
        para_input_train[i, m + 4, :, :] = t25[i + j, :, :]
        para_input_train[i, m + 5, :, :] = t45[i + j, :, :]
        m = m + 6
    m = 0

# 保存所有test部分的x和y数据
for i in range(num_ts):
    # 测试集，与上述一致
    ssh_pre_test[i, 0, :, :] = windu[i + 12 + 492, :, :]
    ssh_pre_test[i, 1, :, :] = windv[i + 12 + 492, :, :]
    ssh_pre_test[i, 2, :, :] = sealevel[i + 12 + 492, :, :]
    ssh_pre_test[i, 3, :, :] = t05[i + 12 + 492, :, :]
    ssh_pre_test[i, 4, :, :] = t25[i + 12 + 492, :, :]
    ssh_pre_test[i, 5, :, :] = t45[i + 12 + 492, :, :]

    # 与上述一致
    for j in range(12):
        para_input_test[i, m, :, :] = windu[i+492+j, :, :]
        para_input_test[i, m+1, :, :] = windv[i+492+j, :, :]
        para_input_test[i, m+2, :, :] = sealevel[i+492+j, :, :]
        para_input_test[i, m + 3, :, :] = t05[i + 492 + j, :, :]
        para_input_test[i, m + 4, :, :] = t25[i + 492 + j, :, :]
        para_input_test[i, m + 5, :, :] = t45[i + 492 + j, :, :]
        m = m+6
    m = 0

# 最后一组需要额外的赋值
ssh_pre_test[-1, 0, :, :] = windu[-1, :, :]
ssh_pre_test[-1, 1, :, :] = windv[-1, :, :]
ssh_pre_test[-1, 2, :, :] = sealevel[-1, :, :]
ssh_pre_test[-1, 3, :, :] = t05[-1, :, :]
ssh_pre_test[-1, 4, :, :] = t25[-1, :, :]
ssh_pre_test[-1, 5, :, :] = t45[-1, :, :]

for j in range(12):
    para_input_test[-1, m, :, :] = windu[108+492+j, :, :]
    para_input_test[-1, m+1, :, :] = windv[108+492+j, :, :]
    para_input_test[-1, m+2, :, :] = sealevel[108+492+j, :, :]
    para_input_test[-1, m + 3, :, :] = t05[108 + 492 + j, :, :]
    para_input_test[-1, m + 4, :, :] = t25[108 + 492 + j, :, :]
    para_input_test[-1, m + 5, :, :] = t45[108 + 492 + j, :, :]
    m = m + 6

print(para_input_train.shape,
      ssh_pre_train.shape,
      para_input_test.shape,
      ssh_pre_test.shape,
      )
#
np.save('datafile_a/train_x.npy', para_input_train)
np.save('datafile_a/train_y.npy', ssh_pre_train)
np.save('datafile_a/test_x.npy', para_input_test)
np.save('datafile_a/test_y.npy', ssh_pre_test)

