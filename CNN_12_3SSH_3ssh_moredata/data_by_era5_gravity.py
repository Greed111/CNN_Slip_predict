import numpy as np
import xarray as xr
from pathlib import Path
from sklearn import preprocessing

# 批量读取海平面高度月异常nc文件,共计28年月数据，27*12+5=329
year = range(1993, 2021)
sealevelpath0 = 'E:/Datamonth/Month_sealevel_abnormaity/'
sealevel_all = []
for i in year:        # 循环文件夹名称
    sealevelpath = sealevelpath0+str(i)
    filepath = Path(sealevelpath)     # 将字符串转化为路径名字
    filelist = list(filepath.glob('*.nc'))      # 将文件夹中的所有nc文件收入列表
    for file in filelist:       # 循环读取数据，加入到xarray矩阵中
        with xr.open_dataset(file) as f:
            sealevel_all.append(f['sla'].sel(longitude=slice(125.125, 280.126, 4), latitude=slice(-15.125, 14.875, 4)).values)
            # 300纬度:-15.125,14.875   505经度：130.125，281.125  longitude和latitude参数分别是 .sel()方法的两个参数 ，还可以加上time参数
sealevel = np.squeeze(np.array(sealevel_all))
print(sealevel.shape)

# xarray.open_dataset.sel这个方法，必须沿着坐标轴正方向走，如果反了只能flip翻转
data_wind = xr.open_dataset('E:\Datamonth\ERA5_Wind_Monthly_19502023.nc')
time = data_wind['time'].values
# print(time[516])
windu = data_wind['u10'].sel(longitude=slice(125, 280.25, 4), latitude=slice(15, -15.25, 4), expver=1).values
windu = np.flip(windu[516:845, :, :], axis=1)
windv = data_wind['v10'].sel(longitude=slice(125, 280.25, 4), latitude=slice(15, -15.25, 4), expver=1).values
windv = np.flip(windv[516:845, :, :], axis=1)
print(windu.shape)      # 329, 31, 156


mask1 = np.isnan(windu)
mask2 = np.isnan(sealevel)
num, fea1, fea2 = windu.shape
# reshaped_tensor = windu.reshape((num, fea1 * fea2))
# transform_adj = preprocessing.MinMaxScaler().fit_transform(reshaped_tensor)
# windu = transform_adj.reshape((num, fea1, fea2))
# reshaped_tensor = windv.reshape((num, fea1 * fea2))
# transform_adj = preprocessing.MinMaxScaler().fit_transform(reshaped_tensor)
# windv = transform_adj.reshape((num, fea1, fea2))
# reshaped_tensor = sealevel.reshape((num, fea1 * fea2))
# transform_adj = preprocessing.MinMaxScaler().fit_transform(reshaped_tensor)
# sealevel = transform_adj.reshape((num, fea1, fea2))

windu[mask1] = 0
windv[mask1] = 0
sealevel[mask2] = 0

num_ts = 26*12-1          # 26年进行检验1993——2019包括首尾,第27年数据不全
para_input_test = np.zeros((num_ts+1, 36, 31, 156), dtype=np.float32)
ssh_pre_test = np.zeros((num_ts+1, 3, 31, 156), dtype=np.float32)
m = 0

# 保存所有test部分的x和y数据
for i in range(num_ts):
    if i < 3 or i >= 315:
        print(i, i+12)
    # 测试集，与上述一致
    ssh_pre_test[i, 0, :, :] = windu[i+12, :, :]
    ssh_pre_test[i, 1, :, :] = windv[i+12, :, :]
    ssh_pre_test[i, 2, :, :] = sealevel[i+12, :, :]
    # 与上述一致
    for j in range(12):
        para_input_test[i, m, :, :] = windu[i+j, :, :]
        para_input_test[i, m+1, :, :] = windv[i+j, :, :]
        para_input_test[i, m+2, :, :] = sealevel[i+j, :, :]
        m = m+3
    m = 0
# 最后一组需要额外的赋值
ssh_pre_test[-1, 0, :, :] = windu[-1, :, :]
ssh_pre_test[-1, 1, :, :] = windv[-1, :, :]
ssh_pre_test[-1, 2, :, :] = sealevel[-1, :, :]
for j in range(12):
    para_input_test[-1, m, :, :] = windu[312+j, :, :]
    para_input_test[-1, m+1, :, :] = windv[312+j, :, :]
    para_input_test[-1, m+2, :, :] = sealevel[312+j, :, :]
    m = m+3

print(para_input_test.shape,
      ssh_pre_test.shape,
      para_input_test[1,1,1,:10],
      )

np.save('otherdata/test_x_2.npy', para_input_test)
np.save('otherdata/test_y_2.npy', ssh_pre_test)