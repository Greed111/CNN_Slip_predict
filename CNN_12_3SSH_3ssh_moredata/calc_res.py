import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.stats import ttest_ind

pre_y = np.load(r'datasave\x_12_4.npy')
truth_y = np.load(r'datasave\y_12_4.npy')
ls_mae = []
ls_rmse = []
for i in range(12):
    tot = 0
    tot1 = 0
    for j in range(31):
        tot = tot + mean_absolute_error(truth_y[i, 2, j], pre_y[i, 2, j])
        tot1 = tot1 + mean_squared_error(truth_y[i, 2, j], pre_y[i, 2, j])
    tot = tot/31
    tot1 = tot1/31
    ls_mae.append(tot)
    ls_rmse.append(tot1)

ls_pre = np.zeros((12, 31, 156), dtype=float)
ls_tru = np.zeros((12, 31, 156), dtype=float)
for i in range(12):
    ls_pre[i, :, :] = np.squeeze(pre_y[i, 2, :, :])
    ls_tru[i, :, :] = np.squeeze(truth_y[i, 2, :, :])
res = ttest_ind(ls_tru, ls_pre).pvalue
print(res[:, 0])

np.save('datasave/analysis_rmse_3.npy', ls_rmse)
np.save('datasave/analysis_mae_3.npy', ls_mae)
np.save('datasave/analysis_Ttest_3.npy', res)

