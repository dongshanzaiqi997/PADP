import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib
# matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


folders = ['CCACP12I0_0.9', 'MF_0.9', 'FWP_20']
methods = ['CCAC', 'MF-PD', 'FWP-20']


fig, ax = plt.subplots(1,1)
#axins = ax.inset_axes((0.3, 0.2, 0.32, 0.24))
for folder, method in zip(folders, methods):
    file = [i for i in glob.glob('data/' + folder + '/run/*.{}'.format('csv'))]  # 加载所有后缀为csv的文件。
    file = pd.concat([pd.read_csv(i) for i in file])
    file.rename(columns={'Unnamed: 0': 't', '0': 've', '2': 'Gap'}, inplace=True)
    file['t'] *= 0.1
    sns.lineplot(x='t', y='Gap', data=file, label=method, linewidth=2.0)
    # sns.lineplot(x='t', y='Gap', data=file, ax=axins, linewidth=2.0)

plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Gap between two vehicles (m)', fontsize=12)
plt.tick_params(labelsize=12)
plt.axhline(y=2, ls="--", label='Required minimum gap', color='red')
plt.legend(loc='upper right', fontsize=11)
plt.ylim(1.8, 7.5)
plt.xlim(0, 8)


plt.savefig('gap.pdf')
#
# df = pd.DataFrame()
# for folder, method in zip(folders, methods):
#     file = [i for i in glob.glob('data/' + folder + '/run/*.{}'.format('csv'))]
#     file = pd.concat([pd.read_csv(i).mean() for i in file])
#     mean_gap = file['2'].values
#     mean_ve = file['0'].values
#     # min_gap =
#     method_index = [method for _ in range(5)]
#     df = df.append(pd.DataFrame({'method' : pd.Series(method_index),
#           'mean_gap' : pd.Series(mean_gap),
#          'mean_ve': pd.Series(mean_ve)}))
#
# plt.figure()
# sns.boxplot(x='method', y='mean_gap', data=df)
# plt.xlabel('Method', fontsize=12)
# plt.ylabel('Average gap between two vehicles', fontsize=12)
# plt.tick_params(labelsize=12)
# plt.axhline(y=2, ls="--", label='Required minimum gap', color='red')
#
# plt.figure()
# sns.boxplot(x='method', y='mean_ve', data=df)
# plt.xlabel('Method', fontsize=12)
# plt.ylabel('Average ego vhicle speed', fontsize=12)
# plt.tick_params(labelsize=12)
plt.show()
#
#

