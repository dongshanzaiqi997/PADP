import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd


folders = ['MF_0.999']
methods = ['MF-PD']
plt.figure(1)
for folder, method in zip(folders, methods):
    file = [i for i in glob.glob('data/' + folder + '/safe_prob/*.{}'.format('csv'))]  # 加载所有后缀为csv的文件。
    for i in range(len(file)):
        f = pd.read_csv(file[i])
        f.rename(columns={'Step': 'Iteration', 'Value': 'Safe probability'}, inplace=True)
        sns.lineplot(x='Iteration', y='Safe probability', label=file[i], data=f, linewidth=2.0)
plt.legend()
plt.figure(5)
for folder, method in zip(folders, methods):
    file = [i for i in glob.glob('data/' + folder + '/r_sum/*.{}'.format('csv'))]  # 加载所有后缀为csv的文件。
    for i in range(len(file)):
        f = pd.read_csv(file[i])
        f.rename(columns={'Step': 'Iteration', 'Value': 'return'}, inplace=True)
        sns.lineplot(x='Iteration', y='return', label=file[i], data=f, linewidth=2.0)
plt.legend()

# plt.figure(1)
# plt.xlabel('Time (s)', fontsize=12)
# plt.ylabel('Gap between two vehicles (m)', fontsize=12)
# plt.tick_params(labelsize=12)
# plt.axhline(y=2, ls="--", label='Required minimum gap', color='red')
# plt.legend(loc='upper right', fontsize=11)
# plt.ylim(1.8, 10)
# plt.xlim(0, 8)
#plt.savefig('gap.pdf')
plt.show()
#
#

