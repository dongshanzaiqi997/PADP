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
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
#
# folders = ['unsafepeter0.9', 'unsafePID0.9']
# methods = ['PI w/o separated integral', 'PI with separated integral (ours)']
#
# fig1, ax1 = plt.subplots(1, 1)
# # axins1 = ax1.inset_axes((0.61, 0.6, 0.32, 0.24))
# for folder, method in zip(folders, methods):
#     file = [i for i in glob.glob('data/'+folder+'/safe_prob/*.{}'.format('csv'))]  # 加载所有后缀为csv的文件。
#     file = pd.concat([pd.read_csv(i) for i in file])
#     file.rename(columns={'Step': 'Iteration', 'Value': 'Safe probability'}, inplace=True)
#     sns.lineplot(x='Iteration', y='Safe probability', ax=ax1, data=file, label=method, linewidth=2.0)
#     # sns.lineplot(x='Iteration', y='Safe probability', data=file, ax=axins1, linewidth=2.0)
#
# plt.tick_params(labelsize=12)
# plt.axhline(y=0.9,ls="--", label='Required threshold 90.0%', color='red')
# plt.ylim(0.5, 1.03)
# plt.xlim(0, 700)
# plt.legend(loc='lower right', fontsize=11)
# ax1.set_xlabel('Iteration', fontsize=12)
# ax1.set_ylabel('Safe probability', fontsize=12)
# # axins1.set_xlim(5000, 5500)
# # axins1.set_ylim(0.985, 1.003)
# # axins1.axhline(y=0.999,ls="--", label='Required Safe Probability 99.9%', color='red')
# # mark_inset(ax1, axins1, loc1=4, loc2=2, fc="none", ec='k', lw=1)
#
# plt.savefig('unsafe_safety0.9.pdf', bbox_inches='tight')
#
#
# fig, ax = plt.subplots(1,1)
# #axins = ax.inset_axes((0.6, 0.6, 0.32, 0.24))
# for folder, method in zip(folders, methods):
#     file = [i for i in glob.glob('data/' + folder + '/r_N/*.{}'.format('csv'))]  # 加载所有后缀为csv的文件。
#     file = pd.concat([pd.read_csv(i) for i in file])
#     file.rename(columns={'Step': 'Iteration', 'Value': 'Cumulative reward'}, inplace=True)
#     temp = sns.lineplot(x='Iteration', y='Cumulative reward', data=file, label=method, linewidth=2.0)
#     if folder =='CCACP15I06trick_0.9':
#         temp.set_zorder(1)
#     #sns.lineplot(x='Iteration', y='Cumulative reward', data=file, ax=axins, linewidth=2.0)
# plt.tick_params(labelsize=12)
# plt.legend(loc='lower right', fontsize=11)
# plt.ylim(-25, 25)
# plt.xlim(0, 700)
# plt.xlabel('Iteration', fontsize=12)
# plt.ylabel('Cumulative reward', fontsize=12)
# # axins.set_xlim(5000, 5500)
# # axins.set_ylim(27, 46)
# # mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)
# plt.savefig('unsafe_return0.9.pdf', bbox_inches='tight')
#
# fig, ax = plt.subplots(1, 1)
# for folder, method in zip(folders, methods):
#     file = [i for i in glob.glob('data/' + folder + '/delta_i/*.{}'.format('csv'))]  # 加载所有后缀为csv的文件。
#     file = pd.concat([pd.read_csv(i) for i in file])
#     file.rename(columns={'Step': 'Iteration', 'Value': 'Integral value'}, inplace=True)
#     ax = sns.lineplot(x='Iteration', y='Integral value', data=file, label=method, linewidth=2.0)
#
# plt.tick_params(labelsize=12)
# plt.legend(loc='lower right', fontsize=11)
# plt.ylim(-25, 25)
# plt.xlim(0, 700)
# plt.xlabel('Iteration', fontsize=12)
# plt.ylabel('Integral value', fontsize=12)
# plt.savefig('unsafe_integral0.9.pdf', pad_inches=0, bbox_inches='tight')


color = sns.color_palette()
color.pop(0); color.pop(0)
sns.set_palette(color)

folders = ['unsafepeter0.999', 'unsafePID0.999']
methods = ['SPIL w/o integral separation', 'SPIL w/ integral separation']

fig, ax = plt.subplots(1, 1, figsize=(6,4.2))
axins = ax.inset_axes((0.6, 0.5, 0.32, 0.24))
for folder, method in zip(folders, methods):
    file = [i for i in glob.glob('data/'+folder+'/safe_prob/*.{}'.format('csv'))]  # 加载所有后缀为csv的文件。
    file = pd.concat([pd.read_csv(i) for i in file])
    file.rename(columns={'Step': 'Iteration', 'Value': 'Safe probability'}, inplace=True)
    sns.lineplot(x='Iteration', y='Safe probability', data=file, label=method, linewidth=2.0)
    sns.lineplot(x='Iteration', y='Safe probability', data=file, ax=axins, linewidth=2.0)


plt.axhline(y=0.999,ls="--", label='constraint threshold 99.9%', color='red')
plt.ylim(0.5, 1.03)
plt.xlim(0, 800)
plt.legend(loc='lower right', fontsize=16)
ax.set_xlabel('Iteration', fontsize=17)
ax.set_ylabel('Safe probability', fontsize=17)
axins.set_xlabel('', fontsize=17)
axins.set_ylabel('', fontsize=17)
plt.tick_params(labelsize=17)
axins.set_xlim(770, 800)
axins.set_ylim(0.99, 1.003)
axins.axhline(y=0.999,ls="--", label='Required Safe Probability 99.9%', color='red')
mark_inset(ax, axins, loc1=4, loc2=2, fc="none", ec='k', lw=1)
fig.tight_layout()
axins.tick_params(labelsize=15)
plt.savefig('unsafe_safety0.999.pdf', bbox_inches='tight')

# fig, ax = plt.subplots(1, 1, figsize=(6,4.2))
# for folder, method in zip(folders, methods):
#     file = [i for i in glob.glob('data/' + folder + '/r_N/*.{}'.format('csv'))]  # 加载所有后缀为csv的文件。
#     file = pd.concat([pd.read_csv(i) for i in file])
#     file.rename(columns={'Step': 'Iteration', 'Value': 'Cumulative reward'}, inplace=True)
#     ax = sns.lineplot(x='Iteration', y='Cumulative reward', data=file, label=method, linewidth=2.0)
#
# plt.tick_params(labelsize=17)
# plt.legend(loc='lower right', fontsize=16)
# plt.ylim(-80, 25)
# plt.xlim(0, 800)
# plt.xlabel('Iteration', fontsize=17)
# plt.ylabel('Cumulative reward', fontsize=17)
# fig.tight_layout()
# plt.savefig('unsafe_return0.999.pdf', bbox_inches='tight')

# fig, ax = plt.subplots(1, 1, figsize=(6,4.2))
# for folder, method in zip(folders, methods):
#     file = [i for i in glob.glob('data/' + folder + '/delta_i/*.{}'.format('csv'))]  # 加载所有后缀为csv的文件。
#     file = pd.concat([pd.read_csv(i) for i in file])
#     file.rename(columns={'Step': 'Iteration', 'Value': 'Integral value'}, inplace=True)
#     ax = sns.lineplot(x='Iteration', y='Integral value', data=file, label=method, linewidth=2.0)
#
# plt.tick_params(labelsize=17)
# plt.legend(loc='lower right', fontsize=16)
# plt.ylim(-1,9)
# plt.xlim(0, 800)
# plt.xlabel('Iteration', fontsize=17)
# plt.ylabel('Integral value', fontsize=17)
# # plt.axhline(y=2.34,ls="--", label='Convergence', color='red')
# fig.tight_layout()
# plt.savefig('unsafe_integral0.999.pdf', bbox_inches='tight')

plt.show()