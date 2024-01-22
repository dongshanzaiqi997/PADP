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


folders = ['CCACP12I0_0.9', 'CCACP80_0.9']
methods = ['penalty weight 12', 'penalty weight 80']

fig1, ax1 = plt.subplots(1, 1, figsize=(3.5,3))
# axins1 = ax1.inset_axes((0.61, 0.6, 0.32, 0.24))
for j, (folder, method) in enumerate(zip(folders, methods)):
    file = [i for i in glob.glob('data/'+folder+'/safe_prob/*.{}'.format('csv'))]  # 加载所有后缀为csv的文件。
    file = pd.concat([pd.read_csv(i) for i in file])
    file.rename(columns={'Step': 'Iteration', 'Value': 'Safe probability'}, inplace=True)
    sns.lineplot(x='Iteration', y='Safe probability', ax=ax1, data=file, label=method, linewidth=2.0)
    # sns.lineplot(x='Iteration', y='Safe probability', data=file, ax=axins1, linewidth=2.0)

plt.tick_params(labelsize=13)
plt.axhline(y=0.9, ls="--", color='red', label='chance constraint')
plt.ylim(0.5, 1.03)
plt.xlim(0, 700)
plt.legend(loc='lower right', fontsize=12)
ax1.set_xlabel('Iteration', fontsize=13)
ax1.set_ylabel('Safe probability', fontsize=13)
# axins1.set_xlim(5000, 5500)
# axins1.set_ylim(0.985, 1.003)
# axins1.axhline(y=0.999,ls="--", label='Required Safe Probability 99.9%', color='red')
# mark_inset(ax1, axins1, loc1=4, loc2=2, fc="none", ec='k', lw=1)
fig1.tight_layout()
plt.savefig('demo_penalty09.pdf', bbox_inches='tight')


# folders = ['unsafepeterI0.6S0.999']
# methods = ['Lagrangian']
# fig, ax = plt.subplots(1, 1, figsize=(3.5,3))
# plt.xlim(0, 700)
# plt.xlabel('Iteration', fontsize=13)
# # axins = ax.inset_axes((0.6, 0.5, 0.32, 0.24))
# # for j, (folder, method) in enumerate(zip(folders, methods)):
# #     file = [i for i in glob.glob('data/'+folder+'/safe_prob/*.{}'.format('csv'))]  # 加载所有后缀为csv的文件。
# #     file = pd.concat([pd.read_csv(i) for i in file])
# #     file.rename(columns={'Step': 'Iteration', 'Value': 'Safe probability'}, inplace=True)
# #     sns.lineplot(x='Iteration', y='Safe probability', data=file, linewidth=2.0, color=sns.xkcd_rgb["windows blue"])
# #
# # ax.set_ylim(0.5, 1.03)
# # ax.set_ylabel('Safe probability', fontsize=13, color=sns.xkcd_rgb["windows blue"])
# # plt.axhline(y=0.999,ls="--", label='Constraint 99.9%', color=sns.xkcd_rgb["windows blue"])
# # plt.tick_params(axis='y',colors=sns.xkcd_rgb["windows blue"])
# #
# # ax2 = ax.twinx()
# # for j, (folder, method) in enumerate(zip(folders, methods)):
# #     file = [i for i in glob.glob('data/' + folder + '/delta_i/*.{}'.format('csv'))]  # 加载所有后缀为csv的文件。
# #     file = pd.concat([pd.read_csv(i) for i in file])
# #     file.rename(columns={'Step': 'Iteration', 'Value': 'Integral value'}, inplace=True)
# #     sns.lineplot(x='Iteration', y='Integral value', data=file, linewidth=2.0, color=sns.xkcd_rgb["pale red"])
# # ax2.set_ylim(-1,9)
# # ax2.set_ylabel('Lagrangian multiplier', fontsize=13, color=sns.xkcd_rgb["pale red"])
# #
# # plt.tick_params(labelsize=13)
# # plt.axhline(y=2.36,ls="--", color=sns.xkcd_rgb["pale red"])
# # plt.tick_params(axis='y',colors=sns.xkcd_rgb["pale red"])
# # #plt.legend(loc='lower right', fontsize=11)
# # fig.tight_layout()
# # plt.savefig('demo_Lag0999.pdf', bbox_inches='tight')
#
# #
# for j, (folder, method) in enumerate(zip(folders, methods)):
#     file = [i for i in glob.glob('data/' + folder + '/delta_i/*.{}'.format('csv'))]  # 加载所有后缀为csv的文件。
#     file = pd.concat([pd.read_csv(i) for i in file])
#     file.rename(columns={'Step': 'Iteration', 'Value': 'Integral value'}, inplace=True)
#     sns.lineplot(x='Iteration', y='Integral value', data=file, linewidth=2.0, label='Lagrangian')
# #ax.set_ylim(-1,9)
# ax.set_ylabel('Lagrange multiplier', fontsize=13)
#
# plt.tick_params(labelsize=13)
# #plt.axhline(y=2.36,ls="--", color=sns.color_palette()[0], label='convergence')
# plt.tick_params(axis='y')
# plt.legend(loc='lower right', fontsize=12)
# fig.tight_layout()
# plt.savefig('demo_Lag09.pdf', bbox_inches='tight')
plt.show()