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


folders = ['CCACP12I0_0.9', 'CCACP80_0.9', 'CCACI18_0.9', 'CCACP15I06trick_0.9']
methods = ['penalty $(K_P=12)$', 'penalty $(K_P=80)$', 'Lagrangian $(K_I=18)$', 'SPIL $(K_P=15, K_I=0.6)$']

fig1, ax1 = plt.subplots(1, 1, figsize=(6,4.2))
# axins1 = ax1.inset_axes((0.61, 0.6, 0.32, 0.24))
for j, (folder, method) in enumerate(zip(folders, methods)):
    file = [i for i in glob.glob('data/'+folder+'/safe_prob/*.{}'.format('csv'))]  # 加载所有后缀为csv的文件。
    file = pd.concat([pd.read_csv(i) for i in file])
    file.rename(columns={'Step': 'Iteration', 'Value': 'Safe probability'}, inplace=True)
    sns.lineplot(x='Iteration', y='Safe probability', ax=ax1, data=file, label=method, linewidth=2.0)
    # sns.lineplot(x='Iteration', y='Safe probability', data=file, ax=axins1, linewidth=2.0)

plt.tick_params(labelsize=13)
plt.axhline(y=0.9,ls="--", label='constraint threshold 90.0%', color='red')
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
plt.savefig('safety0.9.pdf', bbox_inches='tight')


fig, ax = plt.subplots(1, 1, figsize=(6,4.2))
#axins = ax.inset_axes((0.6, 0.6, 0.32, 0.24))
for j, (folder, method) in enumerate(zip(folders, methods)):
    file = [i for i in glob.glob('data/' + folder + '/r_N/*.{}'.format('csv'))]  # 加载所有后缀为csv的文件。
    file = pd.concat([pd.read_csv(i) for i in file])
    file.rename(columns={'Step': 'Iteration', 'Value': 'Cumulative reward'}, inplace=True)
    sns.lineplot(x='Iteration', y='Cumulative reward', data=file, label=method, linewidth=2.0)
    #sns.lineplot(x='Iteration', y='Cumulative reward', data=file, ax=axins, linewidth=2.0)
plt.tick_params(labelsize=13)
plt.legend(loc='lower right', fontsize=12)
plt.ylim(-50, 25)
plt.xlim(0, 700)
plt.xlabel('Iteration', fontsize=13)
plt.ylabel('Cumulative reward', fontsize=13)
# axins.set_xlim(5000, 5500)
# axins.set_ylim(27, 46)
# mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)
fig.tight_layout()
plt.savefig('return0.9.pdf', bbox_inches='tight')


folders = ['CCACP12I0_0.999', 'CCACP80_0.999', 'CCACI18_0.999', 'CCACP15I06trick_0.999']
methods = ['penalty $(K_P=12)$', 'penalty $(K_P=80)$', 'Lagrangian $(K_I=18)$', 'SPIL $(K_P=15, K_I=0.6)$']
fig, ax = plt.subplots(1, 1, figsize=(6,4.2))
axins = ax.inset_axes((0.6, 0.5, 0.32, 0.24))
for j, (folder, method) in enumerate(zip(folders, methods)):
    file = [i for i in glob.glob('data/'+folder+'/safe_prob/*.{}'.format('csv'))]  # 加载所有后缀为csv的文件。
    file = pd.concat([pd.read_csv(i) for i in file])
    file.rename(columns={'Step': 'Iteration', 'Value': 'Safe probability'}, inplace=True)
    sns.lineplot(x='Iteration', y='Safe probability', data=file, label=method, linewidth=2.0)
    sns.lineplot(x='Iteration', y='Safe probability', data=file, ax=axins, linewidth=2.0)

plt.axhline(y=0.999,ls="--", label='constraint threshold 99.9%', color='red')
plt.ylim(0.5, 1.03)
plt.xlim(0, 700)
plt.legend(loc='lower right', fontsize=12)
axins.set_xlabel('', fontsize=13)
axins.set_ylabel('', fontsize=13)
ax.set_xlabel('Iteration', fontsize=13)
ax.set_ylabel('Safe probability', fontsize=13)
axins.set_xlim(673, 700)
axins.set_ylim(0.99, 1.003)
axins.axhline(y=0.999,ls="--", label='Required Safe Probability 99.9%', color='red')
mark_inset(ax, axins, loc1=4, loc2=2, fc="none", ec='k', lw=1)
plt.tick_params(labelsize=13)
fig.tight_layout()
plt.savefig('safety0.999.pdf', bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize=(6,4.2))
for j, (folder, method) in enumerate(zip(folders, methods)):
    file = [i for i in glob.glob('data/' + folder + '/r_N/*.{}'.format('csv'))]  # 加载所有后缀为csv的文件。
    file = pd.concat([pd.read_csv(i) for i in file])
    file.rename(columns={'Step': 'Iteration', 'Value': 'Cumulative reward'}, inplace=True)
    ax = sns.lineplot(x='Iteration', y='Cumulative reward', data=file, label=method, linewidth=2.0)

plt.tick_params(labelsize=13)
plt.legend(loc='lower right', fontsize=12)
plt.ylim(-50, 25)
plt.xlim(0, 700)
plt.xlabel('Iteration', fontsize=13)
plt.ylabel('Cumulative reward', fontsize=13)
fig.tight_layout()
plt.savefig('return0.999.pdf', bbox_inches='tight')

plt.show()