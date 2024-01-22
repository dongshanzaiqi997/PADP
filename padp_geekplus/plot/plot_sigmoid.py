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


def posi(x):
    return np.where(x>=0,x,0)

fig1, ax1 = plt.subplots(1, 1, figsize=(3.5,3))
x = np.linspace(-1, 1,1000)
indi = np.where(x>0, 1, 0)
m1 = 1
m2 = m1/(1+m1) * 0.9
#m2 = 1
tau = 0.07
sig = np.clip((1+tau*m1)/(1+m2*tau*np.exp(np.clip(-x/tau,-10,10))), 0, 1.2)

tau = 1
CVar = posi(x+1)
tau = 1
DC = posi(x+1)-posi(x)
#
sns.lineplot(x=x, y=indi, label='${\\rm \mathbb{I}}(x)$', linewidth=2.0)
# sns.lineplot(x=x, y=CVar, label='CVar', linewidth=2.0)
# sns.lineplot(x=x, y=DC, label='DC', linewidth=2.0)
sns.lineplot(x=x, y=sig, label='sigmoidal', linewidth=2.0)
plt.tick_params(labelsize=13)

# plt.ylim(0.5, 1.03)
# plt.xlim(0, 700)
plt.legend(loc='upper left', fontsize=12)


fig1.tight_layout()
# plt.savefig('indicator.pdf', bbox_inches='tight')


plt.show()