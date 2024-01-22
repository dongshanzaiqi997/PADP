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


fig1, ax1 = plt.subplots(1, 1, figsize=(3, 2))
x = np.linspace(0, 1,1000)
ks = np.where(x>0.05, 0.3, 1)
ks = np.where(x>0.3, 0, ks)
sns.lineplot(x=x, y=ks, label='$K_S$', linewidth=2.0)
plt.xlabel('$\Delta^{k}$')
plt.ylabel('$K_S$')
plt.tick_params(labelsize=8)

# plt.ylim(0.5, 1.03)
# plt.xlim(0, 700)
plt.legend(loc='upper right', fontsize=7)


fig1.tight_layout()
plt.savefig('ks.pdf', bbox_inches='tight')


plt.show()