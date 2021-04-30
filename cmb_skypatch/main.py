# %%
from lib import Lib
import json
import numpy as np
import plot
import matplotlib.pyplot as plt
import healpy as hp

with open('/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/cmb_skypatch/config.json', "r") as f:
    cf = json.load(f)

spdata = dict()
CB_color_cycle = ["#88CCEE", "#CC6677", "#DDCC77", "#117733", "#332288", "#AA4499", 
                             "#44AA99", "#999933", "#882255", "#661100", "#6699CC", "#888888"]


# %%
npatch = [1,2,4,8]
smoothing_par = cf['pa']['smoothing_par']
for n in npatch:
    if str(n) in spdata.keys():
        pass
    else:
        spdata.update({str(n): dict()})
    for smooth in smoothing_par:
        if str(smooth) in spdata[str(n)].keys():
            pass
        else:
            spdata[str(n)].update({str(smooth): 
                Lib(npatch = n, smoothing_par = smooth, C_lF=None, C_lN=None)})


# %%
import healpy as hp
import numpy as np
varmap = spdata['1']['0'].noisevar_map['100']
mean, binedges = np.histogram(np.nan_to_num(varmap), bins=np.logspace(np.log10(varmap.min()),np.log10(varmap.max()),10000))

# %%

hp.mollview(spdata['1']['0'].noisevar_map['100'])

# %%
print(np.where(spdata['1']['0'].noisevar_map['100']==0.0, np.mean(spdata['1']['0'].noisevar_map['100']), spdata['1']['0'].noisevar_map['100']))
print(spdata['1']['0'].noisevar_map['100'].min())


# %%
plot.spectrum_variance(spdata['1']['0'], spdata['1']['0'])


# %%
plot.compare_variance_min(spdata)


# %%
plot.compare_improvement(spdata, ["4_0-2_0","4_1-2_1"])


# %%
plot.compare_errorbars(spdata, ['0'])


# %%
import json

import sys
import random
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
data = spdata['1']['0']
data_fullsky = spdata['1']['0']
C_lS = data.C_lS
cov_ltot_min = data.cov_ltot_min
variance_min_ma = data.variance_min_ma
variance = data.variance

fig, ax = plt.subplots(figsize=(8,6))
ll = np.arange(0,variance_min_ma.shape[0],1)
plt.plot(variance_min_ma, label='Variance, combined patches', lw=3, ls="-", color="black")
plt.plot(data_fullsky.variance_min_ma, label="Variance, full sky", lw=3, ls="--", color="black")

plt.plot(C_lS*ll*(ll+1)/(np.pi*2), label = 'Planck EE', lw=3, ls="-", color="red", alpha=0.5)
for n in range(cov_ltot_min.shape[0]):
    plt.plot(cov_ltot_min[n,:]*ll*(ll+1)/(np.pi*2), lw=1, ls="-", color="red", alpha=0.5)
plt.plot(0,0, label='Minimal powerspectrum from patches', color='red', alpha=0.5, lw=1)

plt.plot(2*C_lS*C_lS/((2*ll+1)), label="Variance, Planck EE, full sky", lw=3, ls="-", color="blue")
# plt.plot(loc_opt_NN, label = "Variance, Planck EE, combined patches", lw=3, ls="--", color="green")

leg1 = plt.legend(loc='upper left')
pa = [None for n in range(variance.shape[1])]
for n in range(len(pa)):
    p = plt.plot(variance[:,n,n], lw=2, ls="-", alpha=0.5)
    col = p[0].get_color()
    pa[n] = Patch(facecolor=col, edgecolor='grey', alpha=0.5)
leg2 = plt.legend(handles=pa[:min(20,len(pa))],
        labels=["" for n in range(min(20,len(pa))-1)] + ['Variance, {} skypatches'.format(str(cov_ltot_min.shape[0]))],
        ncol=variance.shape[1], handletextpad=0.5, handlelength=0.5, columnspacing=-0.5,
        loc='lower left')
ax.add_artist(leg1)

plt.xscale('log')
plt.yscale('log')
plt.xlabel("Multipole")
plt.ylabel(r"Variance $\sigma^2$ [$\mu K^4$]")
plt.xlim((2e1,4e3))
plt.ylim((1e-3,1e6))
# plt.tight_layout()
ax2 = ax.twinx()
ax2.tick_params(axis='y', labelcolor="red")
ax2.set_ylabel(r'Powerspectrum $C_l$ [$\mu K^2$]', color= 'red')
plt.ylim((1e-3,1e6))
plt.yscale('log')
plt.title("Combining {} skypatches".format(str(cov_ltot_min.shape[0])))
plt.savefig("/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/cmb_skypatch/vis/analyticpatching-NoiseSignal{}patches.jpg".format(str(cov_ltot_min.shape[0])))

# %%

print(spdata['1']['0'].C_lN.shape)
for n in range(spdata['1']['0'].C_lN.shape[0]):
    plt.plot(spdata['1']['0'].C_lN[n,0,:]*ll*(ll+1)/(np.pi*2))
plt.yscale('log')
plt.xscale('log')
plt.xlim((2e1,4e3))
plt.ylim((1e-3,1e6))
plt.grid(which='major', axis='both')
plt.grid(which='minor', axis='x')
plt.xlabel('Multipole')
plt.ylabel('Powerspectrum D_l')
# %%
