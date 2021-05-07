# %%
import json
import numpy as np
import plot
import matplotlib.pyplot as plt
import healpy as hp
import component_separation.io as io
from component_separation.cs_util import Planckf, Plancks
import component_separation.powspec as pw
import itertools
PLANCKMAPFREQ = [p.value for p in list(Planckf)]

cs_abs_path = "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/"
sp_abs_path = "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/cmb_skypatch/"
with open(cs_abs_path + 'component_separation/draw/draw.json', "r") as f:
    dcf = json.load(f)
with open(sp_abs_path + 'config.json', "r") as f:
    cf = json.load(f)
fname = io.make_filenamestring(dcf)
speccs =  [spec for spec in ['EE']]
FREQS = [FR for FR in PLANCKMAPFREQ if FR not in cf['pa']['freqfilter']]
freqcomb =  ["{}-{}".format(p[0],p[1])
        for p in itertools.product(FREQS, FREQS)
        if (int(p[1])>=int(p[0]))]
dc = dcf["plot"]["spectrum"]
def _inpathname(freqc,spec):
    return  cs_abs_path+dc["indir_root"]+dc["indir_rel"]+spec+freqc+"-"+dc["in_desc"]+fname

CB_color_cycle = ["#88CCEE", "#CC6677", "#DDCC77", "#117733", "#332288", "#AA4499", 
                             "#44AA99", "#999933", "#882255", "#661100", "#6699CC", "#888888"]
spdataNN = dict()
spdata = dict()


# %%
empiric_spectrum = {freqc: {
        spec: np.array(io.load_cl(_inpathname(freqc,spec)))
        for spec in speccs}  
        for freqc in freqcomb}
cov_emp = pw.build_covmatrices(empiric_spectrum, 3000, cf['pa']['freqfilter'], cf['pa']['specfilter'])
print(cov_emp["EE"].shape)
from lib_emp import Lib_emp
empdata = Lib_emp(C_lF=None, C_lN=None, dov_ltot=np.array([cov_emp["EE"]]))
# %%
from lib import Lib


# %%
# % autoreload
#% aimport -lib
npatch = [1,8]
smoothing_par = cf['pa']['smoothing_par']

for n in npatch:
    if str(n) in spdataNN.keys():
        pass
    else:
        spdataNN.update({str(n): dict()})
    for smooth in smoothing_par:
        if str(smooth) in spdataNN[str(n)].keys():
            pass
        else:
            spdataNN[str(n)].update({str(smooth): 
                Lib(npatch = n, smoothing_par = smooth, C_lF=None, C_lN=None, C_lN_factor=0.000001)})

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
plot.spectrum_variance(empdata, spdata['8']['0'], rd_ylim=(-0.2,0.7))


# %%
plot.compare_variance_min(spdata)


# %%
plot.compare_improvement(spdata, ["8_0-1_0","8_0-1_0"])


# %%
plot.compare_errorbars(spdata, ['0'])


# %%
spdata['1']['0'].C_lN


# %%
hp.mollview(spdata['1']['0'].noisevar_map['100'])


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

# %%
print(spdata['1']['0'].cov_ltot.shape)
# [0,:,idx,idx]
print(spdata['1']['0'].variance.shape)
ll = np.arange(0,spdata['1']['0'].variance_min_ma.shape[0],1)
for idx, n in enumerate(["030", "044", "070", "100", "143", "217", "353"]):
    plt.plot(spdataNN['1']['0'].C_lS/np.sqrt(2 * spdata['1']['0'].cov_ltot[0,:,idx,idx] * spdata['1']['0'].cov_ltot[0,:,idx,idx]/((2*ll+1))), label=n)
plt.plot(spdataNN['1']['0'].C_lS/np.sqrt(spdataNN['1']['0'].variance[:,0,0]), label='optimal CV limit')
plt.plot(spdata['1']['0'].C_lS/np.sqrt(spdata['1']['0'].variance[:,0,0]), label='combined channels')
    # plt.plot(spdata['1']['0'].variance[:,0,0])
plt.legend()
plt.xlim((20,2000))
plt.title('EE-Spectrum and Noise - cosmic variance limit')
plt.xlabel('Multipole')
plt.ylabel('S/N')
plt.ylim((0,30))

# %%
ll = np.arange(0,spdata['1']['0'].variance_min_ma.shape[0],1)
for idx, n in enumerate(["030", "044", "070", "100", "143", "217", "353"]):
    plt.plot(2 * spdata['1']['0'].cov_ltot_min[0,:] * spdata['1']['0'].cov_ltot_min[0,:]/((2*ll+1)), label=n)
plt.plot((spdata['1']['0'].variance_min[:]), label='variance')
# plt.plot(spdata['1']['0'].C_lS/np.sqrt(spdata['1']['0'].variance[:,0,0]), label='combined channels')
    # plt.plot(spdata['1']['0'].variance[:,0,0])
plt.legend()
plt.xlim((20,2000))
plt.title('EE-Spectrum and Noise - cosmic variance limit')
plt.xlabel('Multipole')
# plt.ylabel('S/N')
# plt.ylim((0,30))
plt.yscale('log')

# %%


data = spdata['1']['0']
data_fullsky = spdata['1']['0']
C_lS = data.C_lS
cov_ltot_min = data.cov_ltot_min
variance_min_ma = data.variance_min_ma
variance = data.variance

fig, ax = plt.subplots(figsize=(8,6))
ll = np.arange(0,variance_min_ma.shape[0],1)
plt.plot(variance_min_ma, label='Variance, combined patches', lw=3, ls="-", color="black")
plt.plot(data_fullsky.variance_min_ma   , label="Variance, full sky", lw=3, ls="--", color="black")

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
plt.ylabel(r'Powerspectrum $D_l = \frac{l(l+1)C_l}{2\pi}$')
# %%
