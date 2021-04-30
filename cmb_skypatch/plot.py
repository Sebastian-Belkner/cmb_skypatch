
import json

import sys
import random
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import healpy as hp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
CB_color_cycle = ["#88CCEE", "#CC6677", "#DDCC77", "#117733", "#332288", "#AA4499", 
                             "#44AA99", "#999933", "#882255", "#661100", "#6699CC", "#888888"]


def spectrum_variance(data_fullsky, data):
    C_lS = data.C_lS
    cov_ltot_min = data.cov_ltot_min
    variance_min_ma = data.variance_min_ma
    variance = data.variance

    fig, ax = plt.subplots(figsize=(8,6))
    ll = np.arange(0,variance_min_ma.shape[0],1)
    plt.plot(variance_min_ma, label='Variance, combined patches', lw=3, ls="-", color="black")
    plt.plot(data_fullsky.variance_min_ma, label="Variance, full sky", lw=3, ls="--", color="black")

    plt.plot(C_lS, label = 'Planck EE', lw=3, ls="-", color="red", alpha=0.5)
    for n in range(cov_ltot_min.shape[0]):
        plt.plot(cov_ltot_min[n,:], lw=1, ls="-", color="red", alpha=0.5)
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
    plt.xlim((2e1,3e3))
    plt.ylim((1e-10,1e-2))
    # plt.tight_layout()
    ax2 = ax.twinx()
    ax2.tick_params(axis='y', labelcolor="red")
    ax2.set_ylabel(r'Powerspectrum $C_l$ [$\mu K^2$]', color= 'red')
    plt.ylim((1e-10,1e-2))
    plt.yscale('log')
    plt.title("Combining {} skypatches".format(str(cov_ltot_min.shape[0])))
    plt.savefig("/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/cmb_skypatch/vis/analyticpatching-NoiseSignal{}patches.jpg".format(str(cov_ltot_min.shape[0])))


def compare_variance_min(data):
    plt.figure(figsize=(8,6))
    it=0
    for npatch, smdegdic in data.items():
        it2 = 0
        pr=''
        for smooth_par, val in smdegdic.items():
            if int(npatch)>1:
                pr+=smooth_par+'-'
                if it2==len(list(smdegdic.items()))-1:
                    plt.plot(data['1'][smooth_par].variance_min_ma/val.variance_min_ma-1, label='{} patches - {} degree smoothing '.format(npatch, pr), lw=2, ls="-", color=CB_color_cycle[it])
                else:
                    plt.plot(data['1'][smooth_par].variance_min_ma/val.variance_min_ma-1, lw=2, ls="-", color=CB_color_cycle[it])
                it2+=1
        it+=1

    plt.hlines(0,1e0,3e3, color="black", ls="--")
    plt.xscale('log')
    plt.xlabel("Multipole")
    plt.ylabel(r"$\frac{\sigma^{2}_{Full}-\sigma^{2}_{patched}}{\sigma^{2}_{patched}}= \Delta \sigma^{2}_i$", fontsize=20)
    plt.title("Variance comparison")
    # plt.yscale('log')
    plt.legend(loc='upper left')
    plt.xlim((2e1,3e3))
    # plt.ylim((-.02,3))
    # plt.tight_layout()
    plt.savefig("/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/cmb_skypatch/vis/compare_variance_patches{}_smoothings{}.jpg".format(str([key for key, val in data.items()]), str(list(data[random.choice(list(data.items()))[0]].values()))))


def compare_improvement(data, compare_conf):
    plt.title('Comparison between improvements')
    it=0
    for comp_item in compare_conf:
        item1, item2 = comp_item.split('-')
        patch1, smooth1 = item1.split('_')
        patch2, smooth2 = item2.split('_')
        val = (data['1'][smooth1].variance_min_ma - data[patch1][smooth1].variance_min_ma)/(data['1'][smooth2].variance_min_ma - data[patch2][smooth2].variance_min_ma)
        plt.plot(val, label='i={}, j={} patches -- i={}, j={} smoothing'.format(patch1, patch2, smooth1, smooth2), lw=2, ls="-")
        it+=1
    plt.ylabel(r"$\frac{\Delta \sigma^{2}_i}{\Delta \sigma^{2}_j}$", fontsize=20)
    plt.legend()
    plt.tight_layout()
    plt.title("Improvement comparison")
    plt.savefig("/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/cmb_skypatch/vis/compare_variance_improvement_{}.jpg".format(str(compare_conf)))


def compare_errorbars(data, user_smoothpar):
    fig, ax = plt.subplots(figsize=(8,6))
    it=0
    for npatch, smdegdic in data.items():
        for smooth_par, val in smdegdic.items():
            if smooth_par in user_smoothpar:
                plt.errorbar(x=range(data['1']['0'].C_lS.shape[0]), y=data['1']['0'].C_lS, yerr=np.sqrt(val.variance_min_ma), label="{} patches - {} deg smooth".format(npatch, smooth_par), alpha=0.5)
    plt.legend()
    plt.yscale('log')
    # plt.tight_layout()
    plt.title('Optimal errorbars')

    plt.ylabel(r"Powerspectrum $C_l$ [$\mu K^2$]")
    plt.xlabel("Multipole")
    plt.xscale('log')
    plt.ylim((1e-5,1e-2))
    plt.xlim((1e1,3e3))
    plt.savefig("/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/cmb_skypatch/vis/compare_errorbars{}_smoothings{}.jpg".format(str([key for key, val in data.items()]), str(list(data[random.choice(list(data.items()))[0]].values()))))
