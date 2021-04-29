
import json

import sys
import matplotlib

import healpy as hp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable


def spectrum_variance():
    fig, ax = plt.subplots(figsize=(8,6))

    plt.plot(loc_opt_ma, label='Variance, combined patches', lw=3, ls="-", color="black")
    plt.plot(loc_var_FSma, label="Variance, full sky", lw=3, ls="--", color="black")

    plt.plot(loc_C_lS[:,0,0], label = 'Planck EE', lw=3, ls="-", color="red", alpha=0.5)
    for n in range(npatch):
        plt.plot(loc_cov_min[n,:], lw=1, ls="-", color="red", alpha=0.5)
    plt.plot(0,0, label='Minimal powerspectrum from patches', color='red', alpha=0.5, lw=1)

    plt.plot(2*loc_C_lS[:,0,0]*loc_C_lS[:,0,0]/((2*ll+1)), label="Variance, Planck EE, full sky", lw=3, ls="-", color="blue")
    plt.plot(loc_opt_NN, label = "Variance, Planck EE, combined patches", lw=3, ls="--", color="green")

    leg1 = plt.legend(loc='upper left')
    pa = [None for n in range(loc_var_patches.shape[1])]
    for n in range(loc_var_patches.shape[1]):
        p = plt.plot(loc_var_patches[:,n,n], lw=2, ls="-", alpha=0.5)#, color=CB_color_cycle[0])
    #     p = plt.plot(var_patches_NS[:,n,n], lw=2, ls="-", alpha=0.5)#, color=CB_color_cycle[0])
        col = p[0].get_color()
        pa[n] = Patch(facecolor=col, edgecolor='grey', alpha=0.5)
    leg2 = plt.legend(handles=pa,
            labels=["" for n in range(loc_var_patches.shape[1]-1)] + ['Variance, {} skypatches'.format(str(npatch))],
            ncol=loc_var_patches.shape[1], handletextpad=0.5, handlelength=0.5, columnspacing=-0.5,
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
    plt.title("Combining {} skypatches which have noise and signal".format(str(npatch)))
    plt.savefig("/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/vis/analytic/analyticpatching-NoiseSignal{}patches.jpg".format(str(npatch)))


def compare_variance():
    plt.figure(figsize=(8,6))
    # plt.plot(var_FSma/opt_ma-1, label='{} patches - Noise+Signal'.format(str(npatch)), lw=2, ls="-", color='black')
    # plt.plot(smn_var_FSma/smn_opt_ma-1, label='{} patches - 30 degree smoothed Noise+Signal'.format(str(npatch)), lw=2, ls="-")

    # plt.plot(patch50, label='50 patches ', lw=2, ls="-", color=CB_color_cycle[1])
    plt.plot(smn_patch2, label='2 patches ', lw=2, ls="-", color=CB_color_cycle[0])
    plt.plot(smn_patch4, label='4 patches ', lw=2, ls="-", color=CB_color_cycle[1])
    plt.plot(smn_patch8, label='8 patches ', lw=2, ls="-", color=CB_color_cycle[2])
    plt.plot(smn_patch16, label='16 patches ', lw=2, ls="-", color=CB_color_cycle[3])
    plt.plot(smn_patch32, label='32 patches ', lw=2, ls="-", color=CB_color_cycle[4])
    plt.plot(smn_patch64, label='64 patches ', lw=2, ls="-", color=CB_color_cycle[5])
    # plt.plot(patch40, label='40 patches ', lw=2, ls="-", color=CB_color_cycle[1])
    # plt.plot(patch50, label='50 patches ', lw=2, ls="-", color=CB_color_cycle[3])
    # plt.plot(patch80, label='80 patches ', lw=2, ls="-", color=CB_color_cycle[5])
    # plt.plot(patch100, label='100 patches ', lw=2, ls="-", color=CB_color_cycle[4])


    # plt.plot(patch8, label='8 patches ', lw=2, ls="-", color=CB_color_cycle[4])
    # plt.plot(patch16, label='16 patches ', lw=2, ls="-", color=CB_color_cycle[5])
    plt.hlines(0,1e0,3e3, color="black", ls="--")
    plt.xscale('log')
    plt.xlabel("Multipole")
    plt.ylabel(r"$\frac{\sigma^{2}_{Full}-\sigma^{2}_{patched}}{\sigma^{2}_{patched}}= \Delta \sigma^{2}_i$", fontsize=20)
    plt.title("FWHM 2degree gauss smoothing - Variance comparison")
    # plt.yscale('log')
    plt.legend(loc='upper left')
    plt.xlim((2e1,3e3))
    # plt.ylim((-.02,3))
    # plt.tight_layout()
    plt.savefig("/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/vis/analytic/comppachfullNoiseSignal2-64patches_smoothed.jpg".format(str(npatch)))


def compare_improvement():
    plt.title('FHWM 2 degree gauss smoothing - Comparison between improvements')
    plt.plot(smn_patch4/smn_patch2, label='i=4, j=2 patches ', lw=2, ls="-", color=CB_color_cycle[0])
    plt.plot(smn_patch8/smn_patch4, label='i=8, j=4 patches ', lw=2, ls="-", color=CB_color_cycle[1])
    plt.plot(smn_patch16/smn_patch8, label='i=16, j=8 patches ', lw=2, ls="-", color=CB_color_cycle[2])
    plt.plot(smn_patch32/smn_patch16, label='i=32, j=16 patches ', lw=2, ls="-", color=CB_color_cycle[3])
    plt.plot(smn_patch64/smn_patch32, label='i=64, j=32 patches ', lw=2, ls="-", color=CB_color_cycle[4])

    # plt.plot(smn_patch4/smn_patch2, label='i=100, j=50 patches ', lw=2, ls="-", color=CB_color_cycle[3])
    plt.ylabel(r"$\frac{\Delta \sigma^{2}_i}{\Delta \sigma^{2}_j}$", fontsize=20)
    # plt.xscale('log')
    # plt.plot(patch16/patch8, label='16/8 patches ', lw=2, ls="-", color=CB_color_cycle[3])
    # plt.plot(patch32/patch16, label='32/16 patches ', lw=2, ls="-", color=CB_color_cycle[4])
    plt.legend()
    plt.tight_layout()
    plt.savefig("/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/vis/analytic/compareimprovement2-64patches_smoothed.jpg".format(str(npatch)))
    # plt.ylim((1.8,2.01))


def compare_errorbars():
    fig, ax = plt.subplots(figsize=(8,6))
    plt.errorbar(x=range(lmax+1), y=C_lS[:,0,0], yerr=np.sqrt(var_FSma), label="Full sky", alpha=0.5)
    # plt.errorbar(x=range(lmax+1), y=C_lS[:,0,0], yerr=np.sqrt(var_FSma/(patch2+1)), label="2 patches", alpha=0.5)
    # plt.errorbar(x=range(lmax+1), y=C_lS[:,0,0], yerr=np.sqrt(var_FSma/(patch4+1)), label="4 patches", alpha=0.5)
    plt.errorbar(x=range(lmax+1), y=C_lS[:,0,0], yerr=np.sqrt(var_FSma/(smn_patch2+1)), label="2 patches", alpha=0.5)
    plt.errorbar(x=range(lmax+1), y=C_lS[:,0,0], yerr=np.sqrt(var_FSma/(smn_patch4+1)), label="4 patches", alpha=0.5)
    plt.errorbar(x=range(lmax+1), y=C_lS[:,0,0], yerr=np.sqrt(var_FSma/(smn_patch8+1)), label="8 patches", alpha=0.5)
    plt.errorbar(x=range(lmax+1), y=C_lS[:,0,0], yerr=np.sqrt(var_FSma/(smn_patch16+1)), label="16 patches", alpha=0.5)
    plt.errorbar(x=range(lmax+1), y=C_lS[:,0,0], yerr=np.sqrt(var_FSma/(smn_patch32+1)), label="32 patches", alpha=0.5)
    plt.errorbar(x=range(lmax+1), y=C_lS[:,0,0], yerr=np.sqrt(var_FSma/(smn_patch64+1)), label="64 patches", alpha=0.5)

    plt.legend()
    plt.yscale('log')
    # plt.tight_layout()
    plt.title('FWHM 2degree gauss smoothing - Optimal variances')

    plt.ylabel(r"Powerspectrum $C_l$ [$\mu K^2$]")
    plt.xlabel("Multipole")
    plt.xscale('log')
    plt.ylim((1e-5,1e-2))
    plt.xlim((1e1,3e3))
    plt.savefig("/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/vis/analytic/comperrorbars2-64-patches_smoothed.jpg".format(str(npatch)))
