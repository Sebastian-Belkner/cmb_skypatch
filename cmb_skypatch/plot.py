
import json

import sys
import random
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import healpy as hp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from component_separation.cs_util import Config as csu
from component_separation.cs_util import Helperfunctions as hpf
import numpy as np
import os
import platform
import pandas as pd

from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
CB_color_cycle = ["#88CCEE", "#CC6677", "#DDCC77", "#117733", "#332288", "#AA4499", 
                             "#44AA99", "#999933", "#882255", "#661100", "#6699CC", "#888888"]

uname = platform.uname()
if uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"
    
import cmb_skypatch
with open(os.path.dirname(cmb_skypatch.__file__)+'/config.json', "r") as f:
    cf = json.load(f)


def spectrum_variance(data, rd_ylim = (-0.2,0.7), npatch='1', smoothing_par='0', show=False):
    C_lS = data[npatch]['0'].C_lS
    cov_ltot_min = data[npatch]['0'].cov_ltot_min
    variance_min_ma = data[npatch]['0'].approx_variance_min_ma
    variance = data[npatch]['0'].approx_variance
    CB_color_cycle = ["#88CCEE", "#CC6677", "#DDCC77", "#117733", "#332288", "#AA4499", 
                                 "#44AA99", "#999933", "#882255", "#661100", "#6699CC", "#888888","#88CCEE", "#CC6677", "#DDCC77", "#117733", "#332288", "#AA4499", 
                                 "#44AA99", "#999933", "#882255", "#661100", "#6699CC", "#888888"]
    plt.figure(figsize=(8,6))

    gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
    ax0 = plt.subplot(gs[0])
    ll = np.arange(0,variance_min_ma.shape[0],1)

    ax0.plot(data['1']['0'].approx_variance_min_ma, label='Variance, 1 patch', lw=3, ls="-", color="black", alpha=0.5)
    ax0.plot(data[npatch]['0'].approx_variance_min_ma, label='Variance, {} combined patches'.format(npatch), lw=3, ls="-", color="black")
    # ax0.scatter(
    #     ll,
    #     data_empiric.approx_variance_min_ma*(hpf.llp1e12(np.arange(0,3000+1,1)))**2*1e-24/0.670629620552063,
    #     label="Minimal variance, empiric data",
    #     lw=1, ls="--", color="black", s=3, marker='x', alpha=0.5)

    ax0.plot(2*C_lS*C_lS/((2*ll+1)), label="Variance, Planck EE, full sky", lw=3, ls="-", color="blue")
    # plt.plot(loc_opt_NN, label = "Variance, Planck EE, combined patches", lw=3, ls="--", color="green")

    ax0.plot(C_lS, label = 'Planck EE', lw=3, ls="-", color="red", alpha=0.5)

    for n in range(cov_ltot_min.shape[0]):
        ax0.plot(cov_ltot_min[n,:], lw=1, ls="-", color="red", alpha=0.5)
    ax0.plot(0,0, label='Minimal powerspectrum from patches', color='red', alpha=0.5, lw=1)

    leg1 = ax0.legend(loc='upper left')
    pa = [None for n in range(variance.shape[1])]
    for n in range(len(pa)):
        p = ax0.plot(variance[:,n,n], lw=2, ls="-", alpha=0.3)
        col = p[0].get_color()
        pa[n] = Patch(facecolor=col, edgecolor='grey', alpha=0.3)
    leg2 = ax0.legend(handles=pa[:min(20,len(pa))],
            labels=["" for n in range(min(20,len(pa))-1)] + ['Variance, {} skypatches'.format(str(cov_ltot_min.shape[0]))],
            ncol=variance.shape[1], handletextpad=0.5, handlelength=0.5, columnspacing=-0.5,
            loc='lower left')

    ax0.add_artist(leg1)
    ax0.set_xscale('log')
    ax0.set_yscale('log')
    ax0.set_ylabel(r"Variance $\sigma^2$ [$\mu K^4$]")
    ax0.set_xlim((2e1,3e3))
    # ax0.set_ylim((1e-11,1e-7))
    ax0.set_ylim((1e-10,1e-2))
    # plt.tight_layout()
    ax2 = ax0.twinx()
    ax2.tick_params(axis='y', labelcolor="red")
    ax2.set_ylabel(r'Powerspectrum $C_l$ [$\mu K^2$]', color= 'red')
    ax2.set_ylim((1e-10,1e-2))
    # ax2.set_ylim((1e-11,1e-7))
    ax2.set_yscale('log')
    ax0.set_title("Combining {} skypatches".format(str(cov_ltot_min.shape[0])))
    ax1 = plt.subplot(gs[1])
    binwidth = 200
    bins = np.logspace(np.log10(1), np.log10(cf['pa']['lmax']+1), binwidth)
    bl = bins[:-1]
    br = bins[1:]
    # binmean, binerr , _ = hpf.std_dev_binned(
    #         data_empiric.cov_ltot_min[0,:-1]*hpf.llp1e12(np.arange(0,3000,1))*1e-12/data[npatch]['0'].cov_ltot_min[0,:-1]-1,
    #         lmax=cf['pa']['lmax'],
    #         binwidth=binwidth)
    # binmean, binerr , _ = hpf.std_dev_binned(
    #     (data_empiric.approx_variance_min_ma*(hpf.llp1e12(np.arange(0,3000+1,1)))**2*1e-24/0.670629620552063/data[npatch]['0'].approx_variance_min_ma).data-1,
    #     lmax=cf['pa']['lmax']+1,
    #     binwidth=binwidth)
    # binmean, binerr , _ = hpf.std_dev_binned(
    #     (data['1']['0'].approx_variance_min_ma/data['16']['0'].approx_variance_min_ma).data-1,
    #     lmax=cf['pa']['lmax']+1,
    #     binwidth=binwidth)
    # ax1.errorbar(0.5 * bl + 0.5 * br, binmean, binerr, fmt='x', capsize=1,
    #         color=CB_color_cycle[n], alpha=0.9, label = '1 patch over 16 patch')

    plt.plot((data['1']['0'].approx_variance_min_ma/data[npatch]['0'].approx_variance_min_ma).data[:2000]-1,
            color='black', alpha=0.9, label = '1 patch over {} patches'.format(npatch), lw=2)
    ax1.legend()
    # ax1.scatter(ll, hpf.std_dev_binned(data_empiric.approx_variance_min_ma/variance_min_ma-1, label='Ratio', color='black', s=3, marker='x')
    ax1.set_xlabel("Multipole")
    ax1.set_ylabel(r"Rel. diff.")
    ax1.set_ylim((-0.1,1.0))
    ax1.set_xscale('log')
    ax1.set_xlim((2e1,3e3))
    ax1.hlines(0,2e1,3e3, color='black', ls='--')
    

def compare_variance_min(data, show=False):
    plt.figure(figsize=(8,6))
    it=0
    for npatch, smdegdic in data.items():
        it2 = 0
        pr=''
        for smooth_par, val in smdegdic.items():
            if int(npatch)>1:
                pr+=smooth_par+'-'
                if it2==len(list(smdegdic.items()))-1:
                    plt.plot(data['1'][smooth_par].approx_variance_min_ma/val.approx_variance_min_ma-1, label='{} patches - {} degree smoothing '.format(npatch, pr), lw=2, ls="-", color=CB_color_cycle[it])
                else:
                    plt.plot(data['1'][smooth_par].approx_variance_min_ma/val.approx_variance_min_ma-1, lw=2, ls="-", color=CB_color_cycle[it])
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
    
    if show:
        plt.show()
    plt.savefig("{}compare_variance_patches{}_smoothings{}.jpg".format(cf[mch]['outdir_vis_ap'], str([key for key, val in data.items()]), pr))


def compare_improvement(data, compare_conf, show=False):
    plt.title('Comparison between improvements')
    it=0
    for comp_item in compare_conf:
        item1, item2 = comp_item.split('-')
        patch1, smooth1 = item1.split('_')
        patch2, smooth2 = item2.split('_')
        val = (data['1'][smooth1].approx_variance_min_ma - data[patch1][smooth1].approx_variance_min_ma)/(data['1'][smooth2].approx_variance_min_ma - data[patch2][smooth2].approx_variance_min_ma)
        plt.plot(val, label='i={}, j={} patches -- i={}, j={} smoothing'.format(patch1, patch2, smooth1, smooth2), lw=2, ls="-")
        it+=1
    plt.ylabel(r"$\frac{\Delta \sigma^{2}_i}{\Delta \sigma^{2}_j}$", fontsize=20)
    plt.legend()
    plt.tight_layout()
    plt.title("Improvement comparison")
        
    if show:
        plt.show()
    plt.savefig("{}compare_variance_improvement_{}.jpg".format(cf[mch]['outdir_vis_ap'], str(compare_conf)))


def compare_errorbars(data, user_smoothpar, show=False):
    fig, ax = plt.subplots(figsize=(8,6))
    it=0
    for npatch, smdegdic in data.items():
        pr=''
        for smooth_par, val in smdegdic.items():
            if smooth_par in user_smoothpar:
                pr+=smooth_par
                plt.errorbar(
                    x=range(data['1']['0'].C_lS.shape[0]),
                    y=data['1']['0'].C_lS,
                    yerr=np.sqrt(val.approx_variance_min_ma),
                    label="{} patches - {} deg smooth".format(npatch, smooth_par),
                    alpha=0.5,
                    color=CB_color_cycle[it])
                it+=1
    plt.legend()
    plt.yscale('log')
    # plt.tight_layout()
    plt.title('Optimal errorbars')

    plt.ylabel(r"Powerspectrum $C_l$ [$\mu K^2$]")
    plt.xlabel("Multipole")
    plt.xscale('log')
    plt.ylim((1e-5,1e-2))
    plt.xlim((1e1,3e3))
    if show:
        plt.show()
    plt.savefig("{}compare_errorbars{}_smoothings{}.jpg".format(cf[mch]['outdir_vis_ap'], str([key for key, val in data.items()]), pr))


def s_over_n(data, show=False):
    ll = np.arange(0,spdata['1']['0'].approx_variance_min_ma.shape[0],1)
    for idx, n in enumerate(["030", "044", "070", "100", "143", "217", "353"]):
        plt.plot(spdataNN['1']['0'].C_lS/np.sqrt(2 * spdata['1']['0'].cov_ltot[0,:,idx,idx] * spdata['1']['0'].cov_ltot[0,:,idx,idx]/((2*ll+1))), label=n)
    plt.plot(spdataNN['1']['0'].C_lS/np.sqrt(spdataNN['1']['0'].approx_variance[:,0,0]), label='optimal CV limit')
    plt.plot(spdata['1']['0'].C_lS/np.sqrt(spdata['1']['0'].approx_variance[:,0,0]), label='combined channels')
        # plt.plot(spdata['1']['0'].approx_variance[:,0,0])
    plt.legend()
    plt.xlim((20,2000))
    plt.title('EE-Spectrum and Noise - cosmic variance limit')
    plt.xlabel('Multipole')
    plt.ylabel('S/N')
    plt.ylim((0,30))
    if show:
        plt.show()