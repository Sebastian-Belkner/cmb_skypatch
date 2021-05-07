# %%
import json
import numpy as np
import plot
import matplotlib.pyplot as plt
import healpy as hp
import component_separation.io as io
import matplotlib.gridspec as gridspec
from component_separation.cs_util import Planckf, Plancks
import component_separation.powspec as pw
import itertools
from lib_emp import Lib_emp
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
lmax = 3001
bins = np.logspace(np.log10(1), np.log10(3000+1), 200)
bl = bins[:-1]
br = bins[1:]

def _std_dev_binned(d):
    if type(d) == np.ndarray:
        val = np.nan_to_num(d)
    else:
        val = np.nan_to_num(d.to_numpy())
    n, _ = np.histogram(
        np.linspace(0,lmax,lmax),
        bins=bins)
    sy, _ = np.histogram(
        np.linspace(0,lmax,lmax),
        bins=bins,
        weights=val)
    sy2, _ = np.histogram(
        np.linspace(0,lmax,lmax),
        bins=bins,
        weights=val * val)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    return mean, std, _


# %%
emp_C_ltot = {freqc: {
        spec: np.array(io.load_cl(_inpathname(freqc,spec)))
        for spec in speccs}  
        for freqc in freqcomb}
emp_cov_ltot = pw.build_covmatrices(emp_C_ltot, 3000, cf['pa']['freqfilter'], cf['pa']['specfilter'])

dcf['pa']['freqdset'] = 'DX12-diff'
fname2 = io.make_filenamestring(dcf)
inpath_name2 = cs_abs_path+dc["indir_root"]+dc["indir_rel"]+dc["in_desc"]+fname2
emp_C_lN = io.load_spectrum(inpath_name2, fname2)
emp_cov_lN = pw.build_covmatrices(emp_C_lN, 3000, cf['pa']['freqfilter'], cf['pa']['specfilter'])


# %%
print(emp_cov_lN["EE"].shape)


# %%
emp = {'1': { '0':
    Lib_emp(dov_ltot=np.array([emp_cov_ltot["EE"]]), dov_lN=np.array([emp_cov_lN["EE"]]))}}


# %%
npatches = '1'
smoothing_par = '20'


# %% Noise Powerspectra
plt.figure(figsize=(8,6))
gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
ax0 = plt.subplot(gs[0])
plt.title('Noise spectra from data and theoretical model - {}patches - {} smoothing'.format(npatches, smoothing_par))
ll = np.arange(0,emp['1']['0'].cov_lN.shape[3],1)
ax0.plot(emp['1']['0'].C_lS, lw=3, color='darkred', label='Planck EE, best estimate')
for d in range(spdata[npatches][smoothing_par].cov_lN.shape[2]): 
    ax0.scatter(ll, emp['1']['0'].cov_lN[0,d,d,:], color='black', marker='.', s=2)
    ax0.plot(spdata[npatches][smoothing_par].cov_lN[0,:,d,d], color=CB_color_cycle[d],
        ls='--', lw=1, label="Theoretical, " + cf['pa']['detector'][d]+ "GHz channel",
        alpha=0.5)
    for n in range(1, spdata[npatches][smoothing_par].cov_lN.shape[0]):
        ax0.plot(spdata[npatches][smoothing_par].cov_lN[n,:,d,d], color=CB_color_cycle[d], ls='--', lw=1,
        alpha=0.5)
ax0.scatter(0,0,color='black', marker='.', s=8, label='Empirical (difference map)')
ax0.set_yscale('log')
ax0.set_xscale('log')
ax0.set_xlim((1e2,3e3))
ax0.set_ylim((1e-4,6e0))
ax0.set_ylabel(r'Powerspectrum $C_l$')
ax0.legend()
ax1 = plt.subplot(gs[1])
for d in range(emp['1']['0'].cov_lN.shape[1]):
    binmean, binerr , _ = _std_dev_binned(emp['1']['0'].cov_lN[0,d,d,:]/np.mean(spdata[npatches][smoothing_par].cov_lN[:,:,d,d],axis=0)-1)
    ax1.errorbar(0.5 * bl + 0.5 * br, binmean, binerr, fmt='x', capsize=1,
        color=CB_color_cycle[d], alpha=0.9)#, label=cf['pa']['detector'][d])
    # ax1.plot(emp['1']['0'].cov_lN[0,d,d,:]/spdata[npatches][smoothing_par].cov_lN[0,:,d,d]-1, color=CB_color_cycle[d], alpha=0.5)#, label=cf['pa']['detector'][d])
# ax1.set_yscale('log')
ax1.set_ylim((-0.2,0.5))
ax1.hlines(0,0,3000,  color='black')
ax1.set_ylabel(r'Rel. diff.')
ax1.set_xlabel(r'Multipole l')
ax1.set_xscale('log')
ax1.set_xlim((1e2,3e3))
plt.savefig('CompareNoisespectra{}patches{}smoothing.jpg'.format(npatches, smoothing_par))



# %% Noise and Signal Powerspectra emp['1']['0'].cov_ltot[0,d,d,:] spdata[npatches][smoothing_par].cov_ltot[n,:,d,d]
plt.figure(figsize=(8,6))
gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
ax0 = plt.subplot(gs[0])
plt.title('Spectra from data and theoretical model - {}patches - {} smoothing'.format(npatches, smoothing_par))
ll = np.arange(0,emp['1']['0'].cov_lN.shape[3],1)
ax0.plot(emp['1']['0'].C_lS, lw=3, color='darkred', label='Planck EE, best estimate')
for d in range(spdata[npatches][smoothing_par].cov_lN.shape[2]): 
    ax0.scatter(ll, emp['1']['0'].cov_ltot[0,d,d,:], color='black', marker='.', s=2)
    ax0.plot(spdata[npatches][smoothing_par].cov_ltot[0,:,d,d], color=CB_color_cycle[d],
        ls='--', lw=1, label="Theoretical, " + cf['pa']['detector'][d]+ "GHz channel",
        alpha=0.5)
    for n in range(1, spdata[npatches][smoothing_par].cov_ltot.shape[0]):
        ax0.plot(spdata[npatches][smoothing_par].cov_ltot[n,:,d,d], color=CB_color_cycle[d], ls='--', lw=1,
        alpha=0.5)
ax0.scatter(0,0,color='black', marker='.', s=8, label='Empirical')
ax0.set_yscale('log')
ax0.set_xscale('log')
ax0.set_xlim((1e2,3e3))
ax0.set_ylim((1e-4,6e0))
ax0.set_ylabel(r'Powerspectrum $C_l$')
ax0.legend()
ax1 = plt.subplot(gs[1])
for d in range(emp['1']['0'].cov_ltot.shape[1]):
    binmean, binerr , _ = _std_dev_binned(emp['1']['0'].cov_ltot[0,d,d,:]/np.mean(spdata[npatches][smoothing_par].cov_ltot[:,:,d,d],axis=0)-1)
    ax1.errorbar(0.5 * bl + 0.5 * br, binmean, binerr, fmt='x', capsize=1,
        color=CB_color_cycle[d], alpha=0.9)#, label=cf['pa']['detector'][d])
    # ax1.plot(emp['1']['0'].cov_lN[0,d,d,:]/spdata[npatches][smoothing_par].cov_lN[0,:,d,d]-1, color=CB_color_cycle[d], alpha=0.5)#, label=cf['pa']['detector'][d])
# ax1.set_yscale('log')
ax1.set_ylim((-0.2,0.5))
ax1.hlines(0,0,3000,  color='black')
ax1.set_ylabel(r'Rel. diff.')
ax1.set_xlabel(r'Multipole l')
ax1.set_xscale('log')
ax1.set_xlim((1e2,3e3))
plt.savefig('Comparespectra{}patches{}smoothing.jpg'.format(npatches, smoothing_par))
# ax1.legend()

# %%
mp = hp.mollview(spdata['1'][smoothing_par].noisevar_map['100'], title='{}GHz, with {} smoothing'.format('100', smoothing_par), return_projected_map=True)


# %% Optimal spectra
import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(8,6))
gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
ax0 = plt.subplot(gs[0])
plt.title('Optimal spectra from data and theoretical model - {}patches - {}smoothing'.format(npatches, smoothing_par))
ll = np.arange(0,emp['1']['0'].cov_ltot_min.shape[1],1)
ax0.plot(emp['1']['0'].C_lS, lw=3, color='darkred', label='Planck EE, best estimate')
ax0.scatter(ll, emp['1']['0'].cov_ltot_min[0,:], color='black', marker='.', s=2, label="Empirical")
for n in range(spdata[npatches][smoothing_par].cov_ltot.shape[0]):
    ax0.plot(spdata[npatches][smoothing_par].cov_ltot_min[n,:], color=CB_color_cycle[0],
        ls='--', lw=2,
        alpha=0.5)
ax0.plot(0,0,label="Theoretical, {} patches".format(npatches), color=CB_color_cycle[0])

ax0.set_yscale('log')
ax0.set_xscale('log')
ax0.set_xlim((1e2,3e3))
ax0.set_ylim((1e-4,3e-3))
ax0.set_ylabel(r'Powerspectrum $C_l$')
ax0.legend()
ax1 = plt.subplot(gs[1])
for n in range(emp['1']['0'].cov_ltot_min.shape[0]):
    binmean, binerr , _ = _std_dev_binned(emp['1']['0'].cov_ltot_min[n,:]/np.mean(spdata[npatches][smoothing_par].cov_ltot_min[:,:],axis=0)-1)
    ax1.errorbar(0.5 * bl + 0.5 * br, binmean, binerr, fmt='x', capsize=1,
        color=CB_color_cycle[n], alpha=0.9)#, label=cf['pa']['detector'][d])
    # ax1.plot(emp['1']['0'].cov_lN[0,d,d,:]/spdata[npatches][smoothing_par].cov_lN[0,:,d,d]-1, color=CB_color_cycle[d], alpha=0.5)#, label=cf['pa']['detector'][d])
# ax1.set_yscale('log')
ax1.set_ylim((-0.2,0.3))
ax1.hlines(0,0,3000,  color='black')
ax1.set_ylabel(r'Rel. diff.')
ax1.set_xlabel(r'Multipole l')
ax1.set_xscale('log')
ax1.set_xlim((1e2,3e3))
newax = fig.add_axes([0.8, 0.8, 0.2, 0.2], anchor='NE', zorder=-1)
newax.axis('off')
newax.imshow(mp)
plt.savefig('Compareoptimalspectra{}patches{}smoothing.jpg'.format(npatches, smoothing_par))


# %% Variance
npatches='1'
from matplotlib.patches import Patch
plt.figure(figsize=(8,6))
gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
ax0 = plt.subplot(gs[0])
ll = np.arange(0,spdata[npatches][smoothing_par].approx_variance_min_ma.shape[0],1)
ax0.plot(spdata[npatches][smoothing_par].approx_variance_min_ma, label='Variance, combined patches', lw=3, ls="-", color="black")
ax0.scatter(ll, emp['1']['0'].approx_variance_min_ma, label="Variance, empiric data", lw=2, ls="--", color="black", s=3, marker='x')

ax0.plot(spdata[npatches][smoothing_par].C_lS, label = 'Planck EE', lw=3, ls="-", color="red", alpha=0.5)
for n in range(spdata[npatches][smoothing_par].cov_ltot_min.shape[0]):
    ax0.plot(spdata[npatches][smoothing_par].cov_ltot_min[n,:], lw=1, ls="-", color="red", alpha=0.5)
ax0.plot(0,0, label='Minimal powerspectrum from patches', color='red', alpha=0.5, lw=1)

ax0.plot(2*spdata[npatches][smoothing_par].C_lS*spdata[npatches][smoothing_par].C_lS/((2*ll+1)), label="Variance, Planck EE, full sky", lw=3, ls="-", color="blue")
# plt.plot(loc_opt_NN, label = "Variance, Planck EE, combined patches", lw=3, ls="--", color="green")

leg1 = ax0.legend(loc='upper left')
pa = [None for n in range(spdata[npatches][smoothing_par].approx_variance.shape[1])]
for n in range(len(pa)):
    p = ax0.plot(spdata[npatches][smoothing_par].approx_variance[:,n,n], lw=2, ls="-", alpha=0.5)
    col = p[0].get_color()
    pa[n] = Patch(facecolor=col, edgecolor='grey', alpha=0.5)
leg2 = ax0.legend(handles=pa[:min(20,len(pa))],
        labels=["" for n in range(min(20,len(pa))-1)] + ['Variance, {} skypatches'.format(npatches)],
        ncol=spdata[npatches][smoothing_par].approx_variance.shape[1], handletextpad=0.5, handlelength=0.5, columnspacing=-0.5,
        loc='lower left')
ax0.add_artist(leg1)

ax0.set_xscale('log')
ax0.set_yscale('log')
ax0.set_ylabel(r"Variance $\sigma^2$ [$\mu K^4$]")
ax0.set_xlim((2e1,3e3))
ax0.set_ylim((1e-10,1e-2))
# plt.tight_layout()
ax2 = ax0.twinx()
ax2.tick_params(axis='y', labelcolor="red")
ax2.set_ylabel(r'Powerspectrum $C_l$ [$\mu K^2$]', color= 'red')
ax2.set_ylim((1e-10,1e-2))
ax2.set_yscale('log')
ax0.set_title("Combining {} skypatches".format(npatches))
ax1 = plt.subplot(gs[1])
for n in range(emp['1']['0'].cov_ltot_min.shape[0]):
    binmean, binerr , _ = _std_dev_binned(emp['1']['0'].approx_variance_min/spdata[npatches][smoothing_par].approx_variance_min-1)
    ax1.errorbar(0.5 * bl + 0.5 * br, binmean, binerr, fmt='x', capsize=1,
        color='black', alpha=0.9)
# ax1.scatter(ll, _std_dev_binned(emp.approx_variance_min_ma/variance_min_ma-1, label='Ratio', color='black', s=3, marker='x')
ax1.set_xlabel("Multipole")
ax1.set_ylabel(r"Rel. diff.")
ax1.set_ylim((-0.2,0.7))
ax1.set_xscale('log')
ax1.set_xlim((2e1,3e3))
ax1.hlines(0,2e1,3e3, color='black', ls='--')
plt.savefig("/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/cmb_skypatch/vis/variance-NoiseSignal{}patches{}smoothing.jpg".format(npatches, smoothing_par))

# ax1.legend()
# %%
from lib import Lib


# %%
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
plot.spectrum_variance(emp['1']['0'], spdata['1']['0'], rd_ylim=(-0.2,0.7), npatches= npatches, smoothing_par = smoothing_par)


# %%
plot.compare_variance_min(spdata)


# %%
plot.compare_improvement(spdata, ["8_20-8_5","8_5-8_0"])


# %%
# plot.compare_errorbars(emp, ['0'])
plot.compare_errorbars(spdata, ['0'])


# %%
spdata['1']['0'].C_lN


# %%
hp.mollview(spdata['1']['20'].noisevar_map['100'], title='{}GHz, with {} smoothing'.format('100', '0'))


# %%
print(spdata['1']['0'].cov_ltot.shape)
# [0,:,idx,idx]
print(spdata['1']['0'].approx_variance.shape)
ll = np.arange(0,spdata['1']['0'].approx_variance_min_ma.shape[0],1)
for idx, n in enumerate(["030", "044", "070", "100", "143", "217", "353"]):
    plt.plot(spdataNN['1']['0'].C_lS/np.sqrt(2 * spdata['1']['0'].cov_ltot[0,:,idx,idx] * spdata['1']['0'].cov_ltot[0,:,idx,idx]/((2*ll+1))), label=n)
plt.plot(spdataNN['1']['0'].C_lS/np.sqrt(spdataNN['1']['0'].approx_variance[:,0,0]), label='optimal CV limit')
plt.plot(spdata['1']['0'].C_lS/np.sqrt(spdata['1']['0'].approx_variance[:,0,0]), label='combined channels')
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
