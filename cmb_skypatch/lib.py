"""lib.py: This serves as base library for sky patching.

"""

__author__ = "S. Belkner"


import json
import os
import platform

import component_separation.io as io
import healpy as hp
import numpy as np
import numpy.ma as ma
import pandas as pd
from component_separation.cs_util import Planckf, Plancks
from numpy import inf
import cmb_skypatch

__uname = platform.uname()
if __uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"

with open(os.path.dirname(cmb_skypatch.__file__)+'/config.json', "r") as f:
    cf = json.load(f)

class Lib:
    PLANCKSPECTRUM = [p.value for p in list(Plancks)]
    PLANCKMAPFREQ = [p.value for p in list(Planckf)]


    spectrum_trth = pd.read_csv(
        cf[mch]['powspec_truthfile'],
        header=0,
        sep='    ',
        index_col=0)["Planck-"+"EE"]
    
    # as we are only interested in noise var map, take DX12 data.
    buff = cf['pa']["freqdset"]
    cf['pa']["freqdset"] = 'DX12'
    noisevar_map_raw = io.load_plamap(cf, field=7)
    cf['pa']["freqdset"] = buff

    smoothed_noisevar_map = dict()
    for smooth in cf['pa']['smoothing_par']:
        if float(smooth) != 0.0:
            if str(float(smooth)) not in smoothed_noisevar_map.keys():
                smoothed_noisevar_map[str(float(smooth))] = dict()
            for key, val in noisevar_map_raw.items():
                if key not in smoothed_noisevar_map[str(float(smooth))].keys():
                    smoothed_noisevar_map[str(float(smooth))][key] = dict()
                smoothed_noisevar_map[str(float(smooth))].update({key:
                    hp.smoothing(val, fwhm=smooth*0.0174533, iter=0)})

    __lmax = cf['pa']['lmax']
    __detector = cf['pa']['detector']
    __freqc = [n+"-"+n for n in __detector]
    __beamf = io.load_beamf(__freqc)


    def __init__(self, npatch, smoothing_par = 0, C_lF = None, C_lN = None, C_lS = None, C_lN_factor=1):
        self.shape = (Lib.__lmax+1, len(Lib.__detector), len(Lib.__detector))
        self.npatch = npatch

        if float(smoothing_par) != 0.0:
            self.noisevar_map = Lib.smoothed_noisevar_map[str(float(smoothing_par))]
        else:
            self.noisevar_map = Lib.noisevar_map_raw

        if C_lF is None:
            self.C_lF = np.zeros(self.shape, float)

        self.noiselevel = np.array([self.varmap2noiselevel(self.noisevar_map[freq]) for freq in Lib.__detector])
        if C_lN is None:
            self.C_lN = self.beamf2C_lN(Lib.__beamf, self.noiselevel, Lib.__freqc)*C_lN_factor
        else:
            self.C_lN = C_lN
        ll = np.arange(0,Lib.__lmax+1,1)
        if C_lS is None:
            self.C_lS = Lib.spectrum_trth[:self.shape[0]].to_numpy()/(ll*(ll+1))*2*np.pi
        else:
            self.C_lS = C_lS
        
        self.cov_lS = (np.ones((len(Lib.__detector),len(Lib.__detector),Lib.__lmax+1))* self.C_lS).T
        self.cov_lS[:10,:,:] = np.zeros((10, len(Lib.__detector), len(Lib.__detector)))
        
        self.cov_lN = np.array([
            self.C_lN2cov_lN(self.C_lN[:,n,:]) for n in range(self.C_lN.shape[1])])
        self.fsky = np.zeros((npatch, npatch), float)
        np.fill_diagonal(self.fsky, 1/npatch*np.ones((npatch)))

        self.cov_ltot = self.cov_lS + self.cov_lN
        self.cov_ltot_min = np.concatenate(
            (np.zeros(shape=(self.cov_ltot.shape[0],1)),
             np.array([
                [Lib.cov_l2cov_lmin(self.cov_ltot[n,l])
                    for l in range(1,self.cov_ltot.shape[1])]
                for n in range(self.cov_ltot.shape[0])])),axis=1)

        self.approx_variance = np.zeros((Lib.__lmax+1,self.cov_ltot.shape[0],self.cov_ltot.shape[0]), float)
        for n in range(npatch):
            self.approx_variance[:,n,n] = 2 * self.cov_ltot_min[n,:] * self.cov_ltot_min[n,:]/((2*ll+1)*self.fsky[n,n])
        self.approx_variance[self.approx_variance == inf] = 0
        self.approx_variance_min = np.zeros((self.approx_variance.shape[0]))
        for l in range(self.approx_variance.shape[0]):
            try:
                self.approx_variance_min[l] = np.nan_to_num(
                    Lib.cov_l2cov_lmin(self.approx_variance[l]))#np.sqrt(np.sqrt(fsky[0,0]))))
            except:    
                pass
        self.approx_variance_min_ma = ma.masked_array(self.approx_variance_min, mask=np.where(self.approx_variance_min<=0, True, False))


    def beamf2C_lN(self, beamf, dp, freqc):
        TEB_dict = {
            "T": 0,
            "E": 1,
            "B": 2
        }
        LFI_dict = {
            "030": 28,
            "044": 29,
            "070": 30
        }
        local = np.zeros((len(Lib.__detector), self.npatch, Lib.__lmax+1))
        C = 0
        for ndetector in range(local.shape[0]):
            hdul = beamf[freqc[ndetector]]
            freqs = freqc[ndetector].split('-')
            for npatch in range(local.shape[1]):
                if int(freqs[0]) >= 100 and int(freqs[1]) >= 100:
                    C = 1 / hdul["HFI"][1].data.field(TEB_dict["E"])[:Lib.__lmax+1]**2
                elif int(freqs[0]) < 100 and int(freqs[1]) < 100:
                    b = np.sqrt(hdul["LFI"][LFI_dict[freqs[0]]].data.field(0))
                    buff = np.concatenate((
                        b[:min(Lib.__lmax+1, len(b))],
                        np.array([np.NaN for n in range(max(0, Lib.__lmax+1-len(b)))])))
                    C = 1 / buff**2
                if int(freqs[0])<100:
                    nside = 1024
                else:
                    nside = 2048
                local[ndetector,npatch,:] = C * hp.nside2pixarea(nside) * 1e12 * dp[ndetector,npatch]
        return local


    def C_lN2cov_lN(self, C_lN):
        """Creates auto and cross covariance matrix, with noise only, so no cross.

        Args:
            spectrum ([type]): [description]

        Returns:
            [type]: [description]
        """   
        row, col = np.diag_indices(C_lN.shape[0])
        C = np.zeros(self.shape, float)
        for l in range(C.shape[0]):
            C[l,row,col] = C_lN[:,l]
        return C
    

    def cov2weight(self, cov):
        elaw = np.ones(cov.shape[1])
        weights = (cov @ elaw) / (elaw @ cov @ elaw)[np.newaxis].T
        return weights


    @staticmethod
    def cov_l2cov_lmin(C_l) -> np.array:
        """Returns the minimal covariance using inverse variance weighting, i.e. calculates
        :math:`C^{\texttt{min}}_l = \frac{1}{\textbf{1}^\dagger (C_l^S + C_l^F + C_l^N)^{-1}\textbf{1}}`.

        Args:
            C_l (np.ndarray): An array of auto- and cross-covariance matrices for all instruments, its dimension is [Nspec,Nspec,lmax]

        """
        def isDiag(M):
            i, j = M.shape
            assert i == j 
            test = M.reshape(-1)[:-1].reshape(i-1, j+1)
            return ~np.any(test[:, 1:])

        def invdiagmat(C):
            import copy
            ret = copy.deepcopy(C)
            row, col = np.diag_indices(ret.shape[0])
            ret[row, col] = 1/np.diag(ret)
            return ret

        elaw = np.ones(C_l.shape[-1])
        if isDiag(C_l):
            inv = invdiagmat(C_l)
        else:
            inv = np.linalg.inv(C_l)

        cov_minimal = elaw @ inv @ elaw
        return 1/cov_minimal


    def varmap2noiselevel(self, varmap):
        varmap = np.where(varmap==0.0, np.mean(varmap), varmap)
        patch_bounds = np.array(list(range(self.npatch+1)))/self.npatch
        mean, binedges = np.histogram(varmap, bins=np.logspace(np.log10(varmap.min()),np.log10(varmap.max()),10000))
        patch_noiselevel = np.zeros((len(patch_bounds)-1))
        buff=0
        buff2=0
        patchidx=0
        # boundaries = np.zeros((self.npatch))
        noisebuff = 0
        for idx,n in enumerate(mean):
            buff += mean[idx]
            buff2 += mean[idx]

            if buff <= patch_bounds[patchidx+1] * len(varmap):
                noisebuff +=  mean[idx] * (binedges[idx+1]+binedges[idx])/2
            else:

                # boundaries[patchidx] = (binedges[idx+1]+binedges[idx])/2
                patch_noiselevel[patchidx] = noisebuff/buff2
                buff2=0
                patchidx+=1
                noisebuff=0
        # boundaries[-1] = (binedges[-2]+binedges[-1])/2

        patch_noiselevel[-1] = noisebuff/(buff2)
        # plt.vlines(np.concatenate(([data.min()],boundaries)), ymin=0, ymax=1e6, color='red', alpha=0.5)
        
        print(patch_noiselevel.shape, np.mean(patch_noiselevel))
        return patch_noiselevel
        # noise_level = np.array(get_noiselevel(noisevar_map['143'], self.npatch, '044'))
