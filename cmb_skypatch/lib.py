"""lib.py: This serves as base library for the sky patching.

"""

__author__ = "S. Belkner"


import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
from matplotlib.patches import Patch
from numpy import inf
import json
import platform
from component_separation.cs_util import Planckf, Plancks
import component_separation.io as io


class Lib:
    PLANCKSPECTRUM = [p.value for p in list(Plancks)]
    PLANCKMAPFREQ = [p.value for p in list(Planckf)]
    __uname = platform.uname()
    if __uname.node == "DESKTOP-KMIGUPV":
        __mch = "XPS"
    else:
        __mch = "NERSC"
    with open('/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/cmb_skypatch/config.json', "r") as f:
        __cf = json.load(f)
    __cf[__mch]["indir"] = "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/data/"
    __spectrum_trth = pd.read_csv(
        "/mnt/c/Users/sebas/OneDrive/Desktop/Uni/"+__cf[__mch]['powspec_truthfile'],
        header=0,
        sep='    ',
        index_col=0)
    
    __noisevar_map = io.load_plamap_new(__cf, field=7)
    __lmax = __cf['pa']['lmax']
    __detector = __cf['pa']['detector']
    __freqc = [n+"-"+n for n in __detector]
    __beamf = io.load_beamf(__freqc, abs_path="/mnt/c/Users/sebas/OneDrive/Desktop/Uni/")


    def __init__(self, npatch, smoothing_par = 0, C_lF = None, C_lN = None, C_lS = None):
        self.shape = (Lib.__lmax+1, len(Lib.__detector), len(Lib.__detector))
        self.npatch = npatch

        if float(smoothing_par != 0.0):
            self.noisevar_map = {
                key: hp.smoothing(val, fwhm=smoothing_par)
                    for key, val in Lib.__noisevar_map.items()}
        else:
            self.noisevar_map = Lib.__noisevar_map

        if C_lF == None:
            self.C_lF = np.zeros(self.shape, float)

        self.noiselevel = np.array([self.varmap2noiselevel(self.noisevar_map[freq]) for freq in Lib.__detector])

        if C_lN == None:
            self.C_lN = self.beamf2C_lN(Lib.__beamf, self.noiselevel, Lib.__freqc)
        else:
            self.C_lN = C_lN
        
        if C_lS == None:
            self.C_lS = Lib.__spectrum_trth[:self.shape[0]].to_numpy()
        else:
            self.C_lS = C_lS
        
        ll = np.arange(0,Lib.__lmax+1,1)
        self.cov_lS = (np.ones((len(Lib.__detector),len(Lib.__detector),Lib.__lmax+1))* self.C_lS/(ll*(ll+1))*2*np.pi).T
        self.cov_lS[:10,:,:] = np.zeros((10, len(Lib.__detector), len(Lib.__detector)))
        
        self.cov_lN = self.C_lN2cov_lN()

        self.fsky = np.zeros((npatch, npatch), float)
        np.fill_diagonal(self.fsky, 1/npatch*np.ones((npatch)))

        self.cov_ltot = self.cov_lS + self.cov_lN
        self.cov_ltot_min = np.concatenate(
            (np.zeros(shape=(self.cov_ltot.shape[0],1)),
             np.array([
                [Lib.cov_l2cov_lmin(self.cov_ltot[n,l])
                    for l in range(1,self.cov_ltot.shape[1])]
                for n in range(self.cov_ltot.shape[0])])),axis=1)

        self.variance = np.zeros((Lib.__lmax+1,self.cov_ltot.shape[0],self.cov_ltot.shape[0]), float)
        for n in range(npatch):
            self.variance[:,n,n] = 2 * self.cov_ltot_min[n,:] * self.cov_ltot_min[n,:]/((2*ll+1)*self.fsky[n,n])
        self.variance_min = np.zeros((self.variance.shape[0]))
        for l in range(self.variance.shape[0]):
            try:
                self.variance_min[l] = np.nan_to_num(
                    Lib.cov_l2cov_lmin(self.variance[l]))#np.sqrt(np.sqrt(fsky[0,0]))))
            except:    
                pass


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
        for n in range(self.C_lN.shape[0]):
            hdul = beamf[freqc]
            freqs = freqc[n].split('-')
            for m in range(self.C_lN.shape[1]):
                if int(freqs[0]) >= 100 and int(freqs[1]) >= 100:
                    ret = 1 / hdul["HFI"][1].data.field(TEB_dict["E"])[:Lib.__lmax+1]**2
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
                local[n,m,:] = C * hp.nside2pixarea(nside) * 1e12 * dp
        return local


    def C_lN2cov_lN(self):
        """Creates auto and cross covariance matrix, with noise only, so no cross.

        Args:
            spectrum ([type]): [description]

        Returns:
            [type]: [description]
        """    

        row, col = np.diag_indices(self.C_lN.shape[0])
        C = np.zeros(self.shape, float)
        for l in range(C.shape[0]):
            C[l,row,col] = self.C_lN[:,l]
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
        return patch_noiselevel
    # noise_level = np.array(get_noiselevel(noisevar_map['143'], self.npatch, '044'))