"""lib_emp.py: This serves as base library for sky patching. 
    The functionality of Lib_emp is limited. This class merely
    serves as a helper to serve the same datastructure for empiric data as it does for the
    `Lib` objects.

    The class is put in its own module to avoid executing smoothing, when only
    empiric data is used.

"""

__author__ = "S. Belkner"


import healpy as hp
import numpy as np
import numpy.ma as ma
import pandas as pd

from numpy import inf
import json
import platform
from component_separation.cs_util import Planckf, Plancks
import component_separation.io as io
import cmb_skypatch

__uname = platform.uname()
if __uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"

with open(os.path.dirname(cmb_skypatch.__file__)+'/config.json', "r") as f:
    cf = json.load(f)


class Lib_emp:
    """
    The functionality of Lib_emp is limited. This class merely serves as a helper
    to serve the same datastructure for empiric data as it does for the
    `Lib` objects.
    """

    spectrum_trth = pd.read_csv(
        cf[mch]['powspec_truthfile'],
        header=0,
        sep='    ',
        index_col=0)["Planck-"+"EE"]
    
    __lmax = cf['pa']['lmax']
    __detector = cf['pa']['detector']


    def __init__(self, C_lF = None, C_lN = None, C_lS = None, C_ltot= None,
        D_lF = None, D_lN = None, D_lS = None, D_ltot= None,
        cov_lF = None, cov_lN = None, cov_lS = None, cov_ltot = None,
        dov_lF = None, dov_lN = None, dov_lS = None, dov_ltot = None,
        C_lN_factor=1, fsky=1.):
        ll = np.arange(0,Lib_emp.__lmax+1,1)
        self.shape = (Lib_emp.__lmax+1, len(Lib_emp.__detector), len(Lib_emp.__detector))
        self.fsky = fsky

        if C_lF == None:
            self.C_lF = np.zeros(self.shape, float)
        self.C_lN = C_lN

        if C_lS == None:
            self.C_lS = Lib_emp.spectrum_trth[:self.shape[0]].to_numpy()/(ll*(ll+1))*2*np.pi
        else:
            self.C_lS = C_lS

        self.C_ltot = C_ltot
        self.cov_lS = cov_lS

        if cov_ltot is not None:
            self.cov_ltot = cov_ltot
        elif dov_ltot is not None:
            self.cov_ltot = dov_ltot/(ll*(ll+1))*2*np.pi

        if cov_lN is not None:
            self.cov_ltot = cov_ltot
        elif dov_lN is not None:
            self.cov_lN = dov_lN/(ll*(ll+1))*2*np.pi
        
        self.cov_ltot_min = np.concatenate(
            (np.zeros(shape=(self.cov_ltot.shape[0],1)),
             np.array([
                [Lib_emp.cov_l2cov_lmin(self.cov_ltot[n,:,:,l])
                    for l in range(1,self.cov_ltot.shape[3])]
                for n in range(self.cov_ltot.shape[0])])),axis=1)
        # self.cov_ltot_min = np.array([Lib_emp.cov_l2cov_lmin(cov_ltot[:,:,l])
        #             for l in range(cov_ltot.shape[2])])
        self.approx_variance = np.zeros((Lib_emp.__lmax+1,self.cov_ltot.shape[0],self.cov_ltot.shape[0]), float)
        for n in range(1):
            self.approx_variance[:,n,n] = 2 * self.cov_ltot_min[n,:] * self.cov_ltot_min[n,:]/((2*ll+1)*self.fsky)
        
        self.approx_variance[self.approx_variance == inf] = 0
        self.approx_variance_min = self.approx_variance[:,0,0]
        self.approx_variance_min_ma = ma.masked_array(self.approx_variance_min, mask=np.where(self.approx_variance<=0., True, False))


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