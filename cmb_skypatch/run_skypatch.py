"""run_skypatch.py: Main script

CMB_skypatch lets the user define skypatches at which cmb spectra are being calculated.
The skypatch spectra are combined to a single spectrum.
"""

import json
import numpy as np
import os
import cmb_skypatch.plot
import matplotlib.pyplot as plt
import healpy as hp
import component_separation.io as io
import matplotlib.gridspec as gridspec
import component_separation.powspec as pw
import itertools
import platform
from component_separation.cs_util import Config as csu
from component_separation.cs_util import Helperfunctions as hpf
from cmb_skypatch.lib_emp import Lib_emp
from cmb_skypatch.lib import Lib
from matplotlib.patches import Patch
import cmb_skypatch.plot as plot

import cmb_skypatch
with open(os.path.dirname(cmb_skypatch.__file__)+'/config.json', "r") as f:
    cf = json.load(f)

uname = platform.uname()
if uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"



if __name__ == '__main__':
    filename = io.make_filenamestring(cf)
    print(40*"$")
    print("Starting run with the following settings:")
    print(cf['pa'])
    print("Generated filename(s) for this session: {}".format(filenam--e))
    print(40*"$")

    # Load empiric data, one spectrum per detector
    #   - you may want to run component_separation to create them
    emp_C_ltot = io.load_data(io.spec_sc_path_name)
    emp_cov_ltot = pw.build_covmatrices(emp_C_ltot, cf['pa']['lmax'], cf['pa']['freqfilter'], cf['pa']['specfilter'])
    
    io.spec_sc_path_name
    emp_C_lN = io.load_data(io.noise_sc_path_name)
    emp_cov_lN = pw.build_covmatrices(emp_C_lN, cf['pa']['lmax'], cf['pa']['freqfilter'], cf['pa']['specfilter'])


    # Create a `Lib_emp` object. The data structure is very similar to the `Lib` object
    # Currently, `Lib_emp` doesnt support patches or smoothing, thus dictionary must be set
    # to {'1': { '0': [..] } }.
    emp = {'1': { '0': # emp[npatches][smoothing_par]
        Lib_emp(dov_ltot=np.array([emp_cov_ltot["EE"]]), dov_lN=np.array([emp_cov_lN["EE"]]))}}


    # Create a `Lib` object. One can decide the number of patches and smoothing, for which the spectra
    # are being generated. The spectra are 'analytic', i.e.
    #   * C_lS is taken from planck best fits,
    #   * C_lN is build by generating npatch patches of the sky, based on equi-noise-level-areas and is then
    #       deconvolved with the respective beamwindowfunction,
    #   * C_lF is currently not supported.
    # By creating the Lib-object, the parameter `cov_ltot_min` is computed, which is the
    spdata = dict()
    for n in cf['pa']['npatch']:
        if str(n) in spdata.keys():
            pass
        else:
            spdata.update({str(n): dict()})
        for smooth in cf['pa']['smoothing_par']:
            if str(smooth) in spdata[str(n)].keys():
                pass
            else:
                spdata[str(n)].update({str(smooth): 
                    Lib(npatch = n, smoothing_par = smooth, C_lF=None, C_lN=None)})


    ### When completed, one may compare `cov_ltot_min` for different number of skypatches.
    ### We expect, that cov_ltot_min decreases for increasing number of patches.
    ### plot.py covers diagram generation for visualising the result
    plot.spectrum_variance(emp['1']['0'], spdata['1']['0'], rd_ylim=(-0.2,0.7), npatches= cf['pa']['npatch'], smoothing_par = smoothing_par)
    plot.compare_variance_min(spdata)
    plot.compare_improvement(spdata, ["8_20-8_5","8_5-8_0"])
    plot.compare_errorbars(spdata, ['0'], show=False)