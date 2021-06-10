"""run_skypatch.py: Main script

CMB_skypatch lets the user define skypatches at which cmb spectra are being calculated.
The skypatch spectra are combined to a single spectrum.
"""

import json
import numpy as np
import plot
import matplotlib.pyplot as plt
import healpy as hp
import component_separation.io as io
import matplotlib.gridspec as gridspec
import component_separation.powspec as pw
import itertools
from component_separation.cs_util import Config as csu
from lib_emp import Lib_emp

import cmb_skypatch
with open(os.path.dirname(cmb_skypatch.__file__)+'/config.json', "r") as f:
    cf = json.load(f)


uname = platform.uname()
if uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"

fname = io.make_filenamestring(cf)
speccs =  [spec for spec in ['EE']]
csu.PLANCKMAPFREQ_f
freqcomb =  ["{}-{}".format(p[0],p[1])
        for p in itertools.product(FREQS, FREQS)
        if (int(p[1])>=int(p[0]))]
dc = dcf["plot"]["spectrum"]