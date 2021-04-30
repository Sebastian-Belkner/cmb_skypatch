# %%
from lib import Lib
import json
import plot
import matplotlib.pyplot as plt

with open('/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/cmb_skypatch/config.json', "r") as f:
    cf = json.load(f)

spdata = dict()
CB_color_cycle = ["#88CCEE", "#CC6677", "#DDCC77", "#117733", "#332288", "#AA4499", 
                             "#44AA99", "#999933", "#882255", "#661100", "#6699CC", "#888888"]


# %%
npatch = [1,2,4,8,16,32]
smoothing_par = cf['pa']['smoothing_par']
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
print(spdata)


# %%
plot.spectrum_variance(spdata['1']['1'], spdata['128']['10'])
# plot.spectrum_variance(spdata['1']['1'], spdata['4']['1'])


# %%
plot.compare_variance_min(spdata)


# %%
plot.compare_improvement(spdata, ["4_0-2_0","4_1-2_1"])


# %%
plot.compare_errorbars(spdata, ['0'])


# %%
