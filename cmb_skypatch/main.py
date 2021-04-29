# %%
from lib import Lib
# 2*0.0174533
comb5 = Lib(npatch = 5, smoothing_par = 0.0, C_lF=None, C_lN=None)
comb10 = Lib(npatch = 10, smoothing_par = 0.0, C_lF=None, C_lN=None)


# %%
plot.compare_errorbars()


# %%
