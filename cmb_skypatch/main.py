# %%
from lib import Lib

print(Lib)

# %%
npatch = 5
comb = Lib(npatch, smoothing_par = 2*0.0174533, C_lF=None, C_lN=None)
print(comb.noisevar_map)
# %%

# %%