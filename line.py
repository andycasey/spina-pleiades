
from astropy.table import Table

from utils import load_stan_model, sampling_kwds


spina = Table.read("Abundances_Pleiades_VS_Sun.csv")


#  y = mx + b for all stars simultaneously
# include intrinsic scatter and cluster abundance for all elements
