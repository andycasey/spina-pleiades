
from astropy.table import Table

from .utils import load_stan_model

model = load_stan_model("intrinsic_scatter.stan")

data = Table.read("Abundances_Pleiades_VS_Sun.csv")
