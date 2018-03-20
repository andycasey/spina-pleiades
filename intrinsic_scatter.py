
from astropy.table import Table
from pystan.plots import traceplot

from utils import load_stan_model, sampling_kwds

model = load_stan_model("intrinsic_scatter.stan")

spina = Table.read("Abundances_Pleiades_VS_Sun.csv")

elements = spina.dtype.names[1::2]

all_samples = dict()

for i, element in enumerate(elements):

    print(i, element)

    y = spina[element][1:]
    yerr = spina["err_{}".format(element.strip("[]"))][1:]

    data = dict(S=y.size, y=y, yerr=yerr)

    op_params = model.optimizing(data=data)
    samples = model.sampling(**sampling_kwds(data=data, init=op_params))

    all_samples[element] = samples

    fig = traceplot(samples, pars=("abundance", "intrinsic_scatter"))
    fig.tight_layout()
    fig.savefig("figures/intrinsic_scatter_{}_trace.png".format(
        element.strip("[])")), dpi=300)

