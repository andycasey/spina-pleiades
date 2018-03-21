
import numpy as np
from astropy.table import Table
from pystan.plots import traceplot

from utils import load_stan_model, sampling_kwds


model = load_stan_model("line_with_cluster_abundances.stan")

spina = Table.read("Abundances_Pleiades_VS_Sun.csv")
# Exclude Vesta and sort so it matches existing figure:
indices = np.array([5, 3, 4, 1, 2])
spina = spina[indices]

# Sort 


# Problematic elements (e.g., duplicated ionisation states, missing values)
#exclude_elements = ["[CuI]", "[CrII]", "[TiII]", "[FeII]"]
exclude_elements = []

# Generate data dictionary.
elements = list(set(spina.dtype.names[1::2]).difference(exclude_elements))
S, A = len(spina), len(elements)

y = np.array([spina[element] for element in elements])
yerr = np.array([spina["err_{}".format(el.strip("[]"))] for el in elements])

for i, element in enumerate(elements):
    y[i][spina[element].mask] = np.nan
    yerr[i][spina[element].mask] = np.nan

is_missing = ~np.isfinite(y)
row_indices, column_indices = np.where(is_missing)

filler_error_value = 10**6
filler_value = dict([(column_index, np.nanmean(y.T[column_index])) \
    for column_index in column_indices])

y_ranges = np.array([y - 5 * yerr, y + 5 * yerr])

for row_index, column_index in zip(row_indices, column_indices):
    y[row_index, column_index] = filler_value[column_index]
    yerr[row_index, column_index] = filler_error_value

Lodders = dict([
    ("C", 40),
    ("Na", 958),
    ("Mg", 1336),
    ("Al", 1653),
    ("Si", 1310),
    ("S", 664),
    ("Ca", 1517),
    ("Sc", 1659),
    ("Ti", 1582),
    ("V", 1429),
    ("Cr", 1296),
    ("Mn", 1158),
    ("Fe", 1334),
    ("Co", 1352),
    ("Ni", 1353),
    ("Cu", 1037),
    ("Zn", 726),
    ("Y", 1659),
    ("O", 182),
])

x = np.array([Lodders[el.strip("[I]")] for el in elements]).astype(float)

init = dict(c=np.zeros(A), m=np.zeros(S), scatter=0.001 * np.ones(A),
    ln_f=0, mean_cluster_abundance=np.median(y))


data = dict(S=S, A=A, x=x, y=y.T, yerr=yerr.T, min_y=np.nanmin(y_ranges) ,
    max_y=np.nanmax(y_ranges))
op_params = model.optimizing(data=data, iter=100000, init=init)


import matplotlib.pyplot as plt
fig, axes = plt.subplots(S)
for i, ax in enumerate(axes):

    ax.scatter(np.repeat(x, S), (y.T - op_params["c"]).T.flatten(),
        facecolor="#666666",  s=1)
    ax.scatter(x, y.T[i] - op_params["c"], facecolor="b")
    ax.axhline(0)

    idx = np.argsort(x)
    line = op_params["m"][i] * x 
    ax.plot(x[idx], line[idx], c="r")

samples = model.sampling(**sampling_kwds(data=data, init=op_params,
    chains=2, iter=10000))

# Make trace plots of things we care about.
fig = traceplot(samples, pars=("c", "m", "ln_f"))
fig.tight_layout()
fig.savefig("lines_with_cluster_abundances_trace.pdf", dpi=300)


# Draw figures.
chains = samples.extract()
N, _ = chains["m"].shape
x_sort = np.argsort(x)


#fig, axes = plt.subplots(S)
fig = plt.figure(figsize=(6.5, 6.05))
from matplotlib.gridspec import  GridSpec
from matplotlib.ticker  import  MaxNLocator
gs = GridSpec(S, 2, width_ratios=[4, 1])

offset = np.median(chains["c"], axis=0)
line_axes = []
hist_axes = []
for i in range(S):
    ax = fig.add_subplot(gs[i, 0])
    ax.axhline(0, c="#666666", linestyle=":", zorder=-100)

    indices = np.random.choice(N, size=100, replace=False)

    f = np.median(np.exp(chains["ln_f"]))
    total_yerr = np.sqrt(((1 + f) * yerr.T[i])**2 + np.var(chains["c"][:, i]))
    ok = ~(total_yerr >= filler_error_value)

    ax.scatter(x[ok], (y.T[i] - offset)[ok], facecolor='k')
    ax.errorbar(x[ok], (y.T[i] - offset)[ok], 
        yerr=total_yerr[ok], 
        fmt=None, ecolor="k")

    lines = np.dot(chains["m"][:, i].reshape((-1, 1)), np.atleast_2d(x))

    p16, p50, p84 = np.percentile(lines, [5, 50, 95], axis=0)

    ax.fill_between(x[x_sort], p16[x_sort], p84[x_sort],
        facecolor="r", alpha=0.5, zorder=-1)
    ax.plot(x[x_sort], p50[x_sort], c="r", lw=1, zorder=0)

    if not ax.is_last_row():
        ax.set_xticks([])

    line_axes.append(ax)

    ax = fig.add_subplot(gs[i, 1])
    ax.hist(chains["m"][:, i] * 1e5, bins=50, facecolor="r", histtype="step",
        edgecolor="r")
    ax.axvline(0, c="#666666", zorder=-1, linestyle=":")

    ax.set_yticks([])
    hist_axes.append(ax)


ylimits = np.hstack([ax.get_ylim() for ax in line_axes]).flatten()
ylimits = (np.min(ylimits), np.max(ylimits))

for i, ax in enumerate(line_axes):
    ax.set_ylim(ylimits)
    ax.set_ylabel(r"$[{\rm X}/{\rm H}]$")

    ax.text(0.025, 0.9, spina["id"][i], transform=ax.transAxes,
        horizontalalignment="left", verticalalignment="top")

xlimits = np.hstack([ax.get_xlim() for ax in hist_axes]).flatten()
xlimits = (np.min(xlimits), np.max(xlimits))


for ax in hist_axes:
    ax.set_xlim(xlimits)
    #xtick_formatter = ScalarFormatter()
    #xtick_formatter.set_powerlimits((-5, 5))
    ax.xaxis.set_major_locator(MaxNLocator(3))
    #ax.xaxis.set_major_formatter(xtick_formatter)
    ##    verticalalignment="top", horizontalalignment="right")

    ax.set_xlabel(r"$m$ $(10^{-5})$")

fig.tight_layout()
fig.subplots_adjust(wspace=0)

fig.savefig("lines_with_cluster_abundances_fits.pdf", dpi=300)



