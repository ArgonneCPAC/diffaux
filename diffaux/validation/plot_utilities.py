import os
import re

import numpy as np


def get_subsample(dlen, N=10):
    sample = np.random.sample(dlen)
    fraction = min(float(N) / float(dlen), 1.0)
    index = sample < fraction
    Nobj = np.count_nonzero(index)
    print(f"Selected {Nobj} objects")

    return Nobj, index


def get_nrow_ncol(nq, nrow=2):
    nrow = nrow if nq <= 6 else 3
    nrow = 1 if nq < 4 else nrow
    nrow = 4 if nq > 12 else nrow
    ncol = int(np.ceil(float(nq) / float(nrow)))

    return nrow, ncol


def save_fig(fig, plotdir, pltname):
    figname = os.path.join(plotdir, pltname)
    fig.savefig(figname)  # , bbox_inches='tight')
    print(f"Saving {figname}")


def fix_plotid(plotid):
    plotid = re.sub(r"_\*", "", plotid)
    plotid = re.sub(r"\*", "", plotid)
    plotid = "_" + plotid if len(plotid) > 0 and plotid[0] != "_" else plotid
    return plotid
