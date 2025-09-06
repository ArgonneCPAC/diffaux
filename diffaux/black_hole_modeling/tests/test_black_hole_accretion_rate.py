""""""

import numpy as np

from .. import black_hole_accretion_rate as bhar


def test_eddington_ratio_distribution():
    redshift = 0.5
    rate_table, pdf_table = bhar.eddington_ratio_distribution(redshift)
    assert np.all(np.isfinite(pdf_table))
