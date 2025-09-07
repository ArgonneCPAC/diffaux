""" """

import numpy as np
from jax import random as jran

__all__ = ("bh_mass_from_bulge_mass", "monte_carlo_black_hole_mass")
fixed_seed = 43


def bh_mass_from_bulge_mass(bulge_mass):
    """
    Kormendy & Ho (2013) fitting function for the Mbh--Mbulge power law relation.

    Parameters
    ----------
    bulge_mass : ndarray
        Numpy array of shape (ngals, ) storing the stellar mass of the bulge
        in units of solar mass assuming h=0.7

    Returns
    -------
    bh_mass : ndarray
        Numpy array of shape (ngals, ) storing black hole mass

    """
    prefactor = 0.49 * (bulge_mass / 100.0)
    return prefactor * (bulge_mass / 1e11) ** 0.15


def monte_carlo_black_hole_mass(bulge_mass, ran_key):
    """
    Monte Carlo realization of the Kormendy & Ho (2013) fitting function
    for the Mbh--Mbulge power law relation.

    Parameters
    ----------
    bulge_mass : ndarray
        Numpy array of shape (ngals, ) storing the stellar mass of the bulge
        in units of Msun assuming h=0.7

    ran_key : jax.random.key
        Random number seed

    Returns
    -------
    bh_mass : ndarray
        Numpy array of shape (ngals, ) storing black hole mass in units of
        Msun assuming h=0.7

    """
    loc = np.log10(bh_mass_from_bulge_mass(bulge_mass))
    scale = 0.28
    lg_bhm = jran.normal(ran_key, shape=bulge_mass.shape) * scale + loc
    bh_mass = 10**lg_bhm

    return bh_mass
