# -*- coding: utf-8 -*-
"""
Created on February 2023

@author: admartin, dlmccann
"""

import numpy as np
import xarray as xr
from scipy.optimize import least_squares
import seastar
from seastar.utils.tools import dotdict

# import seastar.gmfs.doppler
import pdb


def ambiguity_closest_to_truth(truth, results, method='windcurrent', windcurrentratio=10):
    """

    METHOD
    Distance calculated using:
    - wind and current with a given ratio between wind and current
    - only wind
    - only current

    Parameters
    ----------
    truth : ``xarray.Dataset``
        Truth in term of:
        - geophysical parameters: WindSpeed, WindDirection, CurrentVel, CurrentDir, others (waves)
        - geographic coordinates: longitude, latitude (optional?)
        ...
    results : ``xarray.Dataset``
        unsorted results


    Returns
    -------
    level2 : ``xarray.Dataset``
        L2.shape (antenna, across, along)
    """

    print("To Be Done")

    return level2

def solve_ambiguity():
    """
    As function of:
    - cost values: chi2
    - closest to truth
    ...



    """
    return level2



