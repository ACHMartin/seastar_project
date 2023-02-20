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


def ambiguity_closest_to_truth(lmout, truth, windcurrentratio=10):
    """
    Define a dictionary (windcurrent, wind, current) of indexes for lmout.x on Ambiguities closest to the Truth.
    Closest distances are defined as euclidian distance on the wind only (wind), current only (current)
    or combining distance between wind and current using the given windcurrentratio (windcurrent).
    Parameters
    ----------
    lmout : ``xarray.Dataset``
        Have to contain the following ".x" data_vars and
         coordinates "x_variables" = [u,v,c_u,c_v]; "Ambiguities";
    truth : ``xarray.Dataset``
        Have to contain the following geophysical parameters:
        WindSpeed, WindDirection, CurrentVelocity, CurrentDirection, others (waves)
        Should be in the same dimension as lmout
    windcurrentratio : ``float``
        ratio to combine the distance between wind and current. Default value of 10
    Returns
    -------
    index : ``dict`` of ``xarray.DataArray``
        with keys "windcurrent", "wind", "current" corresponding on which quantity the distance is calculated
        combining dataArray. Can be applied directly to lmout with lmout.isel(Ambiguities=ind['windcurrent'])
    """

    mytruth = xr.Dataset()
    (mytruth['WindU'], mytruth['WindV']) = \
        seastar.utils.tools.windSpeedDir2UV(truth.WindSpeed, truth.WindDirection)
    (mytruth['CurrentU'], mytruth['CurrentV']) = \
        seastar.utils.tools.currentVelDir2UV(truth.CurrentVelocity, truth.CurrentDirection)
    mytruth['x'] = xr.concat(
        [mytruth['WindU'], mytruth['WindV'], mytruth['CurrentU'], mytruth['CurrentV']],
        'x_variables'
    )

    err = truth - lmout # keep only the 'x' variable

    err['dist_x_reduce'] = xr.concat(
        [err.x.sel(x_variables='u')**2 + err.x.sel(x_variables='v')**2,
         windcurrentratio * (err.x.sel(x_variables='c_u')**2 + err.x.sel(x_variables='c_v')**2) ],
        'x_reduce'
    )
    err.coords['x_reduce'] = ['uv','c_uv']

    ind = dict({
        'windcurrent': err.dist_x_reduce.sum(dim='x_reduce').argmin(dim='Ambiguities'),
        'wind': err.dist_x_reduce.sel(x_reduce='uv').argmin(dim='Ambiguities'),
        'current': err.dist_x_reduce.sel(x_reduce='c_uv').argmin(dim='Ambiguities'),
    })

    return ind

def ambiguity_sort_by_cost(lmout):
    """
    Return an indexes for lmout on Ambiguities for the smallest cost.
    Parameters
    ----------
    lmout : ``xarray.Dataset``
        Have to contain the following ".cost" data_vars
    Returns
    -------
    index : ``xarray.DataArray``
        a dataArray index. Can be applied directly to lmout with lmout.isel(Ambiguities=index_cost)
    """
    index = lmout.cost.argmin(dim='Ambiguities')
    return index



def solve_ambiguity(lmout, ambiguity):
    """
    As function of:
    - cost values: chi2
    - closest to truth
    ...
method='windcurrent',


    """

    if ambiguity == 'sort_by_cost':
        lmout = ambiguity_sort_by_cost(lmout)


    print("To Be Done - solve_ambiguity")

    return lmout



