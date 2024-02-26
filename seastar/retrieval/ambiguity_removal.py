# -*- coding: utf-8 -*-
"""Functions for ambiguitay removal of simultaneous retrieval data products."""

import numpy as np
import xarray as xr
from scipy.optimize import least_squares
import seastar
from seastar.utils.tools import dotdict
from seastar.retrieval.spatial_ambiguity_selection import solve_ambiguity_spatial_selection
# import seastar.gmfs.doppler
import pdb


def ambiguity_closest_to_truth(lmout, truth, windcurrentratio=10):
    """
    Find ambiguity closest to truth.

    Define a dictionary (windcurrent, wind, current) of indexes for lmout.x
    on Ambiguities closest to the Truth. Closest distances are defined as
    euclidian distance on the wind only (wind), current only (current) or
    combining distance between wind and current using the given windcurrent
    ratio (windcurrent).

    Parameters
    ----------
    lmout : ``xarray.Dataset``
        Have to contain the following `.x` data_vars and
         coordinates `x_variables` = [u,v,c_u,c_v]; `Ambiguities`;
    truth : ``xarray.Dataset``
        Have to contain the following geophysical parameters:
        EarthRelativeWindSpeed, EarthRelativeWindDirection,
        CurrentVelocity, CurrentDirection, others (waves)
        Should be in the same dimension as lmout
    windcurrentratio : ``float``
        ratio to combine the distance between wind and current.
        Default value of 10
    Returns
    -------
    index : ``dict`` of ``xarray.DataArray``
        with keys `windcurrent`, `wind`, `current` corresponding on
        which quantity the distance is calculated combining dataArray.
        Can be applied directly to lmout with
        ``lmout.isel(Ambiguities=ind['windcurrent'])``
    """

    mytruth = xr.Dataset()
    (mytruth['EarthRelativeWindU'], mytruth['EarthRelativeWindV']) = \
        seastar.utils.tools.windSpeedDir2UV(
            truth.EarthRelativeWindSpeed, truth.EarthRelativeWindDirection)
    (mytruth['CurrentU'], mytruth['CurrentV']) = \
        seastar.utils.tools.currentVelDir2UV(
            truth.CurrentVelocity, truth.CurrentDirection)
    mytruth['x'] = xr.concat(
        [mytruth['EarthRelativeWindU'], mytruth['EarthRelativeWindV'],
            mytruth['CurrentU'], mytruth['CurrentV']],
        'x_variables'
    )

    err = mytruth - lmout  # keep only the 'x' variable

    err['dist_x_reduce'] = xr.concat(
        [err.x.sel(x_variables='u')**2 + err.x.sel(x_variables='v')**2,
         windcurrentratio * (err.x.sel(x_variables='c_u')**2 + err.x.sel(x_variables='c_v')**2)],
        'x_reduce'
    )
    err.coords['x_reduce'] = ['uv', 'c_uv']

    ind_dict = dict({
        'windcurrent': err.dist_x_reduce.sum(dim='x_reduce').argmin(dim='Ambiguities'),
        'wind': err.dist_x_reduce.sel(x_reduce='uv').argmin(dim='Ambiguities'),
        'current': err.dist_x_reduce.sel(x_reduce='c_uv').argmin(dim='Ambiguities'),
    })

    return ind_dict


def ambiguity_sort_by_cost(lmout):
    """
    Find ambiguity by cost.

    Return an indexes for lmout on Ambiguities for the smallest cost.

    Parameters
    ----------
    lmout : ``xarray.Dataset``
        Have to contain the following `.cost` data_vars
    Returns
    -------
    index : ``xarray.DataArray``
        a dataArray index. Can be applied directly to lmout with
        ``lmout.isel(Ambiguities=index_cost)``
    """
    cost = xr.where(np.isnan(lmout.cost), np.inf, lmout.cost)
    index = cost.argmin(dim='Ambiguities')
    return index


def solve_ambiguity(lmout, ambiguity):
    """
    Solve ambiguity.

    Send back the solution after resolving the ambiguities following
    the algorithm defined in the ambiguity dictionary.

    Parameters
    ----------
    lmout : ``xarray.Dataset``
        dataset of N dimension with required fields, .x and .cost
        and required coords: Ambiguities, x_variables
    ambiguity : ``dict``
        Dictionary with keys `name`:
        - name = `sort_by_cost`
        - name = `closest_truth`
            `truth` HAVE to be in the dict.
            optional `method` within `windcurrent` (default), `wind`, `current`
            optional `windcurrentratio` default = 10
        - name = `spatial_selection`
            `costfunction` HAVE to be in the dict
            `initial solution` HAVE to be in the dict
            optional 'windcurrentratio' default = 10
            optional 'iterationnumber' default = 2
            optional 'window' default = 3
    Returns
    ----------
    sol : ``xarray.Dataset``
        solution with ambiguities resolved
    """
    if 'name' not in ambiguity:
        raise Exception(
            "'name' should be provided with value between 'sort_by_cost', 'closest_truth', or 'spatial_selection'")

    if ambiguity['name'] == 'sort_by_cost':
        index = ambiguity_sort_by_cost(lmout)
        sol = lmout.isel(Ambiguities=index)
    elif ambiguity['name'] == 'closest_truth':
        if 'method' not in ambiguity:
            ambiguity['method'] = 'windcurrent'
        elif ambiguity['method'] not in ['windcurrent', 'wind', 'current']:
            raise Exception(
                "ambiguity['method'] should be 'windcurrent', 'wind' or 'current'")
        if 'windcurrentratio' not in ambiguity:
            ambiguity['windcurrentratio'] = 10
        elif not ambiguity['windcurrentratio'] > 0:
            raise Exception("ambiguity.windcurrentratio should be positive")
        if 'truth' not in ambiguity:
            raise Exception("ambiguity['truth'] HAVE to be provided for closest_truth method with"
                            "same size as lmout")
        index_dict = ambiguity_closest_to_truth(lmout,
                                                ambiguity['truth'],
                                                windcurrentratio=ambiguity['windcurrentratio'])
        sol = lmout.isel(Ambiguities=index_dict[ambiguity['method']])
    elif ambiguity['name'] == 'spatial_selection':
        if 'costfunction' not in ambiguity:
            raise Exception(
                "ambiguity['costfunction'] HAVE to be provided for spatial_selection method"
                "Must take single cell from `lmout`, a box around it from `initial` as input, weight"
                "and return total cost for all for ambiguities")
        elif 'initial solution' not in ambiguity:
            raise Exception(
                "ambiguity['initial solution'] HAVE to be provided for spatial_selection method")
        if 'windcurrentratio' not in ambiguity:
            ambiguity['windcurrentratio'] = 10
        elif not ambiguity['windcurrentratio'] > 0:
            raise Exception("ambiguity.windcurrentratio should be positive")
        if 'iterationnumber' not in ambiguity:
            ambiguity['iterationnumber'] = 2
        if 'window' not in ambiguity:
            ambiguity['window'] = 3
        sol = solve_ambiguity_spatial_selection(lmout,
                                                ambiguity['initial solution'],
                                                cost_function=ambiguity['costfunction'],
                                                pass_number=ambiguity['iterationnumber'],
                                                weight=ambiguity['windcurrentratio'],
                                                window=ambiguity['window'])
    else:
        raise Exception(
            "ambiguity['name'] should be 'sort_by_cost', 'closest_truth', or 'spatial_selection'")

    return sol
