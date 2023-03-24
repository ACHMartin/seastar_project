# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 16:48:48 2022

@author: admartin, dlmccann
"""
import multiprocessing
import numpy as np
import xarray as xr
import seastar
from seastar.retrieval import cost_function, ambiguity_removal
# from seastar.utils.tools import da2py

# import pdb # pdb.set_trace() # where we want to start to debug

def find_minima_parallel_task(element):
    """
    Parallel processing task
    
    Defines the task to be passed to multiprocessing.map() for MPI

    Parameters
    ----------
    element : ``xr.Dataset``
        Dataset containing  `level1`, `noise` and `gmf` data

    Returns
    -------
    lmout : ``xr.Dataset``
        Dataset containing minima information from least squares

    """
    sl1 = element['level1']
    sn = element['noise']
    gmf = element['gmf']
    lmout = cost_function.find_minima(sl1,
                                      sn,
                                      gmf,
                                      )
    lmout = lmout.sortby('cost')
    return lmout


def wind_current_retrieval(level1, noise, gmf, ambiguity):
    """
    Compute ocean surface WIND and CURRENT magnitude and direction
    by minimisation of a cost function.

    Compute Ocean Surface Vector Wind (OSVW) in (m/s) and
    direction (degrees N) in the meteorological convention (coming from).
    Assumed neutral wind at 10m.

    Compute Total Surface Current Vector (TSCV) in (m/s) and
    direction (degrees N) in the oceanographic convention (going to).

    Parameters
    ----------
    level1 : ``xarray.Dataset``
        L1 observable noisy dataset (Sigma0, RSV, geometry)
    noise : ``xarray.Dataset``
    gmf : ``dict``
        Dict of dicts containing names of the `doppler` and `nrcs` gmfs
    ambiguity : ``dict``
    Returns
    -------
    level2 : ``xarray.Dataset``
        L2 dataset with a new dimension with ambiguity
        L2.shape (ambiguity, x, y)
    level2.CurrentMagnitude : ``xarray.DataArray``
        Magnitude of surface current vector (m/s)
    level2.CurrentDirection : ``xarray.DataArray``
        Surface current direction (degrees N) in oceanographic convention
        (going to)
    level2.WindSpeed : ``xarray.DataArray``
        Ocean Surface Wind Speed (m/s)
    level2.WindDirection : ``xarray.DataArray``
        Ocean Surface Wind Direction (degrees N) in meteorologic convention
        (coming from)
    """

    lmout = run_find_minima(level1, noise, gmf)
    sol = ambiguity_removal.solve_ambiguity(lmout, ambiguity)

    level2 = sol2level2(sol)

    return level2

def sol2level2(sol):
    """
    Convert solution.x into WindU, WindV, WindSpeed, WindCurrent, CurrentU, V, Velocity, Direction

    Parameters
    ----------
    sol : ``xarray.Dataset``
        solution without ambiguities with ".x" field
    Returns
    -------
    level2 : ``xarray.Dataset``
    """
    level2 = sol.drop_vars(sol.data_vars)
    level2['x'] = sol.x  # .isel(Ambiguities=0)
    level2['cost'] = sol.cost
    level2['CurrentU'] = level2.x.sel(x_variables='c_u')
    level2['CurrentV'] = level2.x.sel(x_variables='c_v')
    level2['WindU'] = level2.x.sel(x_variables='u')
    level2['WindV'] = level2.x.sel(x_variables='v')

    [level2['CurrentVelocity'], cdir] = \
        seastar.utils.tools.currentUV2VelDir(
            level2['CurrentU'],
            level2['CurrentV']
        )
    level2['CurrentDirection'] = (level2.CurrentVelocity.dims, cdir)

    [level2['WindSpeed'], wdir] = \
        seastar.utils.tools.windUV2SpeedDir(
            level2['WindU'],
            level2['WindV']
        )
    level2['WindDirection'] = (level2.WindSpeed.dims, wdir)

    return level2

def run_find_minima(level1, noise, gmf):
    """
    Run find minima on xD dimension DataSet.
    Parameters
    ----------
    level1 : ``xarray.Dataset``
        L1 observables noisy dataset (Sigma0, RSV, geometry)
    noise : ``xarray.Dataset``
        Defined noise with data_vars "Sigma0", "RSV" on the same grid as level1
    gmf : ``dict``
    Returns
    -------
    sol : ``xarray.Dataset``
        x dimension dataset of find_minima output containing among other ".x = [u,v,c_u,c_v]" and ".cost"
        with dimension along "Ambiguities" of size=4 by construction.
    """
    list_L1s0 = list(level1.Sigma0.dims)
    list_L1s0.remove('Antenna')
    
    # Vectorize input data for parallel implementation
    if len(list_L1s0) > 1:  # 2d or more
        level1_stack = level1.stack(z=tuple(list_L1s0))
        noise_stack = noise.stack(z=tuple(list_L1s0))
        input_mp = [None] * level1_stack.z.size
        for ii in range(level1_stack.z.size):
            input_mp[ii] = dict({
                'level1': level1_stack.isel(z=ii),
                'noise': noise_stack.isel(z=ii),
                'gmf': gmf,

            })
        
        with multiprocessing.Pool() as pool:
            lmoutmap = pool.map(find_minima_parallel_task, input_mp)

        lmmap = xr.concat(lmoutmap, dim='z')
        lmmap = lmmap.set_index(z=list_L1s0)
        sol = lmmap.unstack(dim='z')
    elif len(list_L1s0) == 1:  # 1d
        length = level1[list_L1s0[0]].size
        lmoutmap = [None] * length
        for ii in range(length):
            sl1 = level1.isel({list_L1s0[0]: ii})
            sn = noise.isel({list_L1s0[0]: ii})
            lmout = cost_function.find_minima(sl1, sn, gmf)
            lmout = lmout.sortby('cost')
            # lmout = ambiguity_removal.solve_ambiguity(lmout, ambiguity)
            lmoutmap[ii] = lmout
        sol = xr.concat(lmoutmap, dim=list_L1s0[0])
    else:  # single pixel
        sol = cost_function.find_minima(level1, noise, gmf)
        sol = sol.sortby('cost')
        # sol = ambiguity_removal.solve_ambiguity(lmout, ambiguity)

    return sol



def compute_current_magnitude_and_direction(level1, level2):
    """
    Compute surface current velocity and direction.

    Compute surface current velocity (m/s) and direction (degrees N)
    from radial surface current (RSC) components measured from two
    orthogonal antennas

    Parameters
    ----------
    level1 : ``xarray.Dataset``
        L1 dataset
    level2 : ``xarray.Dataset``
        L2 dataset

    Returns
    -------
    level2 : ``xarray.Dataset``
        L2 dataset
    level2.CurrentVelocity : ``xarray.DataArray``
        Magnitude of surface current velocity vector (m/s)
    level2.CurrentDirection : ``xarray.DataArray``
        Surface current direction (degrees N) in oceanographic convention

    """
    antenna_angle = np.mod(level1.sel(Antenna='Fore').AntennaAzimuthImage -
                           level1.sel(Antenna='Aft').AntennaAzimuthImage,
                           360)
    level2['CurrentVelocity'] = np.sqrt(
        level1.sel(Antenna='Fore').RadialSurfaceCurrent ** 2
        + level1.sel(Antenna='Aft').RadialSurfaceCurrent ** 2)\
        / np.sin(np.radians(antenna_angle))
    level2.CurrentVelocity.attrs['long_name'] =\
        'Current velocity'
    level2.CurrentVelocity.attrs['description'] =\
        'Total surface current velocity for each pixel in the image'
    level2.CurrentVelocity.attrs['units'] = 'm/s'
    u_1 = level1.sel(Antenna='Fore').RadialSurfaceCurrent\
        * np.sin(np.radians(level1.sel(Antenna='Fore').AntennaAzimuthImage))
    v_1 = level1.sel(Antenna='Fore').RadialSurfaceCurrent\
        * np.cos(np.radians(level1.sel(Antenna='Fore').AntennaAzimuthImage))
    u_2 = level1.sel(Antenna='Aft').RadialSurfaceCurrent\
        * np.sin(np.radians(level1.sel(Antenna='Aft').AntennaAzimuthImage))
    v_2 = level1.sel(Antenna='Aft').RadialSurfaceCurrent\
        * np.cos(np.radians(level1.sel(Antenna='Aft').AntennaAzimuthImage))

    direction = np.degrees(np.arctan2((u_1 + u_2), (v_1 + v_2)))
    ind_pos = direction < 0
    direction_corrected = np.mod(-xr.where(ind_pos,
                                 180 + (180 - np.abs(direction)),
                                 direction
                                 ) + 90,
                                 360)

    level2['CurrentDirection'] = direction_corrected
    level2.CurrentDirection.attrs['long_name'] =\
        'Current direction'
    level2.CurrentDirection.attrs['description'] =\
        'Total surface current direction (oceanographic convention)'\
        ' for each pixel in the image'
    level2.CurrentDirection.attrs['units'] = 'deg'

    level2['CurrentVelocity'] = level2.CurrentVelocity.assign_coords(
        coords={'longitude': level2.longitude, 'latitude': level2.latitude})
    level2['CurrentDirection'] = level2.CurrentDirection.assign_coords(
        coords={'longitude': level2.longitude, 'latitude': level2.latitude})
    return level2
