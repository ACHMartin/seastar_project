# -*- coding: utf-8 -*-
"""Functions to compute Level-2 (L2) products."""
import multiprocessing
import numpy as np
import xarray as xr
import seastar
from seastar.retrieval import cost_function, ambiguity_removal
# from seastar.utils.tools import da2py
from tqdm import tqdm
from typing import Optional

from _version import __version__
from _logger import logger

import sys

# import pdb # pdb.set_trace() # where we want to start to debug


def find_minima_parallel_task(element):
    """
    Parallel processing task.

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
    Compute ocean surface and earth relative WIND and CURRENT magnitude and direction
    by minimisation of a cost function.

    Compute Ocean Surface Vector Wind (OSVW) and Earth Relative (ERW) in (m/s) and
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
    level2.EarthRelativeWindSpeed : ``xarray.DataArray``
        EarthRelative Wind Speed (m/s)
    level2.EarthRelativeWindDirection : ``xarray.DataArray``
        EarthRelative Wind Direction (degrees N) in meteorologic convention
        (coming from)
    level2.OceanSurfaceWindSpeed : ``xarray.DataArray``
        Ocean Surface Wind Speed (m/s)
    level2.OceanSurfaceWindDirection : ``xarray.DataArray``
        Ocean Surface Wind Direction (degrees N) in meteorologic convention
        (coming from)
    """

    lmout = run_find_minima(level1, noise, gmf)
    sol = ambiguity_removal.solve_ambiguity(lmout, ambiguity)

    level2 = sol2level2(sol)

    return level2

def sol2level2(sol):
    """
    Convert solution.x into EarthRelativeWindU, EarthRelativeWindV, EarthRelativeWindSpeed, EarthRelativeWindDirection,
    same for OceanSurfaceWindU, V, Speed, Direction and
     CurrentU, V, Velocity, Direction

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
    level2['EarthRelativeWindU'] = level2.x.sel(x_variables='u')
    level2['EarthRelativeWindV'] = level2.x.sel(x_variables='v')

    level2 = seastar.utils.tools.EarthRelativeUV2all(level2)

    return level2


def run_find_minima(level1, noise, gmf, serial=False):
    """
    Run find minima on xD dimension DataSet.

    Parameters
    ----------
    level1 : ``xarray.Dataset``
        L1 observables noisy dataset (Sigma0, RSV, geometry)
    noise : ``xarray.Dataset``
        Defined noise with data_vars `Sigma0`, `RSV` on the same grid as ``level1``
    gmf : ``dict``
        Geophysical Model Function
    Returns
    -------
    sol : ``xarray.Dataset``
        x dimension dataset of find_minima output containing among other
        `.x = [u,v,c_u,c_v]` and `.cost`
        with dimension along `Ambiguities` of size=4 by construction.
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
        if serial:
            lmoutmap = map(find_minima_parallel_task, input_mp)
        else:
            with multiprocessing.Pool() as pool:
                results = []
                for result in tqdm(pool.imap(find_minima_parallel_task, input_mp), total=len(input_mp)):
                    results.append(result)
            lmoutmap = results

        lmmap = xr.concat(lmoutmap, dim='z')
        lmmap = lmmap.set_index(z=list_L1s0)
        sol = lmmap.unstack(dim='z')
    elif len(list_L1s0) == 1:  # 1d
        dim_name = list_L1s0[0]
        dim_length = len(level1[list_L1s0[0]])
        input_mp = [None] * dim_length
        for ii in range(dim_length):
            input_mp[ii] = dict({
                'level1': level1.isel({dim_name: ii}),
                'noise': noise.isel({dim_name: ii}),
                'gmf': gmf,
            })
        if serial:
            lmoutmap = map(find_minima_parallel_task, input_mp)
        else:
            with multiprocessing.Pool() as pool:
                lmoutmap = pool.map(find_minima_parallel_task, input_mp)

        sol = xr.concat(lmoutmap, dim=dim_name)
        sol = sol.set_index({dim_name: dim_name})
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
    direction_corrected = np.mod(xr.where(ind_pos,
                                 180 + (180 - np.abs(direction)),
                                 direction
                                 ),
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


def sequential_current_retrieval(level1, dict_env, gmf):
    """
    Sequential current retrieval (SCR) processing.
    Compute surface current velocity and direction from radial surface current
    (RSC) components measured from level1 B or C products.

    Parameters
    ----------
    level1 : ``xarray.Dataset``
        L1 dataset
    dict_env ``xarray.Dataset``
        Dictionary containing the environment information needed for SCR inversion.        
    gmf ``str``
        Name of the Doppler GMF to use for the retrieval.

    Returns
    -------
    level2 : ``xarray.Dataset``
        L2 dataset.
    """
    
    level1 = level1.copy()

    # Compute auxiliary data
    logger.info("compute auxiliary data")
    aux = seastar.performance.scene_generation.generate_constant_env_field(level1.IncidenceAngleImage, dict_env)

    #Compute RSC
    logger.info("compute RSC")
    rsc = [seastar.oscar.level1.compute_radial_surface_current(level1.sel(Antenna=a), aux.sel(Antenna=a), gmf = gmf) for a in level1.Antenna] 
    level1['RadialSurfaceCurrent'] = xr.concat(rsc, 'Antenna', join='outer') 

    #L2 Processing
    #Initialise l2 dataset
    logger.info("Initialise l2 dataset")
    level2=init_level2(level1)

    #Compute current magnitude and direction
    logger.info("Compute current magnitude and direction")
    level2=seastar.retrieval.level2.compute_current_magnitude_and_direction(level1, level2)

    #Compute current vectors
    logger.info("Compute current vectors")
    level2['CurrentU'], level2['CurrentV'] = \
    seastar.utils.tools.currentVelDir2UV(level2['CurrentVelocity'], level2['CurrentDirection'])
    
    return level2


def init_level2(level1):
    """
    Initialise level2 dataset.

    Parameters
    ----------
    dsa : xarray.Dataset
        OSCAR SAR dataset for the aft antenna pair
    dsf : xarray.Dataset
        OSCAR SAR dataset for the fore antenna pair
    dsm : xarray.Dataset
        OSCAR SAR dataset for the mid antenna

    Returns
    -------
    level2 : xarray.Dataset
        OSCAR SAR L2 processing dataset
    level2.RadialSurfaceVelocity : xarray.DataArray
        Radial surface velocities (m/s) for the Fore and Aft antennas
        with corresponding dimension 'Antenna' and Coords ['Fore','Aft']
    level2.LookDirection : xarray.DataArray
        Antenna look direction (degrees N) for the Fore and Aft antennas
        with corresponding dimension 'Antenna' and Coords ['Fore','Aft']

    """
    level2 = xr.Dataset()
    level2.attrs = level1.attrs.copy()
#    level2.coords['longitude'] = level1.sel(Antenna='Fore').LonImage
#    level2.coords['latitude'] = level1.sel(Antenna='Fore').LatImage
    level2.coords['longitude'] = level1.longitude
    level2.coords['latitude'] = level1.latitude
#    level2 = level2.drop('Antenna')

    return level2


def is_valid_gmf_dict(gmf_dict):
    """
    Check the gmf dict in OSCAR L2 processing chain. Shall be:
    gmf={
        'nrcs': {'name': 'nscat4ds'},
        'doppler': {'name': 'mouche12' or 'yurovsky19'},
    }

    Parameters
    ----------
        gmf_dict : ``dict``
            Dictionnary containing the information about the NRCS GMF and Doppler GMF

    Returns:
        bool
            True if the format of the dictionnaire is correct.
    """
    # Clés et valeurs valides attendues
    valid_keys = {"nrcs", "doppler"}
    valid_values = {
        "nrcs": {"nscat4ds"},
        "doppler": {"mouche12", "yurovsky19", "oscar20220522T11-18_v20250318"}
    }

    # Vérifie que toutes les clés sont valides
    for key in gmf_dict:
        if key not in valid_keys:
            logger.error(f"Invalid key: {key}. Expected keys: {valid_keys}. The code will crash.")
            raise ValueError(f"Invalid key: {key}. Expected keys: {valid_keys}")

        # Vérifie que la clé "name" existe pour chaque bloc
        if "name" not in gmf_dict[key]:
            logger.error(f"Missing 'name' field for key: {key}. The code will crash.")
            raise ValueError(f"Missing 'name' field for key: {key}")

        # Vérifie que la valeur est bien dans la liste autorisée
        value = gmf_dict[key]["name"]
        if value not in valid_values[key]:
            logger.error(f"Invalid value '{value}' for key '{key}'. Expected values: {valid_values[key]}. The code will crash.")
            raise ValueError(f"Invalid value '{value}' for key '{key}'. Expected values: {valid_values[key]}")

    return True



