# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 16:48:48 2022

@author: admartin, dlmccann
"""

import numpy as np
import xarray as xr
import seastar


def compute_radial_surface_current(level1, level2, aux, gmf='mouche12'):
    """
    Compute radial surface current (RSC).

    Compute radial surface current (RSC) from radial surface velocity (RSV)
    and the wind artifact surface velocity (WASV) from:
        RSC = RSV - WASV

    Parameters
    ----------
    level1 : xarray.Dataset
        L1 dataset
    level2 : xarray.Dataset
        L2 dataset
    aux : xarray.Dataset
        Dataset containing geophysical wind data
    gmf : str, optional
        Choice of geophysical model function to compute the WASV.
        The default is 'mouche12'.

    Returns
    -------
    level2 : xarray.Dataset
        L2 dataset

    """
    dswasv_f = seastar.gmfs.doppler.compute_wasv(level1.sel(Antenna='Fore'),
                                                 aux,
                                                 gmf)
    dswasv_a = seastar.gmfs.doppler.compute_wasv(level1.sel(Antenna='Aft'),
                                                 aux,
                                                 gmf)

    level2['WASV'] = xr.concat(
        [dswasv_f, dswasv_a],
        'Antenna',
        join='outer',
    )

    level2['RadialSurfaceCurrent'] = xr.concat(
        [level1.RadialSurfaceVelocity.sel(Antenna='Fore') - dswasv_f,
         level1.RadialSurfaceVelocity.sel(Antenna='Aft') - dswasv_a],
        'Antenna', join='outer')
    level2['RadialSurfaceCurrent'] = level2.RadialSurfaceCurrent.assign_coords(
        Antenna=('Antenna', ['Fore', 'Aft']))

    return level2



def compute_current_magnitude_and_direction(level1, level2):
    """
    Compute surface current magnitude and direction.

    Compute surface current magnitude (m/s) and direction (degrees N)
    from radial surface current (RSC) components measured from two
    orthogonal antennas

    Parameters
    ----------
    level2 : xarray.Dataset
        L2 dataset
    dsf : xarray.Dataset
        Fore antenna ATI SAR dataset
    dsa : xarray.Dataset
        Aft antenna ATI SAR dataset

    Returns
    -------
    level2 : xarray.Dataset
        L2 dataset
    level2.CurrentMagnitude : xarray.DataArray
        Magnitude of surface current vector (m/s)
    level2.CurrentDirection : xarray.DataArray
        Surface current direction (degrees N) in oceanographic convention

    """
    antenna_angle = np.mod(level1.sel(Antenna='Fore').AntennaAzimuthImage -
                           level1.sel(Antenna='Aft').AntennaAzimuthImage,
                           360)
    level2['CurrentMagnitude'] = np.sqrt(
        level2.sel(Antenna='Fore').RadialSurfaceCurrent ** 2
        + level2.sel(Antenna='Aft').RadialSurfaceCurrent ** 2)\
        / np.sin(np.radians(antenna_angle))

    ind_pos = (level2.sel(Antenna='Fore').RadialSurfaceCurrent >
               level2.sel(Antenna='Aft').RadialSurfaceCurrent) *\
        np.cos(np.radians(antenna_angle))
    # temporary_direction = xr.DataArray(np.empty(ind_pos.shape),
    #                                    coords=[level2.CrossRange,
    #                                            level2.GroundRange],
    #                                    dims=('CrossRange', 'GroundRange'))
    temporary_direction = xr.where(ind_pos,
                                   np.degrees(np.arccos(
                                       level2.sel(Antenna='Fore').RadialSurfaceCurrent /
                                       level2.CurrentMagnitude)),
                                   - np.degrees(np.arccos(
                                       level2.sel(Antenna='Fore').RadialSurfaceCurrent /
                                       level2.CurrentMagnitude)))
    level2['CurrentDirection'] = np.mod(level1.sel(Antenna='Fore').AntennaAzimuthImage
                                        + (-temporary_direction), 360)

    return level2


def generate_wind_field_from_single_measurement(aux, u10, wind_direction, ds):
    """
    Generate 2D fields of wind velocity and direction.

    Generate 2D fields of wind velocity u10 (m/s) and direction (degrees) in
    wind convention from single observations.

    Parameters
    ----------
    level2 : xarray.Dataset
        L2 dataset
    u10 : Wind velocity at 10m above sea surface (m/s)
    wind_direction : Wind direction (degrees N) in wind convention

    Returns
    -------
    level2 : xarray.Dataset
        L2 dataset
    level2.u10Image: 2D field of u10 wind velocities (m/s)
    level2.WindDirectionImage : 2D field of wind directions (degrees N)

    """
    wind_direction = np.mod(wind_direction - 180, 360)
    u10Image = xr.DataArray(
        np.zeros((ds.CrossRange.shape[0], ds.GroundRange.shape[0]))
        + u10,
        coords=[ds.CrossRange, ds.GroundRange],
        dims=('CrossRange', 'GroundRange'))
    WindDirectionImage = xr.DataArray(
        np.zeros((ds.CrossRange.shape[0], ds.GroundRange.shape[0]))
        + wind_direction,
        coords=[ds.CrossRange, ds.GroundRange],
        dims=('CrossRange', 'GroundRange'))
    return u10Image, WindDirectionImage
