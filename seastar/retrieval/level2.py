# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 16:48:48 2022

@author: admartin, dlmccann
"""

import numpy as np
import xarray as xr
import seastar


def compute_current_magnitude_and_direction(level1, level2):
    """
    Compute surface current magnitude and direction.

    Compute surface current magnitude (m/s) and direction (degrees N)
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
    level2 : xarray.Dataset
        L2 dataset
    level2.CurrentMagnitude : xarray.DataArray
        Magnitude of surface current vector (m/s)
    level2.CurrentDirection : xarray.DataArray
        Surface current direction (degrees N) in oceanographic convention

    """
    #antenna_angle = np.mod(level1.sel(Antenna='Fore').AntennaAzimuthImage -
    #                       level1.sel(Antenna='Aft').AntennaAzimuthImage,
    #                       360)
    #level2['CurrentMagnitude'] = np.sqrt(
    #    level1.sel(Antenna='Fore').RadialSurfaceCurrent ** 2
    #    + level1.sel(Antenna='Aft').RadialSurfaceCurrent ** 2)\
    #    / np.sin(np.radians(antenna_angle))

    #ind_pos = (level1.sel(Antenna='Fore').RadialSurfaceCurrent >
    #           level1.sel(Antenna='Aft').RadialSurfaceCurrent) *\
    #    np.cos(np.radians(antenna_angle))
    # temporary_direction = xr.DataArray(np.empty(ind_pos.shape),
    #                                    coords=[level2.CrossRange,
    #                                            level2.GroundRange],
    #                                    dims=('CrossRange', 'GroundRange'))
    #temporary_direction = xr.where(ind_pos,
    #                               np.degrees(np.arccos(
    #                                   level1.sel(Antenna='Fore').RadialSurfaceCurrent /
    #                                   level2.CurrentMagnitude)),
    #                               - np.degrees(np.arccos(
    #                                   level1.sel(Antenna='Fore').RadialSurfaceCurrent /
    #                                   level2.CurrentMagnitude)))
    #level2['CurrentDirection'] = np.mod(level1.sel(Antenna='Fore').AntennaAzimuthImage
    #                                    + (-temporary_direction), 360)
    antenna_angle = np.mod(level1.sel(Antenna='Fore').AntennaAzimuthImage -
                           level1.sel(Antenna='Aft').AntennaAzimuthImage,
                           360)
    level2['CurrentMagnitude'] = np.sqrt(
        level1.sel(Antenna='Fore').RadialSurfaceCurrent ** 2
        + level1.sel(Antenna='Aft').RadialSurfaceCurrent ** 2)\
        / np.sin(np.radians(antenna_angle))
    level2.CurrentMagnitude.attrs['long_name'] =\
        'Total surface current magnitude for each pixel in the image'

    level2.CurrentMagnitude.attrs['units'] = '[m/s]'
    u_1 = level1.sel(Antenna='Fore').RadialSurfaceCurrent\
        * np.cos(np.radians(level1.sel(Antenna='Fore').AntennaAzimuthImage))
    v_1 = level1.sel(Antenna='Fore').RadialSurfaceCurrent\
        * np.sin(np.radians(level1.sel(Antenna='Fore').AntennaAzimuthImage))
    u_2 = level1.sel(Antenna='Aft').RadialSurfaceCurrent\
        * np.cos(np.radians(level1.sel(Antenna='Aft').AntennaAzimuthImage))
    v_2 = level1.sel(Antenna='Aft').RadialSurfaceCurrent\
        * np.sin(np.radians(level1.sel(Antenna='Aft').AntennaAzimuthImage))

    direction = np.degrees(np.arctan2((u_1 + u_2), (v_1 + v_2)))
    ind_pos = direction < 0

    direction_corrected = xr.where(ind_pos,
                                   180 + (180 - np.abs(direction)),
                                   direction
                                   )

    level2['CurrentDirection'] = direction_corrected
    level2.CurrentDirection.attrs['long_name'] =\
        'Total surface current direction (oceanographic convention)'\
        ' for each pixel in the image'
    level2.CurrentDirection.attrs['units'] = '[degrees]'

    level2['CurrentMagnitude'] = level2.CurrentMagnitude.assign_coords(
        coords={'longitude': level2.longitude, 'latitude': level2.latitude})
    level2['CurrentDirection'] = level2.CurrentDirection.assign_coords(
        coords={'longitude': level2.longitude, 'latitude': level2.latitude})
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
