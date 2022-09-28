# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 16:48:48 2022

@author: admartin, dlmccann
"""

import numpy as np
import xarray as xr


def compute_current_magnitude(level2):
    """
    Compute surface current magnitude (m/s) from radial surface velocity
    components measured from two orthogonal antennas

    Parameters
    ----------
    level2 : L2 dataset in netCDF format

    Returns
    -------
    level2 : L2 dataset in netCDF format
    level2.AntennaAngleImage : Angle between two orthogonal antennas (degrees)
    level2.CurrentMagnitude : Magnitude of surface current vector (m/s)

    """
    level2['AntennaAngleImage'] = np.mod(level2.AntennaAzimuthImageFore
                                         - level2.AntennaAzimuthImageAft, 360)
    level2['CurrentMagnitude'] = np.sqrt(level2.RadialSurfaceVelocityFore**2
                                         + level2.RadialSurfaceVelocityAft**2)\
        / np.sin(np.radians(level2.AntennaAngleImage))

    return level2


def compute_current_direction(level2):
    """
    Compute surface current direction from orthogonal radial surface current
    components measured from two antennas.

    Parameters
    ----------
    level2 : L2 dataset in netCDF format

    Returns
    -------
    level2 : L2 dataset in netCDF format
    level2.CurrentDirection : Surface current direction (degrees N) in
    oceanographic convention

    """

    ind_pos = (level2.RadialSurfaceVelocityFore >
               level2.RadialSurfaceVelocityAft) *\
        np.cos(np.radians(level2.AntennaAngleImage))
    temporary_direction = xr.DataArray(np.empty(ind_pos.shape),
                                       coords=[ind_pos.CrossRange,
                                               ind_pos.GroundRange],
                                       dims=('CrossRange', 'GroundRange'))
    temporary_direction = xr.where(ind_pos,
                                   np.degrees(np.arccos(
                                       level2.RadialSurfaceVelocityFore /
                                       level2.CurrentMagnitude)),
                                   - np.degrees(np.arccos(
                                       level2.RadialSurfaceVelocityFore /
                                       level2.CurrentMagnitude)))
    level2['CurrentDirection'] = np.mod(level2.AntennaAzimuthImageFore
                                        + (-temporary_direction), 360)
    return level2


def compute_current_vectors(level2):
    """
    Compute u (East) and v (North) surface velocity vector components (m/s)
    from current magnitude and direction

    Parameters
    ----------
    level2 : L2 dataset in netCDF format

    Returns
    -------
    level2 : L2 dataset in netCDF format
    level2.CurrentVectorUComponent : u (East) surface current vector component
    field (m/s)
    level2.CurrentVectorVComponent : v (North) surface current vector component
    field (m/s)

    """
    z = level2.CurrentMagnitude * np.exp(-1j * (level2.CurrentDirection - 90)
                                         / 180 * np.pi)
    level2['CurrentVectorUComponent'] = np.real(z)  # toward East
    level2['CurrentVectorVComponent'] = np.imag(z)  # toward North
    return level2


def compute_relative_wind_direction(radar_azimuth, wind_direction):
    """
    Compute angle between radar beam and wind direction (degrees)
    0 degrees = up-wind
    180 degrees = down-wind
    90, -90 degrees = cross-wind

    Parameters
    ----------
    radar_azimuth : Radar beam azimuth, either scalar value or array (radians).
    wind_direction : Wind direction in oceanographic convention (degrees N).

    Returns
    -------
    relative_wind_direction : Angle between radar beam and wind direction
    (degrees)

    """
    radar_beam_u_component = np.sin(radar_azimuth)
    radar_beam_v_component = np.cos(radar_azimuth)
    wind_u_component = np.sin(np.radians(wind_direction))
    wind_v_component = np.cos(np.radians(wind_direction))
    relative_wind_direction = np.degrees(np.arccos(
        ((radar_beam_u_component * wind_u_component) +
         (radar_beam_v_component * wind_v_component)) /
        (np.sqrt(radar_beam_u_component ** 2 + radar_beam_v_component ** 2) *
         np.sqrt(wind_u_component ** 2 + wind_v_component ** 2))))
    return relative_wind_direction


def generate_wind_field_from_single_measurement(level2, u10, wind_direction):
    """
    Generate 2D fields of wind velocity u10 (m/s) and direction (degrees N)
    from single measurements

    Parameters
    ----------
    level2 : L2 dataset in netCDF format
    u10 : Wind velocity at 10m above sea surface (m/s)
    wind_direction : Wind direction (degrees N) in wind convention

    Returns
    -------
    level2 : L2 dataset in netCDF format
    level2.u10Image: 2D field of u10 wind velocities (m/s)
    level2.WindDirectionImage : 2D field of wind directions (degrees N)

    """
    wind_direction = np.mod(wind_direction - 180, 360)
    level2['u10Image'] = xr.DataArray(
        np.zeros(level2.RadialSurfaceVelocityFore.shape) + u10,
        coords=[level2.CrossRange, level2.GroundRange],
        dims=('CrossRange', 'GroundRange'))
    level2['WindDirectionImage'] = xr.DataArray(
        np.zeros(level2.RadialSurfaceVelocityFore.shape) + wind_direction,
        coords=[level2.CrossRange, level2.GroundRange],
        dims=('CrossRange', 'GroundRange'))
    return level2


def compute_surface_currents(level2):
    """
    Compute surface current magnitude, direction and vector components

    Parameters
    ----------
    level2 : L2 dataset in netCDF format

    Returns
    -------
    level2 : L2 dataset in netCDF format

    """
    level2 = compute_current_magnitude(level2)
    level2 = compute_current_direction(level2)
    level2 = compute_current_vectors(level2)
    return level2
