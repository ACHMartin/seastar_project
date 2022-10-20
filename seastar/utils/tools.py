#!/usr/bin/env python
# coding=utf-8

import numpy as np


def currentVelDir2UV(vel, cdir):
    """
    Compute current vector components from direction and magnitude.

    Parameters
    ----------
    vel : array, float
        Magnitude of current (m/s).
    cdir : array, float
        Direction of current (degrees N) in oceanographic convention.

    Returns
    -------
    u : array, float
        u component (positive East) of current vector.
    v : array, float
        v component (positive North) of current vector..

    """
    z = vel * np.exp(-1j * (cdir - 90) / 180 * np.pi)
    u = z.real
    v = z.imag
    return u, v


def _currentUV2VelDir(u, v):
    """
    Converts U and V currents to velocity and direction (in degrees).

    :param u: velocity
    :type u: ``float``
    :param v: velocity
    :type v: ``float``
    :return: velocity, direction
    :rtype: ``float``, ``float``
    """

    tmp = u + 1j * v
    vel = np.abs(tmp)
    cdir = np.mod(90 - np.angle(tmp, deg=True), 360)

    return vel, cdir


def windSpeedDir2UV(wspd, wdir):
    """
    Compute wind vector components from direction and magnitude.

    Parameters
    ----------
    vel : array, float
        Magnitude of wind (m/s).
    cdir : array, float
        Direction of wind (degrees N) in wind convention.

    Returns
    -------
    u : array, float
        u component (positive East) of wind vector.
    v : array, float
        v component (positive North) of wind vector..

    """
    z = wspd * np.exp(-1j * (wdir + 90) / 180 * np.pi)
    u = z.real  # toward East
    v = z.imag  # toward North
    return u, v


def windUV2SpeedDir(u, v):
    tmp = u + 1j * v
    wspd = np.abs(tmp)
    wdir = np.mod(-90 - np.angle(tmp) * 180 / np.pi, 360)
    return wspd, wdir


def wavenumber2wavelength(wavenumber):
    wavelength = 2 * np.pi / wavenumber
    return wavelength


def compute_relative_wind_direction(windDirection, lookDirection):
    """
    Compute relative wind direction.

    Compute angle between radar beam and wind direction (degrees)
    0 degrees = up-wind
    180 degrees = down-wind
    90, -90 degrees = cross-wind

    Parameters
    ----------
    windDirection : float, xarray.DataArray
        Wind direction in oceanographic convention (degrees N), i.e. the
        direction from where the wind is blowing: e.g., a wind direction of
        0 degrees corresponds to a wind blowing from the North.
    lookDirection : float, xarray.DataArray
        Antenna look direction, either scalar value or array in oceanographic
        convention(degrees N), i.e., the direction to where the antenna is
        looking: e.g., a look direction of 0 degrees is looking North and with
        a wind direction of 0 degrees is looking Up-Wind.


    Returns
    -------
    relative_wind_direction : float, xarray.DataArray
        Angle between radar beam and wind direction (degrees)

    """
    relative_wind_direction = \
        np.abs(
            np.mod(
                windDirection - lookDirection + 180,
                360
            ) - 180
        )
    return relative_wind_direction
