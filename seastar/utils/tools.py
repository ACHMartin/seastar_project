#!/usr/bin/env python
# coding=utf-8

import numpy as np

def currentVelDir2UV(vel, cdir):
    z = vel * np.exp(-1j * (cdir - 90) / 180 * np.pi)
    u = z.real  # toward East
    v = z.imag  # toward North
    return u, v


def _currentUV2VelDir(u, v):
    """Converts U and V currents to velocity and direction (in degrees).

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
    Compute the relative wind direction between the antenna lookDirection and the windDirection.
    Assuming the same meteorological convention for lookDir and windDir.
    :param windDirection: Direction from where the wind is blowing. North wind -> 0°
    :param lookDirection: Direction in which the radar is looking. Looking toward the North -> 0°, in this case
    facing the wind -> upwind
    :return: relative_wind_direction between 0° and 180°. 0° for upwind; 90° crosswind; 180° downwind
    """
    relative_wind_direction = \
        np.abs(
            np.mod(
                windDirection - lookDirection + 180,
                360
            ) - 180
        )
    return relative_wind_direction