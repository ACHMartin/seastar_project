#!/usr/bin/env python
# coding=utf-8

import numpy as np


def currentVelDir2UV(vel, cdir):
    z = vel * np.exp(-1j * (cdir - 90) / 180 * np.pi)
    u = z.real  # toward East
    v = z.imag  # toward North
    return u, v


def currentUV2VelDir(u, v):
    """some kind of function

    :param u: velocity
    :type u: float
    :param v: velocity
    :type v: float
    :return: velocity, direction,
    :rtype: float, float
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
