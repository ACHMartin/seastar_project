# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 13:48:48 2022

@author: admartin
"""

import numpy as np

def compute_SLC_Master_Slave(ds):
    """

    :param ds:
    :return:
    """
    ds['SigmaSLCMaster'] = (ds.SigmaImageSingleLookRealPart + 1j * ds.SigmaImageSingleLookImaginaryPart)
    if 'SigmaImageSingleLookRealPartSlave' in ds.data_vars:
        ds['SigmaSLCSlave'] = (ds.SigmaImageSingleLookRealPartSlave + 1j * ds.SigmaImageSingleLookImaginaryPartSlave)
    return ds

def add_central_electromagnetic_wavenumber(ds):
    # k_e = 2 * np.pi * f_e / c
    ds['CentralWavenumber'] =
    return ds

def compute_multilooking_Master_Slave(ds, window=3):
    """

    :param ds:
    :param window: default 3 pixels multilooking; 5, 7 are good values as well
    :return:
    """
    ds['I_avg'] = (ds.SigmaSLCMaster * np.conjugate(ds.SigmaSLCSlave))\
                    .rolling( GroundRange=window ).mean().rolling( CrossRange=window ).mean()
    ds['Amplitude'] = np.abs(ds.I_avg)
    ds['Interferogram'] = (
        ['CrossRange', 'GroundRange'],
        np.angle( ds.I_avg, deg=False )
    )
    ds['Master_avg'] = (np.abs(ds.SigmaSLCMaster) ** 2)\
                        .rolling( GroundRange=window ).mean().rolling( CrossRange=window ).mean()
    ds['Slave_avg'] = ( np.abs(ds.SigmaSLCSlave) ** 2 )\
                        .rolling( GroundRange=window ).mean().rolling( CrossRange=window ).mean()
    ds['Coherence'] = ds.Amplitude / np.sqrt( ds['Master_avg'] * ds['Slave_avg'] )
    return ds

def compute_incidence_angle(ds):
    """
    In degree
    :param ds:
    :return:
    """
    ds[IncidenceAngle] = 1#
    return ds


def compute_antenna_azimuth_direction(ds):
    """
    Relative to the North in degree

    :param ds:
    :return:
    """
    ds[AzimuthAngle]
    return ds

def compute_time_lag_Master_Slave(ds, options):
    if options = 'from_SAR_time':
        ds['TimeLag'] = (ds.OrbTimeImage - ds.OrbTimeImageSlave)
    if options = 'from_aircraf_velocity'
        # 
    return ds

def init_level2(ds):
    return level2