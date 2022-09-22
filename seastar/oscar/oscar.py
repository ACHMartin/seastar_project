# -*- coding: utf-8 -*-
"""
Functions to calculate varirous L2 data products for the OSCAR 3-beam ATI
SAR instrument.

Created on Fri Sep 16 13:48:48 2022

@author: admartin, dlmccann
"""

import numpy as np
import xarray as xr
import scipy as sp


def add_antenna_baseline(ds, baseline=0.2):
    """
    Add antenna baseline to dataset if not already present

    Parameters
    ----------
    ds : OSCAR SAR dataset in netCDF format
    baseline : Antenna baseline (m), default = 0.2m
    Returns
    -------
    ds : OSCAR SAR dataset in netCDF format
    ds.Baseline : Antenna baseline distance (m)

    """

    if 'Baseline' not in ds.data_vars:
        ds['Baseline'] = baseline
    return ds


def compute_SLC_Master_Slave(ds):
    """
    Calculate SLC master/slave complex images and add to dataset

    Parameters
    ----------
    ds : OSCAR SAR dataset in netCDF format
    ds.SigmaImageSingleLookRealPart : Real part of SLC image
    ds.SigmaImageSingleLookImaginaryPart: Imaginary part of SLC image

    Returns
    -------
    ds : OSCAR SAR dataset in netCDF format
    ds.SigmaSLCMaster: Master SLC image
    ds.SigmaSLCSlave: Slave SLC image

    """
    ds['SigmaSLCMaster'] = (ds.SigmaImageSingleLookRealPart + 1j *
                            ds.SigmaImageSingleLookImaginaryPart)
    if 'SigmaImageSingleLookRealPartSlave' in ds.data_vars:
        ds['SigmaSLCSlave'] = (ds.SigmaImageSingleLookRealPartSlave + 1j *
                               ds.SigmaImageSingleLookImaginaryPartSlave)
    return ds


def add_central_electromagnetic_wavenumber(ds):
    """
    Calculate wavenumber of central electromagnetic frequency and add to
    dataset

    Parameters
    ----------
    ds : OSCAR SAR dataset in netCDF format.
    ds.CentralFreq : Central radar frequency (Hz)

    Returns
    -------
    ds : OSCAR SAR dataset in netCDF format.
    ds.CentralWavenumber : Central radar wavenumber (rad / m)

    """

    ds['CentralWavenumber'] = 2 * np.pi * ds.CentralFreq / sp.constants.c
    return ds


def compute_multilooking_Master_Slave(ds, window=3):
    """
    Calculate multilooking Master/Slave L2 image products and add to dataset

    Parameters
    ----------
    ds : OSCAR SAR dataset in netCDF format.
    window : Integer averaging window size. The default is 3.

    Returns
    -------
    ds : OSCAR SAR dataset in netCDF format.
    ds.IAvg : Rolling average multilooking SLC image
    ds.Amplitude : Absolute multilooking amplitude
    ds.Interferogram : Interferogram (radians)
    ds.MasterAvg : rolling agerage Master SLC image
    ds.SlaveAvg : rolling average Slave SLC image
    ds.Coherence : Multilook image coherence
    """
    ds['IAvg'] = (ds.SigmaSLCMaster * np.conjugate(ds.SigmaSLCSlave))\
        .rolling(GroundRange=window).mean().rolling(CrossRange=window).mean()
    ds['Amplitude'] = np.abs(ds.IAvg)
    ds['Interferogram'] = (
        ['CrossRange', 'GroundRange'],
        np.angle(ds.IAvg, deg=False)
        )
    ds['MasterAvg'] = (np.abs(ds.SigmaSLCMaster) ** 2)\
        .rolling(GroundRange=window).mean().rolling(CrossRange=window).mean()
    ds['SlaveAvg'] = (np.abs(ds.SigmaSLCSlave) ** 2)\
        .rolling(GroundRange=window).mean().rolling(CrossRange=window).mean()
    ds['Coherence'] = ds.Amplitude / np.sqrt(ds['MasterAvg'] * ds['SlaveAvg'])
    return ds


def compute_incidence_angle(ds):
    """
    Calculate incidence angle between radar beam and sea surface and add to
    dataset

    Parameters
    ----------
    ds : OSCAR SAR dataset in netCDF format.

    Returns
    -------
    ds : OSCAR SAR dataset in netCDF format.
    ds.IncidenceAngle : Incidence angle between radar beam and
    sea surface (radians)

    """
    X, Y = np.meshgrid(ds.CrossRange, ds.GroundRange, indexing='ij')
    ds['IncidenceAngle'] = np.arctan(ds.OrbHeightImage / Y)

    return ds


def compute_antenna_azimuth_direction(ds, antenna):
    """
    Calculate antenna azimuth relative to North in degrees and add to dataset

    Parameters
    ----------
    ds : OSCAR SAR dataset in netCDF format.
    antenna : 'fore' Fore beam pair, 'aft' Aft beam pair


    Returns
    -------
    ds : OSCAR SAR dataset in netCDF format.
    ds.OrbitHeadingImage : Aircraft heading of each image pixel
    (degrees from North)
    ds.AntennaAzimuthImage : Antenna beam azimuth of each image pixel
    (degrees from North)

    """
    m, n = ds.OrbTimeImage.shape
    head = np.interp(np.reshape(ds.OrbTimeImage, (1, m * n)), ds.GPSTime,
                     ds.OrbitHeading)
    head = np.reshape(head, (m, n))
    ds['OrbitHeadingImage'] = xr.DataArray.copy(ds.OrbTimeImage, data=head)
    ds.OrbitHeadingImage.attrs['long_name'] =\
        'Heading (degrees N) of the airfraft for each pixel in the image'
    ds.OrbitHeadingImage.attrs['units'] = '[degrees]'

    # Assuming antennas pointing to port. If pointing to starboard
    # then change sign
    if antenna == 'fore':
        ds['AntennaAzimuthImage'] = np.mod(ds.OrbitHeadingImage - 45, 360)
    if antenna == 'aft':
        ds['AntennaAzimuthImage'] = np.mod(ds.OrbitHeadingImage - 135, 360)

    return ds


def compute_time_lag_Master_Slave(ds, options='from_SAR_time'):
    """
    Compute time lag tau between Master/Slave images and add to dataset

    Parameters
    ----------
    ds : OSCAR SAR dataset in netCDF format.
    options : Time lag computation method. The default is 'from_SAR_time'.

    Returns
    -------
    ds : OSCAR SAR dataset in netCDF format.
    ds.TimeLag : Time lag tau between Master/Slave images (s)

    """
    if options == 'from_SAR_time':
        ds['TimeLag'] = (ds.OrbTimeImage - ds.OrbTimeImageSlave)
    if options == 'from_aircraft_velocity':
        ds['TimeLag'] = (ds.Baseline / ds.MeanForwardVelocity)
    return ds


def compute_radial_surface_velocity(ds, options='from_SAR_time'):
    """
    Compute radial surface velocity from SAR interferogram and time lag
    between antenna pairs, using either the time lag calculated directly from
    timing information (option = 'from_SAR_time') or calculated using the
    aircraft's velocity (option = 'from_aircraft_velocity)

    Parameters
    ----------
    ds : OSCAR SAR dataset in netCDF format.
    options : Time lag computation method. The default is 'from_SAR_time'.

    Returns
    -------
    ds : OSCAR SAR dataset in netCDF format.
    ds.RadialSuraceVelocity : Surface velocity (m/s) along a radar beam radial
    """
    if options == 'from_SAR_time':
        ds['RadialSurfaceVelocity'] = ds.Interferogram /\
            (ds.TimeLag * ds.CentralWavenumber * ds.Baseline)
    if options == 'from_aircraft_velocity':
        ds['RadialSurfaceVelocity'] = - (ds.MeanForwardVelocity /
                                         (ds.CentralWavenumber * ds.Baseline))\
            * (ds.Interferogram / np.sin(ds.IncidenceAngle))
    return ds


#def init_level2(ds):
    
    
    
    
    
   # return level2