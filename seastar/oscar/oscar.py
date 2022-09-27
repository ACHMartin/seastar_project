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
    ds.IntensityAvgComplexMasterSlave : Rolling average multilooking SLC image
    ds.Intensity : Multilook image pixel intensity
    ds.Interferogram : Interferogram (radians)
    ds.IntensityAvgMaster : rolling agerage Master SLC intensity image
    ds.IntensityAvgSlave : rolling average Slave SLC intensity image
    ds.Coherence : Multilook image coherence
    """
    ds['IntensityAvgComplexMasterSlave'] = (ds.SigmaSLCMaster * np.conjugate(ds.SigmaSLCSlave))\
        .rolling(GroundRange=window).mean().rolling(CrossRange=window).mean()
    ds['Intensity'] = np.abs(ds.IntensityAvgComplexMasterSlave)
    ds['Interferogram'] = (
        ['CrossRange', 'GroundRange'],
        np.angle(ds.IntensityAvgComplexMasterSlave, deg=False)
        )
    ds['IntensityAvgMaster'] = (np.abs(ds.SigmaSLCMaster) ** 2)\
        .rolling(GroundRange=window).mean().rolling(CrossRange=window).mean()
    ds['IntensityAvgSlave'] = (np.abs(ds.SigmaSLCSlave) ** 2)\
        .rolling(GroundRange=window).mean().rolling(CrossRange=window).mean()
    ds['Coherence'] = ds.Intensity / np.sqrt(ds.IntensityAvgMaster
                                             * ds.IntensityAvgSlave)
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
    ds.IncidenceAngleImage : Incidence angle between radar beam and
    nadir (radians) for each pixel

    """
    X, Y = np.meshgrid(ds.CrossRange, ds.GroundRange, indexing='ij')
    ds['IncidenceAngleImage'] = np.arctan(Y / ds.OrbHeightImage)

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
        ds['TimeLag'] = (ds.Baseline / 2 * ds.MeanForwardVelocity)
    return ds


def compute_radial_surface_velocity(ds):
    """
    Compute radial surface velocity from SAR interferogram and time lag
    between antenna pairs

    Parameters
    ----------
    ds : OSCAR SAR dataset in netCDF format.

    Returns
    -------
    ds : OSCAR SAR dataset in netCDF format.
    ds.RadialSuraceVelocity : Surface velocity (m/s) along a radar beam radial
    """

    ds['RadialSurfaceVelocity'] = ds.Interferogram /\
        (ds.TimeLag * ds.CentralWavenumber
         * np.sin(ds.IncidenceAngleImage))

    return ds


def init_level2(dsa, dsf, dsm):
    """
    Initialise level2 dataset with

    Parameters
    ----------
    dsa : TYPE
        DESCRIPTION.
    dsf : TYPE
        DESCRIPTION.
    dsm : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # Fore antenna variables
    dsf = add_central_electromagnetic_wavenumber(dsf)
    dsf = compute_SLC_Master_Slave(dsf)
    dsf = compute_multilooking_Master_Slave(dsf, window=7)
    dsf = add_antenna_baseline(dsf)
    dsf = compute_SLC_Master_Slave(dsf)
    dsf = compute_multilooking_Master_Slave(dsf, window=7)
    dsf = compute_incidence_angle(dsf)
    dsf = compute_antenna_azimuth_direction(dsf, antenna='fore')
    dsf = compute_time_lag_Master_Slave(dsf)
    dsf = compute_radial_surface_velocity(dsf)

    # Aft antenna variables
    dsa = add_central_electromagnetic_wavenumber(dsa)
    dsa = compute_SLC_Master_Slave(dsa)
    dsa = compute_multilooking_Master_Slave(dsa, window=7)
    dsa = add_antenna_baseline(dsa)
    dsa = compute_SLC_Master_Slave(dsa)
    dsa = compute_multilooking_Master_Slave(dsa, window=7)
    dsa = compute_incidence_angle(dsa)
    dsa = compute_antenna_azimuth_direction(dsa, antenna='aft')
    dsa = compute_time_lag_Master_Slave(dsa)
    dsa = compute_radial_surface_velocity(dsa)

    level2 = xr.Dataset()

    # Write level2 attributes
    level2.attrs['Title'] = dsf.attrs['Title']
    # Add data arrays to dataset
    level2['RadialSurfaceVelocityFore'] = dsf.RadialSurfaceVelocity
    level2['AntennaAzimuthImageFore'] = dsf.AntennaAzimuthImage
    level2['IncidenceAngleImageFore'] = dsf.IncidenceAngleImage

    level2['RadialSurfaceVelocityAft'] = dsa.RadialSurfaceVelocity
    level2['AntennaAzimuthImageAft'] = dsa.AntennaAzimuthImage
    level2['IncidenceAngleImageAft'] = dsa.IncidenceAngleImage
    return level2
