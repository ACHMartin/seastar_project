# -*- coding: utf-8 -*-
"""
Functions to calculate varirous L2 data products for the OSCAR instrument.

Created on Fri Sep 16 13:48:48 2022

@author: admartin, dlmccann
"""

import numpy as np
import xarray as xr
import scipy as sp


def add_antenna_baseline(ds, baseline=0.2):
    """
    Add antenna baseline to dataset if not already present.

    Parameters
    ----------
    ds : xarray.Dataset
        OSCAR SAR dataset
    baseline : float
        Antenna baseline (m), default = 0.2m

    Returns
    -------
    ds : xarray.Dataset
        OSCAR SAR dataset in netCDF format
    ds.Baseline : float
        Antenna baseline distance (m)

    """
    if 'Baseline' not in ds.data_vars:
        ds['Baseline'] = baseline
    return ds


def compute_SLC_Master_Slave(ds):
    """
    Calculate SLC master/slave complex images.

    Parameters
    ----------
    ds : xarray.Dataset
        OSCAR SAR dataset
    ds.SigmaImageSingleLookRealPart : xarray.DataArray
        Real part of SLC image
    ds.SigmaImageSingleLookImaginaryPart: xarray.DataArray
        Imaginary part of SLC image

    Returns
    -------
    ds : xarray.Dataset
        OSCAR SAR dataset
    ds.SigmaSLCMaster: xarray.DataArray
        Master SLC image
    ds.SigmaSLCSlave: xarray.DataArray
        Slave SLC image

    """
    ds['SigmaSLCMaster'] = (ds.SigmaImageSingleLookRealPart + 1j *
                            ds.SigmaImageSingleLookImaginaryPart)
    if 'SigmaImageSingleLookRealPartSlave' in ds.data_vars:
        ds['SigmaSLCSlave'] = (ds.SigmaImageSingleLookRealPartSlave + 1j *
                               ds.SigmaImageSingleLookImaginaryPartSlave)
    return ds


def add_central_electromagnetic_wavenumber(ds):
    """
    Calculate wavenumber of central electromagnetic frequency.

    Parameters
    ----------
    ds : xarray.Dataset
        OSCAR SAR dataset
    ds.CentralFreq : xarray.DataArray
        Central radar frequency (Hz)

    Returns
    -------
    ds : xarray.Dataset
        OSCAR SAR dataset
    ds.CentralWavenumber : xarray.DataArray
        Central radar wavenumber (rad / m)

    """
    ds['CentralWavenumber'] = 2 * np.pi * ds.CentralFreq / sp.constants.c
    return ds


def compute_multilooking_Master_Slave(ds, window=3):
    """
    Calculate multilooking Master/Slave L2 image products.

    Parameters
    ----------
    ds : xarray.Dataset
        OSCAR SAR dataset
    window : int
        Integer averaging window size. The default is 3.

    Returns
    -------
    ds : xarray.Dataset
        OSCAR SAR dataset
    ds.IntensityAvgComplexMasterSlave : xarray.DataArray
        Rolling average multilooking SLC image
    ds.Intensity : xarray.DataArray
        Multilook image pixel intensity
    ds.Interferogram : xarray.DataArray
        Interferogram (radians)
    ds.IntensityAvgMaster : xarray.DataArray
        rolling agerage Master SLC intensity image
    ds.IntensityAvgSlave : xarray.DataArray
        rolling average Slave SLC intensity image
    ds.Coherence : xarray.DataArray
        Multilook image coherence
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
    Calculate incidence angle between radar beam and sea surface.

    Parameters
    ----------
    ds : xarray.Dataset
        OSCAR SAR dataset

    Returns
    -------
    ds : xarray.Dataset
        OSCAR SAR dataset
    ds.IncidenceAngleImage : Incidence angle between radar beam and
    nadir (radians) for each pixel

    """
    X, Y = np.meshgrid(ds.CrossRange, ds.GroundRange, indexing='ij')
    ds['IncidenceAngleImage'] = np.arctan(Y / ds.OrbHeightImage)

    return ds


def compute_antenna_azimuth_direction(ds, antenna):
    """
    Calculate antenna azimuth relative to North in degrees.

    Parameters
    ----------
    ds : xarray.Dataset
        OSCAR SAR dataset
    antenna : 'fore' Fore beam pair, 'aft' Aft beam pair


    Returns
    -------
    ds : xarray.Dataset
        OSCAR SAR dataset
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
    Compute time lag tau between Master/Slave images.

    Parameters
    ----------
    ds : xarray.Dataset
        OSCAR SAR dataset
    options : Time lag computation method. The default is 'from_SAR_time'.

    Returns
    -------
    ds : xarray.Dataset
        OSCAR SAR dataset
    ds.TimeLag : Time lag tau between Master/Slave images (s)

    """
    if options == 'from_SAR_time':
        ds['TimeLag'] = (ds.OrbTimeImage - ds.OrbTimeImageSlave)
    if options == 'from_aircraft_velocity':
        ds['TimeLag'] = (ds.Baseline / 2 * ds.MeanForwardVelocity)
    return ds


def compute_radial_surface_velocity(ds):
    """
    Compute radial surface velocity from SAR interferogram.

    Parameters
    ----------
    ds : xarray.Dataset
        OSCAR SAR dataset

    Returns
    -------
    ds : xarray.Dataset
        OSCAR SAR dataset
    ds.RadialSuraceVelocity : Surface velocity (m/s) along a radar beam radial
    """
    ds['RadialSurfaceVelocity'] = ds.Interferogram /\
        (ds.TimeLag * ds.CentralWavenumber
         * np.sin(ds.IncidenceAngleImage))

    return ds


def init_level2(dsa, dsf, dsm):
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
