# -*- coding: utf-8 -*-
"""
Functions to calculate varirous L2 data products for the OSCAR instrument.

Created on Fri Sep 16 13:48:48 2022

@author: admartin, dlmccann
"""

import numpy as np
import xarray as xr
import scipy as sp
import seastar
import re


def init_level1_dataset(dsf, dsa, dsm):
    """
    Initialise level1 dataset.

    Initialise a combined-look-direction level1 OSCAR dataset from separate
    look directions and add a corresponding 'Antenna' dim and coord.

    Parameters
    ----------
    dsf : xarray.Dataset
        OSCAR dataset in the fore-beam look direction
    dsa : xarray.Dataset
        OSCAR dataset in the aft-beam look direction
    dsm : xarray.Dataset
        OSCAR dataset in the mid-beam look direction

    Returns
    -------
    ds_level1 : xarray.Dataset
        OSCAR dataset with combined look directions

    """
    # Check antenna polarisation fields and re-format for 'HH', 'VV'
    dsf = check_antenna_polarization(dsf)
    dsa = check_antenna_polarization(dsa)
    dsm = check_antenna_polarization(dsm)
    # Find variables missing in dsm and build NaN filled variables in their
    # place
    ds_diff = dsf[[x for x in dsf.data_vars if x not in dsm.data_vars]]
    ds_diff.where(ds_diff == np.nan, other=np.nan)
    dsm = dsm.merge(ds_diff)

    ds_level1 = xr.concat([dsf,
                           dsa,
                           dsm],
                          'Antenna', join='outer',
                          coords='all')
    ds_level1 = ds_level1.assign_coords(Antenna=('Antenna',
                                                 ['Fore', 'Aft', 'Mid']))

    return ds_level1


def check_antenna_polarization(ds):
    """
    Check polarization and standardize format.

    Check the Tx and Rx polarization variables in an OSCAR dataset and
    create a new 'Polarization' variable in a standard 'HH', 'VV' style
    format.

    Parameters
    ----------
    ds : xarray.Dataset
        OSCAR SAR dataset

    Returns
    -------
    ds : xarray.Dataset
        OSCAR SAR dataset

    """
    polarization = [str(ds.TxPolarization.data), str(ds.RxPolarization.data)]
    polarization = ''.join(polarization)
    ds['Polarization'] = re.sub('[^VH]', '', polarization)

    return ds


def add_antenna_baseline(ds, baseline):
    """
    Add antenna baseline to dataset if not already present.

    Parameters
    ----------
    ds : xarray.Dataset
        OSCAR SAR dataset
    baseline : float
        Antenna baseline (m)

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
    
    if 'SigmaSLCMaster' not in ds.data_vars:
        ds = compute_SLC_Master_Slave(ds)
    if 'SigmaSLCSlave' in ds.data_vars:
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
    else:
        ds['Intensity'] = np.abs((ds.SigmaSLCMaster * np.conjugate(ds.SigmaSLCMaster)) \
            .rolling(GroundRange=window).mean().rolling(CrossRange=window).mean())

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
    ds['IncidenceAngleImage'] = np.degrees(np.arctan(np.sqrt(2) *
                                                     Y / ds.OrbHeightImage))
    

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
    elif antenna == 'aft':
        ds['AntennaAzimuthImage'] = np.mod(ds.OrbitHeadingImage - 135, 360)
    elif antenna == 'mid':
        ds['AntennaAzimuthImage'] = np.mod(ds.OrbitHeadingImage - 90, 360)
    else:
        raise Exception('Unknown parameter, should be fore, aft or mid')

    return ds


def compute_time_lag_Master_Slave(ds, options):
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
    if options not in ['from_SAR_time','from_aircraft_velocity']:
        raise Exception('Unknown time lag computation method: Please refer to'
                        'function documentation')
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
    if 'CentralWavenumber' not in ds.data_vars:
        ds = add_central_electromagnetic_wavenumber(ds)
    
    ds = compute_incidence_angle(ds)
    ds['RadialSurfaceVelocity'] = ds.Interferogram /\
        (ds.TimeLag * ds.CentralWavenumber
         * np.sin(np.radians(ds.IncidenceAngleImage)))

    return ds


def init_level2(level1, dsm):
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
    level2.RadialSurfaceVelocity : xarray.DataArray
        Radial surface velocities (m/s) for the Fore and Aft antennas
        with corresponding dimension 'Antenna' and Coords ['Fore','Aft']
    level2.LookDirection : xarray.DataArray
        Antenna look direction (degrees N) for the Fore and Aft antennas
        with corresponding dimension 'Antenna' and Coords ['Fore','Aft']

    """
    level2 = xr.Dataset()

#    level2['RadialSurfaceVelocity'] = xr.concat(
#        [level1.RadialSuraceVelocity.sel(Antenna='Fore'),
#         level1.RadialSuraceVelocity.sel(Antenna='Aft')],
#        'Antenna', join='outer')
#    level2['RadialSurfaceVelocity'] = level2.RadialSurfaceVelocity.\
#        assign_coords(Antenna=('Antenna', ['Fore', 'Aft']))
#    level2['LookDirection'] = xr.concat(
#        [level1.AntennaAzimuthImage.sel(Antenna='Fore'),
#         level1.AntennaAzimuthImage.sel(Antenna='Aft')],
#        'Antenna', join='outer')
#    level2['LookDirection'] = level2.LookDirection.assign_coords(
#        Antenna=('Antenna', ['Fore', 'Aft']))
#    level2['IncidenceAngleImage'] = xr.concat([level1.AntennaAzimuthImage.sel(Antenna='Fore'),
#                                               dsa.IncidenceAngleImage],
#                                              'Antenna', join='outer')
#    level2['IncidenceAngleImage'] = level2.IncidenceAngleImage.assign_coords(
#        Antenna=('Antenna', ['Fore', 'Aft']))
    return level2


def init_auxiliary(level1, dsa, dsf, u10, wind_direction):
    
    "A Dataset containing WindSpeed, WindDirection,"
    "IncidenceAngleImage, LookDirection, Polarization"
    "All matrix should be the same size"
    "Polarization (1=VV; 2=HH)"

    WindSpeed, WindDirection =\
        generate_wind_field_from_single_measurement(u10,
                                                    wind_direction,
                                                    level1)
    aux = xr.Dataset()
    aux['WindSpeed'] = WindSpeed
    aux['WindDirection'] = WindDirection
#    aux['LookDirection'] = xr.concat([dsf.AntennaAzimuthImage,
#                                      dsa.AntennaAzimuthImage],
#                                     'Antenna', join='outer')
#    aux['LookDirection'] = aux.LookDirection.assign_coords(
#        Antenna=('Antenna', ['Fore', 'Aft']))
#    aux['IncidenceAngleImage'] = xr.concat([dsf.IncidenceAngleImage,
#                                            dsa.IncidenceAngleImage],
#                                           'Antenna', join='outer')
#    aux['IncidenceAngleImage'] = aux.IncidenceAngleImage.assign_coords(
#        Antenna=('Antenna', ['Fore', 'Aft']))
#    #Polarization (1=VV; 2=HH)
#    aux['Polarization'] = xr.DataArray(data=[np.zeros(WindSpeed.shape)+1,
#                                       np.zeros(WindSpeed.shape)+1],
#                                       dims=aux.LookDirection.dims,
#                                       coords=aux.LookDirection.coords,
#                                       )
#    aux['CentralWavenumber'] = dsf.CentralWavenumber.data
#    aux['CentralFreq'] = dsf.CentralFreq.data
    
    return aux
    
def generate_wind_field_from_single_measurement(u10, wind_direction, ds):
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
    wind_direction = np.mod(wind_direction, 360)
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



