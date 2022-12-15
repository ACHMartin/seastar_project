# -*- coding: utf-8 -*-
"""
Functions to calculate L1 data products for the OSCAR airborne SAR instrument.

"""

import numpy as np
import xarray as xr
import scipy as sp
import seastar
import re
import warnings


def fill_missing_variables(ds_dict, antenna_id):
    """
    Fill missing variables in OSCAR datasets.

    Identified variables that exist in one antenna dataset but not in
    others and creates NaN-filled variables in their place to aid with
    the L1 processing chain.

    Parameters
    ----------
    ds : ``dict``
        OSCAR data stored as a dict with antenna number as keys and loaded
        data in ``xarray.Dataset`` format as values
    antenna_ident : ``list``
        List of antenna identifiers in the form ['Fore', 'Mid', 'Aft'],
        corresponding to the data and keys stored in `ds`

    Returns
    -------
    ds : ``dict``
        OSCAR data stored as a dict with antenna number as keys and loaded
        data in ``xarray.Dataset`` format as values

    """
    fore_id = list(ds_dict.keys())[antenna_id.index('Fore')]
    mid_id = list(ds_dict.keys())[antenna_id.index('Mid')]
    aft_id = list(ds_dict.keys())[antenna_id.index('Aft')]

    # Find vars that dont exist in Mid, but exist in Fore
    ds_diff = ds_dict[fore_id]\
        [[x for x in ds_dict[fore_id].data_vars if x not in ds_dict[mid_id].data_vars]]
    ds_diff.where(ds_diff == np.nan, other=np.nan)
    ds_dict[mid_id] = ds_dict[mid_id].merge(ds_diff)
    
    # Find vars that dont exist in Fore, but exist in Mid
    ds_diff = ds_dict[mid_id]\
        [[x for x in ds_dict[mid_id].data_vars if x not in ds_dict[fore_id].data_vars]]
    ds_diff.where(ds_diff == np.nan, other=np.nan)
    ds_dict[fore_id] = ds_dict[fore_id].merge(ds_diff)

    # Find vars that dont exist in Aft, but exist in Mid
    ds_diff = ds_dict[mid_id]\
        [[x for x in ds_dict[mid_id].data_vars if x not in ds_dict[aft_id].data_vars]]
    ds_diff.where(ds_diff == np.nan, other=np.nan)
    ds_dict[aft_id] = ds_dict[aft_id].merge(ds_diff)

    return ds_dict


def merge_beams(ds_dict, antenna_id):
    """
    Merge three beams into single dataset.

    Generate a combined-look-direction OSCAR L1 dataset from separate
    look directions and add a corresponding 'Antenna' dimension and coordinate.

    Parameters
    ----------
    ds : ``dict``
        OSCAR data stored as a dict with antenna number as keys and loaded
        data in ``xarray.Dataset`` format as values
    antenna_ident : ``list``
        List of antenna identifiers in the form ['Fore', 'Mid', 'Aft'],
        corresponding to the data and keys stored in `ds`

    Returns
    -------
    ds_level1 : ``xarray.Dataset``
        OSCAR dataset with combined look directions

    """
    ds_level1 = xr.concat(list(ds_dict.values()),
                          'Antenna', join='outer',
                          coords='all')
    ds_level1 = ds_level1.assign_coords(Antenna=('Antenna', antenna_id))
    key_list = list(ds_dict.keys())
    ds_level1.coords['latitude'] = xr.merge(
            [ds_dict[key_list[0]].LatImage.dropna(dim='CrossRange'),
             ds_dict[key_list[1]].LatImage.dropna(dim='CrossRange'),
             ds_dict[key_list[2]].LatImage.dropna(dim='CrossRange')],
            ).LatImage
    ds_level1.coords['longitude'] = xr.merge(
            [ds_dict[key_list[0]].LonImage.dropna(dim='CrossRange'),
             ds_dict[key_list[1]].LonImage.dropna(dim='CrossRange'),
             ds_dict[key_list[2]].LonImage.dropna(dim='CrossRange')],
            ).LonImage

    return ds_level1


def check_antenna_polarization(ds):
    """
    Check polarization and standardize format.

    Check the Tx and Rx polarization variables in an OSCAR dataset and
    create a new 'Polarization' variable in a standard 'HH', 'VV' style
    format.

    Parameters
    ----------
    ds : ``xarray.Dataset``
        OSCAR SAR dataset

    Returns
    -------
    Polarization : ``str``
        Antenna polarization in 'HH', 'VV' format
    ds : ``xarray.Dataset``
        OSCAR SAR dataset

    """
    polarization = [str(ds.TxPolarization.data), str(ds.RxPolarization.data)]
    polarization = ''.join(polarization)
    ds['Polarization'] = re.sub('[^VH]', '', polarization)
    ds.Polarization.attrs['long_name'] =\
        'Antenna polarization, Transmit / Receive.'
    ds.Polarization.attrs['units'] = '[none]'
    return ds


def compute_antenna_baseline(antenna_baseline):
    """
    Add antenna baseline to dataset if not already present.

    Parameters
    ----------
    ds : ``xarray.Dataset``
        OSCAR SAR dataset
    baseline : ``float``
        Antenna baseline (m)

    Returns
    -------
    ds : ``xarray.Dataset``
        OSCAR SAR dataset in netCDF format
    ds.Baseline : ``float``
        Antenna baseline distance (m)

    """
    baseline = xr.DataArray(data=antenna_baseline)
    baseline.attrs['long_name'] =\
        'Antenna ATI baseline'
    baseline.attrs['units'] = '[m]'
    return baseline


def compute_SLC_Master_Slave(ds):
    """
    Compute SLC master/slave complex images.

    Parameters
    ----------
    ds : ``xarray.Dataset``
        OSCAR SAR dataset

    Returns
    -------
    ds : ``xarray.Dataset``
        OSCAR SAR dataset
    ds.SigmaSLCMaster: ``xarray.DataArray``
        Master SLC image
    ds.SigmaSLCSlave: ``xarray.DataArray``
        Slave SLC image

    """
    ds['SigmaSLCMaster'] = (ds.SigmaImageSingleLookRealPart + 1j *
                            ds.SigmaImageSingleLookImaginaryPart)
    ds.SigmaSLCMaster.attrs['long_name'] =\
        'Single Look Complex image, Master antenna'
    ds.SigmaSLCMaster.attrs['units'] = '[none]'
    if 'SigmaImageSingleLookRealPartSlave' in ds.data_vars:
        ds['SigmaSLCSlave'] = (ds.SigmaImageSingleLookRealPartSlave + 1j *
                               ds.SigmaImageSingleLookImaginaryPartSlave)
        ds.SigmaSLCSlave.attrs['long_name'] =\
            'Single Look Complex image, Slave antenna'
        ds.SigmaSLCSlave.attrs['units'] = '[none]'
    return ds


def add_central_electromagnetic_wavenumber(ds):
    """
    Calculate wavenumber of the central electromagnetic frequency.

    Parameters
    ----------
    ds : ``xarray.Dataset``
        OSCAR SAR dataset
    ds.CentralFreq : ``float``, ``xarray.DataArray``
        Central radar frequency (Hz)

    Returns
    -------
    ds : ``xarray.Dataset``
        OSCAR SAR dataset
    ds.CentralWavenumber : ``float``, ``xarray.DataArray``
        Central radar wavenumber (rad / m)

    """
    ds['CentralWavenumber'] = 2 * np.pi * ds.CentralFreq / sp.constants.c
    ds.CentralWavenumber.attrs['long_name'] =\
        'Central electromagnetic wavenumber'
    ds.CentralWavenumber.attrs['units'] = '[radians / m]'
    return ds


def compute_multilooking_Master_Slave(ds, window=3):
    """
    Calculate multilooking Master/Slave L2 image products.

    Parameters
    ----------
    ds : ``xarray.Dataset``
        OSCAR SAR dataset
    window : ``int``
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
    ds['IntensityAvgComplexMasterSlave'] = (ds.SigmaSLCMaster * np.conjugate(ds.SigmaSLCSlave))\
        .rolling(GroundRange=window).mean().rolling(CrossRange=window).mean()
    ds.IntensityAvgComplexMasterSlave.attrs['long_name'] =\
        'Average intensity of Master/Slave single looc complex images'
    ds.IntensityAvgComplexMasterSlave.attrs['units'] = '[none]'
    ds['Intensity'] = np.abs(ds.IntensityAvgComplexMasterSlave)
    ds.Intensity.attrs['long_name'] =\
        'Absolute single look complex image intensity'
    ds.Intensity.attrs['units'] = '[none]'
    ds['Interferogram'] = (
        ['CrossRange', 'GroundRange'],
        np.angle(ds.IntensityAvgComplexMasterSlave, deg=False)
        )
    ds.Interferogram.attrs['long_name'] =\
        'Interferogram between master/slave antenna pair'
    ds.Interferogram.attrs['units'] = '[radians]'
    ds['IntensityAvgMaster'] = (np.abs(ds.SigmaSLCMaster) ** 2)\
        .rolling(GroundRange=window).mean().rolling(CrossRange=window).mean()
    ds['IntensityAvgSlave'] = (np.abs(ds.SigmaSLCSlave) ** 2)\
        .rolling(GroundRange=window).mean().rolling(CrossRange=window).mean()
    ds['Coherence'] = ds.Intensity / np.sqrt(ds.IntensityAvgMaster
                                             * ds.IntensityAvgSlave)
    ds.Coherence.attrs['long_name'] =\
        'Coherence between master/slave antenna pair'
    ds.Coherence.attrs['units'] = '[none]'
    return ds

def compute_local_coordinates(ds):
    lookdirec = re.sub('[^LR]', '', str(ds.LookDirection.data))
    utmzone = int(ds.UTMZone)
    utmhemi = dict({0: 'N', 1: 'S'})[int(ds.Hemisphere)]
    E, N = seastar.utils.tools.wgs2utm_v3(ds.OrbLatImage,
                                          ds.OrbLonImage,
                                          utmzone,
                                          utmhemi
                                          )
    gridinfo = ds.GBPGridInfo.data
    if lookdirec == 'R':
        orb_x = (E - gridinfo[0]) * np.sin(gridinfo[8])\
            + (N-gridinfo[1]) * np.cos(gridinfo[8])
        orb_y = (E - gridinfo[0]) * np.cos(gridinfo[8])\
            - (N - gridinfo[1]) * np.sin(gridinfo[8])
    if lookdirec == 'L':
        orb_x = (E - gridinfo[0]) * np.sin(gridinfo[8])\
            + (N - gridinfo[1]) * np.cos(gridinfo[8])
        orb_y = -(E - gridinfo[0]) * np.cos(gridinfo[8])\
            + (N - gridinfo[1]) * np.sin(gridinfo[8])

    return orb_x, orb_y


def compute_incidence_angle_from_simple_geometry(ds):

    X, Y = np.meshgrid(ds.CrossRange, ds.GroundRange, indexing='ij')
    if 'SquintImage' in ds:
        ds['IncidenceAngleImage'] = np.degrees(
            np.arctan(Y / (np.cos(ds.SquintImage) / ds.OrbHeightImage))
            )

    elif 'SquintImage' not in ds:
        warnings.warn(
            "WARNING: No computed antenna squint present,"
            "continuing with 45 degree assumption"
            )
        ds['IncidenceAngleImage'] = np.degrees(np.arctan(np.sqrt(2)
                                                         * Y
                                                         / ds.OrbHeightImage
                                                         )
                                               )
        ds.IncidenceAngleImage.attrs['long_name'] =\
            'Incidence angle of beam from nadir'
        ds.IncidenceAngleImage.attrs['units'] = '[degrees]'

    return ds

def compute_incidence_angle_from_GBPGridInfo(ds):
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

    z_axis = ds.DEMImage
    x_axis = ds.CrossRange.data
    y_axis = ds.GroundRange.data
    lookdirec = re.sub('[^LR]', '', str(ds.LookDirection.data))
    if lookdirec == 'R':
        y_axis = np.abs(y_axis)

    utmzone = int(ds.UTMZone)
    utmhemi = dict({0: 'N', 1: 'S'})[int(ds.Hemisphere)]
    E, N = seastar.utils.tools.wgs2utm_v3(ds.OrbLatImage,
                                          ds.OrbLonImage,
                                          utmzone,
                                          utmhemi
                                          )
    gridinfo = ds.GBPGridInfo.data
    if lookdirec == 'R':
        orb_x = (E - gridinfo[0]) * np.sin(gridinfo[8])\
            + (N-gridinfo[1]) * np.cos(gridinfo[8])
        orb_y = (E - gridinfo[0]) * np.cos(gridinfo[8])\
            - (N - gridinfo[1]) * np.sin(gridinfo[8])
    if lookdirec == 'L':
        orb_x = (E - gridinfo[0]) * np.sin(gridinfo[8])\
            + (N - gridinfo[1]) * np.cos(gridinfo[8])
        orb_y = -(E - gridinfo[0]) * np.cos(gridinfo[8])\
            + (N - gridinfo[1]) * np.sin(gridinfo[8])
    orb_z = ds.OrbHeightImage
    xlen = len(x_axis)
    ylen = len(y_axis)

#speedup_factor_x = 50
#speedup_factor_y = 50
    speedup_factor_x = 48
    speedup_factor_y = 40
    orbx = xr.DataArray(data=orb_x,
                        coords=[ds.CrossRange, ds.GroundRange],
                        dims=('CrossRange', 'GroundRange')
                        )
    orbx = orbx.coarsen(CrossRange=speedup_factor_x, boundary='trim').mean()\
        .coarsen(GroundRange=speedup_factor_y, boundary='trim').mean()
    xaxis = orbx.CrossRange
    yaxis = orbx.GroundRange
    orbx = orbx.data

    orby = xr.DataArray(data=orb_y,
                        coords=[ds.CrossRange, ds.GroundRange],
                        dims=('CrossRange', 'GroundRange')
                        )
    orby = orby.coarsen(CrossRange=speedup_factor_x, boundary='trim').mean()\
        .coarsen(GroundRange=speedup_factor_y, boundary='trim').mean()
    orby = orby.data

    orbz = xr.DataArray(data=orb_z,
                        coords=[ds.CrossRange, ds.GroundRange],
                        dims=('CrossRange', 'GroundRange')
                        )
    orbz = orbz.coarsen(CrossRange=speedup_factor_x, boundary='trim').mean()\
        .coarsen(GroundRange=speedup_factor_y, boundary='trim').mean()
    orbz = orbz.data

    zaxis = xr.DataArray(data=z_axis,
                         coords=[ds.CrossRange, ds.GroundRange],
                         dims=('CrossRange', 'GroundRange')
                         )
    zaxis = zaxis.coarsen(CrossRange=speedup_factor_x, boundary='trim').mean()\
        .coarsen(GroundRange=speedup_factor_y, boundary='trim').mean()
    zaxis = zaxis.data

    xaxis2, yaxis2 = np.meshgrid(xaxis, yaxis, indexing='ij')
    x_axis2, y_axis2 = np.meshgrid(x_axis, y_axis, indexing='ij')
    yaxis22 = yaxis2

    tamimg = zaxis.shape
    look_angle = np.empty(zaxis.shape)
    inc_angle = np.empty(zaxis.shape)
    for l in range(tamimg[0]):
        for k in range(tamimg[1]):
            lm1 = l - 1
            lp1 = l + 2
            km1 = k - 1
            kp1 = k + 2
            if lm1 == -1:
                lm1 = 0
            if lp1 > tamimg[0]:
                lp1 = tamimg[0]
            if km1 == -1:
                km1 = 0
            if kp1 > tamimg[1]:
                kp1 = tamimg[1]
            zaxis1 = zaxis[lm1:lp1, km1:kp1]
            rim = np.sqrt(
                (orbx[lm1:lp1, km1:kp1] - xaxis2[lm1:lp1, km1:kp1]) ** 2
                + (orby[lm1:lp1, km1:kp1] - yaxis22[lm1:lp1, km1:kp1]) ** 2
                + (orbz[lm1:lp1, km1:kp1] - zaxis[lm1:lp1, km1:kp1]) ** 2
                )
            rgr = np.sqrt(
                (orbx[lm1:lp1, km1:kp1] - xaxis2[lm1:lp1, km1:kp1]) ** 2
                + (orby[lm1:lp1, km1:kp1] - yaxis22[lm1:lp1, km1:kp1]) ** 2
                )
            vetorp = np.array([xaxis2[l, k], yaxis22[l, k], zaxis[l, k]])
            vetoro = np.array([orbx[l, k].T, orby[l, k].T, orbz[l, k].T])
            vetor = vetorp - vetoro
            vetoru = np.zeros(vetor.shape)
            vetoruuu = np.zeros(vetor.shape)
            vetoruuu[2] = 1  # look angle suplement
            vetoru[1] = 1
            escalarprod = sum(vetor * vetoru)
            magvetor = np.sqrt(vetor[0] ** 2 + vetor[1] ** 2 + vetor[2] ** 2)
            theta = np.arccos(escalarprod / magvetor) - np.pi / 2
            escalarprod = sum(vetor * vetoruuu)
            magvetor = np.sqrt(vetor[0] ** 2 + vetor[1] ** 2 + vetor[2] ** 2)
            look_angle[l, k] = np.pi-np.arccos(escalarprod / magvetor)
            tamrgr = rgr.shape
            # Expect to see RunTimeWarnings in this block, due to
            # unsatisfactory behaviour of np.nanmean. It is considered safe to
            # ignore these warnings as no simple workaround exists.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                deltarggroundx = rgr - np.roll(rgr, (0, 1), axis=(0, 1))
                deltarggroundx[:, 0] = np.nan
                deltarggroundx = np.nanmean(deltarggroundx)
                deltarggroundy = rgr - np.roll(rgr, (1, 0), axis=(0, 1))
                deltarggroundy[0, :] = np.nan
                deltarggroundy = np.nanmean(deltarggroundy)
                deltazaxisx = zaxis1 - np.roll(zaxis1, (0, 1), axis=(0, 1))
                deltazaxisx[:, 0] = np.nan
                deltazaxisx = np.nanmean(deltazaxisx)
                deltazaxisy = zaxis1 - np.roll(zaxis1, (1, 0), axis=(0, 1))
                deltazaxisy[0, :] = np.nan
                deltazaxisy = np.nanmean(deltazaxisy)
                deltargground = deltarggroundy * np.cos(theta)\
                    + deltarggroundx * np.sin(theta)
                deltazaxis = deltazaxisy * np.cos(theta)\
                    + deltazaxisx * np.sin(theta)

        alpha_s = np.arctan(deltazaxis / deltargground)
        inc_angle[l, k] = look_angle[l, k] - alpha_s
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        inc_angle = xr.DataArray(data=inc_angle,
                                 coords=[xaxis, yaxis],
                                 dims=('CrossRange', 'GroundRange')
                                 )
        look_angle = xr.DataArray(data=look_angle,
                                  coords=[xaxis, yaxis],
                                  dims=('CrossRange', 'GroundRange')
                                  )

    ds['IncidenceAngleImage'] = inc_angle.interp_like(ds.CrossRange)

    return ds


def compute_antenna_azimuth_direction(ds, antenna):
    """
    Calculate antenna azimuth relative to North in degrees.

    Parameters
    ----------
    ds : ``xarray.Dataset``
        OSCAR SAR dataset
    antenna : ``str``
        'Fore' Fore beam pair, 'Aft' Aft beam pair


    Returns
    -------
    ds : ``xarray.Dataset``
        OSCAR SAR dataset
    ds.OrbitHeadingImage : Aircraft heading of each image pixel
    (degrees from North)
    ds.AntennaAzimuthImage : Antenna beam azimuth of each image pixel
    (degrees from North)

    """
    if antenna not in ['Fore', 'Aft', 'Mid']:
        raise Exception('Unknown or missing Antenna direction format. Ensure'
                        'antenna variable either Fore, Aft or Mid')
    if 'OrbitHeadingImage' not in ds:
        m, n = ds.OrbTimeImage.shape
        head = np.interp(np.reshape(ds.OrbTimeImage, (1, m * n)), ds.GPSTime,
                         ds.OrbitHeading)
        head = np.reshape(head, (m, n))
        ds['OrbitHeadingImage'] = xr.DataArray.copy(ds.OrbTimeImage, data=head)
        ds.OrbitHeadingImage.attrs['long_name'] =\
            'Heading (degrees N) of the airfraft for each pixel in the image'
        ds.OrbitHeadingImage.attrs['units'] = '[degrees]'

    antenna_direc = {'Fore': 45, 'Aft': 135, 'Mid': 90}
    lookdirec = re.sub('[^LR]', '', str(ds.LookDirection.data))
    look_direc_angle = {'L': -90, 'R': 90}

    if 'SquintImage' in ds:
        ds['AntennaAzimuthImage'] = np.mod(
            np.mod(
                ds.OrbitHeadingImage
                + look_direc_angle[lookdirec],
                360)
            + ds.SquintImage,
            360)
    elif 'SquintImage' not in ds:
        warnings.warn(
            "WARNING: No computed antenna squint present,"
            "continuing with 45 degree Fore/Aft squint assumption"
        )
        ds['AntennaAzimuthImage'] = np.mod(ds.OrbitHeadingImage
                                           + (np.sign(
                                               look_direc_angle[lookdirec])
                                               * antenna_direc[antenna]),
                                           360)
    ds.AntennaAzimuthImage.attrs['long_name'] =\
        'Antenna azimuth direction for each pixel in the image'
    ds.AntennaAzimuthImage.attrs['units'] = '[degrees North]'
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
    if options not in ['from_SAR_time', 'from_aircraft_velocity']:
        raise Exception('Unknown time lag computation method: Please refer to'
                        'function documentation')
    if options == 'from_SAR_time':
        ds['TimeLag'] = (ds.OrbTimeImage - ds.OrbTimeImageSlave)
    if options == 'from_aircraft_velocity':
        ds['TimeLag'] = (ds.Baseline / 2 * ds.MeanForwardVelocity)
    ds.TimeLag.attrs['long_name'] = 'Time difference between antenna pair'
    ds.TimeLag.attrs['units'] = '[s]'
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
    if 'IncidenceAngleImage' not in ds:
        warnings.warn(
            "WARNING: Incidence Angle not present in dataset,"
            "computing incidence angle using simple geometry."
            )
        ds = compute_incidence_angle_from_simple_geometry(ds)
    ds['RadialSurfaceVelocity'] = ds.Interferogram /\
        (ds.TimeLag * ds.CentralWavenumber
         * np.sin(np.radians(ds.IncidenceAngleImage)))
    ds.RadialSurfaceVelocity.attrs['long_name'] =\
        'Radial Surface Velocity (RSV) along antenna beam direction'
    ds.RadialSurfaceVelocity.attrs['units'] = '[m/s]'
    return ds


def compute_radial_surface_current(level1, aux, gmf='mouche12'):
    """
    Compute radial surface current (RSC).

    Compute radial surface current (RSC) from radial surface velocity (RSV)
    and the wind artifact surface velocity (WASV) from:
        RSC = RSV - WASV

    Parameters
    ----------
    level1 : xarray.Dataset
        L1 dataset
    aux : xarray.Dataset
        Dataset containing geophysical wind data
    gmf : str, optional
        Choice of geophysical model function to compute the WASV.
        The default is 'mouche12'.

    Returns
    -------
    level2 : xarray.Dataset
        L2 dataset

    """
    dswasv = seastar.gmfs.doppler.compute_wasv(level1,
                                               aux,
                                               gmf
                                               )
    rsv_list = [level1.RadialSurfaceVelocity.sel(Antenna=a)
                - dswasv.sel(Antenna=a)
                for a in list(level1.Antenna.data)
                ]
    warnings.warn(
        "WARNING: Applying direction convention correction on the Aft beam,"
        "Be aware this may be an obsolete correction in the future and will"
        "lead to an error in retrieved current direction."
    )
    rsv_list[list(level1.Antenna.data).index('Aft')] = \
        -level1.RadialSurfaceVelocity.sel(Antenna='Aft')\
        - dswasv.sel(Antenna='Aft')
    level1['RadialSurfaceCurrent'] = xr.concat(rsv_list,
                                               'Antenna',
                                               join='outer')
    level1['RadialSurfaceCurrent'] = level1.RadialSurfaceCurrent\
        .assign_coords(Antenna=('Antenna',
                                list(level1.Antenna.data)
                                )
                       )
    level1.RadialSurfaceCurrent.attrs['long_name'] =\
        'Radial Surface Current (RSC) along antenna beam direction, corrected'\
        'for Wind Artifact Surface Velocity (WASV)'
    level1.RadialSurfaceCurrent.attrs['units'] = '[m/s]'

    return level1


def init_level2(level1):
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
#    level2.coords['longitude'] = level1.sel(Antenna='Fore').LonImage
#    level2.coords['latitude'] = level1.sel(Antenna='Fore').LatImage
    level2.coords['longitude'] = level1.longitude
    level2.coords['latitude'] = level1.latitude
#    level2 = level2.drop('Antenna')

    return level2


def init_auxiliary(level1, u10, wind_direction):

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



