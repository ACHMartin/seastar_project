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
    ds_dict : ``dict``
        OSCAR data stored as a dict with antenna number as keys and loaded
        data in ``xarray.Dataset`` format as values

    """
    fore_id = list(ds_dict.keys())[antenna_id.index('Fore')]
    mid_id = list(ds_dict.keys())[antenna_id.index('Mid')]
    aft_id = list(ds_dict.keys())[antenna_id.index('Aft')]
    # Find vars that dont exist in Mid , but exist in Fore
    for var in ds_dict[fore_id].data_vars:
        if var not in ds_dict[mid_id].data_vars:
            var_shape = (
                len(ds_dict[mid_id][list(ds_dict[mid_id].dims)[0]]),
                len(ds_dict[mid_id][list(ds_dict[mid_id].dims)[1]]))
            ds_dict[mid_id][var] = xr.DataArray(
                data=np.full(var_shape, np.NaN),
                coords=ds_dict[mid_id].coords,
                dims=ds_dict[mid_id].dims)
    # Find vars that dont exist in Fore for Aft , but exist in Mid
    for var in ds_dict[mid_id].data_vars:
        for antenna in [fore_id, aft_id]:
            if var not in ds_dict[antenna].data_vars:
                ds_dict[antenna][var] = xr.DataArray(
                    data=np.full(ds_dict[mid_id][var].shape, np.NaN),
                    coords=ds_dict[mid_id][var].coords,
                    dims=ds_dict[mid_id][var].dims)

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
        OSCAR SAR dataset containing Tx and Rx polarization strings

    Returns
    -------
    Polarization : ``str``, ``xr.DataArray``
        Antenna polarization in 'HH', 'VV' format

    """
    polarization = [str(ds.TxPolarization.data), str(ds.RxPolarization.data)]
    polarization = ''.join(polarization)
    Polarization = xr.DataArray(data=re.sub('[^VH]', '', polarization))
    Polarization.attrs['long_name'] =\
        'Antenna polarization, Transmit / Receive.'
    Polarization.attrs['units'] = '[none]'

    return Polarization


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
    Calculate multilooking Master/Slave L1b image products.

    Parameters
    ----------
    ds : ``xarray.Dataset``
        OSCAR L1a dataset
    window : ``int``
        Integer averaging window size. The default is 3.

    Returns
    -------
    ds_out : ``xarray.Dataset``
        Dataset containing computed L1b variables

    """
    ds_out = xr.Dataset()
    if 'SigmaSLCMaster' not in ds.data_vars:
        ds = compute_SLC_Master_Slave(ds)
    if 'SigmaSLCSlave' not in ds.data_vars:
        IntensityAvgComplexMasterSlave = (ds.SigmaSLCMaster ** 2)\
            .rolling(GroundRange=window).mean()\
            .rolling(CrossRange=window).mean()
    else:
        IntensityAvgComplexMasterSlave = (ds.SigmaSLCMaster * np.conjugate(ds.SigmaSLCSlave))\
            .rolling(GroundRange=window).mean()\
            .rolling(CrossRange=window).mean()
    ds_out['Intensity'] = np.abs(IntensityAvgComplexMasterSlave)
    ds_out.Intensity.attrs['long_name'] = 'SLC Intensity'
    ds_out.Intensity.attrs['description'] =\
        'Absolute single look complex image intensity'
    ds_out.Intensity.attrs['units'] = ''
    if 'SigmaSLCSlave' not in ds.data_vars:
        ds_out['Interferogram'] = (
            ['CrossRange', 'GroundRange'],
            np.full(IntensityAvgComplexMasterSlave.shape, np.NaN))
        ds_out.Interferogram.attrs['description'] =\
            'Interferogram between master/slave antenna pair.'\
            ' Values set to NaN as no Slave data present in beam dataset'
    else:
        ds_out['Interferogram'] = (
            ['CrossRange', 'GroundRange'],
            np.angle(IntensityAvgComplexMasterSlave, deg=False)
            )
        ds_out.Interferogram.attrs['description'] =\
            'Interferogram between master/slave antenna pair.'
    ds_out.Interferogram.attrs['long_name'] = 'Interferogram'
    ds_out.Interferogram.attrs['units'] = 'rad'
    if 'SigmaSLCSlave' in ds.data_vars:
        IntensityAvgMaster = (np.abs(ds.SigmaSLCMaster) ** 2)\
            .rolling(GroundRange=window).mean()\
            .rolling(CrossRange=window).mean()
        IntensityAvgSlave = (np.abs(ds.SigmaSLCSlave) ** 2)\
            .rolling(GroundRange=window).mean()\
            .rolling(CrossRange=window).mean()
        ds_out['Coherence'] = ds_out.Intensity / np.sqrt(IntensityAvgMaster
                                                         * IntensityAvgSlave)
        ds_out.Coherence.attrs['long_name'] =\
            'Coherence'
        ds_out.Coherence.attrs['description'] =\
            'Coherence between master/slave antenna pair'
        ds_out.Coherence.attrs['units'] = ''
    return ds_out

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
        IncidenceAngleImage = np.degrees(
            np.arctan(Y / (np.cos(ds.SquintImage) / ds.OrbHeightImage))
            )

    elif 'SquintImage' not in ds:
        warnings.warn(
            "WARNING: No computed antenna squint present,"
            "continuing with 45 degree assumption"
            )
        IncidenceAngleImage = np.degrees(np.arctan(np.sqrt(2)
                                                   * Y
                                                   / ds.OrbHeightImage
                                                   )
                                         )
        IncidenceAngleImage.attrs['long_name'] =\
            'Incidence angle of beam from nadir'
        IncidenceAngleImage.attrs['units'] = '[degrees]'

    return IncidenceAngleImage

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


def compute_antenna_azimuth_direction(ds, antenna, return_heading=False):
    """
    Calculate antenna azimuth relative to North in degrees.

    Parameters
    ----------
    ds : ``xarray.Dataset``
        OSCAR SAR dataset with "SquintImage" as a required field
    antenna : ``str``
        'Fore' Fore beam pair, 'Aft' Aft beam pair
    return_heading: ``bool``
        Option to return OrbitHeadingImage variable, default=False

    Returns
    -------

    AntennaAzimuthImage : ``xr.DataArray``
        Antenna beam azimuth of each image pixel (degrees from North)
    OrbitHeadingImage : ``xr.DataArray``, optional
        Aircraft heading of each image pixel (degrees from North)

    """
    if antenna not in ['Fore', 'Aft', 'Mid']:
        raise Exception('Unknown or missing Antenna direction format. Ensure'
                        'antenna variable either Fore, Aft or Mid')
    if 'OrbitHeadingImage' not in ds:
        m, n = ds.OrbTimeImage.shape
        head = np.interp(np.reshape(ds.OrbTimeImage, (1, m * n)), ds.GPSTime,
                         ds.OrbitHeading)
        head = np.reshape(head, (m, n))
        OrbitHeadingImage = xr.DataArray.copy(ds.OrbTimeImage, data=head)
        OrbitHeadingImage.attrs['long_name'] =\
            'Heading from North'
        OrbitHeadingImage.attrs['description'] =\
            'Heading (degrees N) of the airfraft for each pixel in the image'
        OrbitHeadingImage.attrs['units'] = 'deg'
    else:
        OrbitHeadingImage = ds.OrbitHeadingImage

    # antenna_direc = {'Fore': 45, 'Aft': 135, 'Mid': 90}
    lookdirec = re.sub('[^LR]', '', str(ds.LookDirection.data))
    look_direc_angle = {'L': -90, 'R': 90}

    if 'SquintImage' in ds:
        AntennaAzimuthImage = np.mod(
            OrbitHeadingImage
            + look_direc_angle[lookdirec]
            + ds.SquintImage,
            360)
    else:
        raise Exception('SquintImage is a required field in ds')
    AntennaAzimuthImage.attrs['long_name'] =\
        'Antenna azimuth'
    AntennaAzimuthImage.attrs['description'] =\
        'Antenna azimuth direction for each pixel in the image'
    AntennaAzimuthImage.attrs['units'] = 'deg'
    if return_heading:
        return AntennaAzimuthImage, OrbitHeadingImage
    else:
        return AntennaAzimuthImage


def compute_time_lag_Master_Slave(ds, options):
    """
    Compute time lag tau between Master/Slave images.

    Parameters
    ----------
    ds : ``xarray.Dataset``
        OSCAR SAR dataset containing orbit, time and baseline information
    options : ``str``
        Time lag computation method. The default is 'from_SAR_time'.

    Returns
    -------

    TimeLag : ``float``, ``xr.DataArray``
        Time lag tau between Master/Slave images (s)

    """
    if options not in ['from_SAR_time', 'from_aircraft_velocity']:
        raise Exception('Unknown time lag computation method: Please refer to'
                        'function documentation')
    if options == 'from_SAR_time':
        if 'OrbTimeImageSlave' in ds.data_vars:
            TimeLag = (ds.OrbTimeImage - ds.OrbTimeImageSlave)
            TimeLag.attrs['long_name'] = 'Time lag'
            TimeLag.attrs['description'] =\
                'Time difference between antenna master and slave.'
            TimeLag.attrs['units'] = 's'
        else:
            TimeLag = xr.DataArray(data=np.NaN)
            TimeLag.attrs['long_name'] = 'Time lag'
            TimeLag.attrs['description'] =\
                'Time difference between antenna master and slave.'\
                ' Set to NaN as no slave data in beam dataset.'
            TimeLag.attrs['units'] = 's'
    if options == 'from_aircraft_velocity':
        TimeLag = (ds.Baseline / 2 * ds.MeanForwardVelocity)
        TimeLag.attrs['long_name'] = 'Time lag'
        TimeLag.attrs['description'] =\
            'Time difference between antenna master and slave.'
        TimeLag.attrs['units'] = 's'
    return TimeLag


def compute_radial_surface_velocity(ds_L1a, ds_ml):
    """
    Compute radial surface velocity from SAR interferogram.

    Parameters
    ----------
    ds_L1a : ``xarray.Dataset``
        OSCAR L1a dataset
    ds_ml : ``xarray.Dataset``
         OSCAR L1b dataset of computed multilooking variables

    Returns
    -------

    RadialSuraceVelocity : ``float``, ``xr.DataArray``
        Surface velocity (m/s) along a radar beam radial
    IncidenceAngleImage : ``float``, ``xr.DataArray``, optional
        Incidence angle (degrees from nadir) of the radar beam. Rturned if
        IncidenceAngleImage was not already present in the input dataset and
        its computation by simple geometry was required.
    """
    return_flag = False
    if 'CentralWavenumber' not in ds_L1a.data_vars:
        ds_L1a = add_central_electromagnetic_wavenumber(ds_L1a)
    if 'IncidenceAngleImage' not in ds_L1a:
        warnings.warn(
            "WARNING: Incidence Angle not present in dataset"
            "computing incidence angle using simple geometry."
            "IncidenceAngleImage sent as output"
            )
        return_flag = True
        IncidenceAngleImage =\
            compute_incidence_angle_from_simple_geometry(ds_L1a)
    RadialSurfaceVelocity = ds_ml.Interferogram /\
        (ds_ml.TimeLag * ds_L1a.CentralWavenumber
         * np.sin(np.radians(ds_L1a.IncidenceAngleImage)))
    RadialSurfaceVelocity.attrs['long_name'] =\
        'Radial Surface Velocity'
    RadialSurfaceVelocity.attrs['units'] = 'm/s'
    if return_flag:
        return RadialSurfaceVelocity, IncidenceAngleImage
    else:
        return RadialSurfaceVelocity


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
#    warnings.warn(
#        "WARNING: Applying direction convention correction on the Aft beam,"
#        "Be aware this may be an obsolete correction in the future and will"
#        "lead to an error in retrieved current direction."
#    )
#    rsv_list[list(level1.Antenna.data).index('Aft')] = \
#        -level1.RadialSurfaceVelocity.sel(Antenna='Aft')\
#        - dswasv.sel(Antenna='Aft')
    level1['RadialSurfaceCurrent'] = xr.concat(rsv_list,
                                               'Antenna',
                                               join='outer')
    level1['RadialSurfaceCurrent'] = level1.RadialSurfaceCurrent\
        .assign_coords(Antenna=('Antenna',
                                list(level1.Antenna.data)
                                )
                       )
    level1.RadialSurfaceCurrent.attrs['long_name'] =\
        'Radial Surface Current'
    level1.RadialSurfaceCurrent.attrs['description'] =\
        'Radial Surface Current (RSC) along antenna beam direction, corrected'\
        'for Wind Artifact Surface Velocity (WASV)'
    level1.RadialSurfaceCurrent.attrs['units'] = 'm/s'

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


def replace_dummy_values(ds, dummy_val=-9999, replace=np.NaN):
    """
    Replace dummy values.

    Removes dummy values from dataset variables and replaces with a set value.
    Default dummy value is -9999 and default replacement is NaN.

    Parameters
    ----------
    ds : ``xarray.Dataset``
        Dataset containing variables with dummy values to replace.
    dummy_val : ``int``, ``float``, optional
        Dummy value to replace. The default is -9999.
    replace : ``int``, ``float``, optional
        Constant value to replace Dummy values with. The default is NaN

    Returns
    -------
    ds : ``xarray.Dataset``
        Dataset with variables scrubbed for `dummy_val` and replaced with
        `replace`

    """
    for var in ds.data_vars:
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            bad_val = ds[var].data == dummy_val
            if np.sum(bad_val) > 0:
                ds[var] = ds[var].where(~bad_val, replace)

    return ds


def track_title_to_datetime(title):
    """
    Track title to datetime conversion.

    Converts the Title attribute of an OSCAR .netcdf dataset to
    numpy.datetime64 format.

    Parameters
    ----------
    title : ``str``
        Dataset title in the form "Track : YYYYMMDDTHHMMSS"

    Returns
    -------
    track_time : ``np.datetime64``
        Time in numpy.datetime64 format

    """
    year = title.split()[2].split('T')[0][0:4]
    month = title.split()[2].split('T')[0][4:6]
    day = title.split()[2].split('T')[0][6:8]
    hour = title.split()[2].split('T')[1][0:2]
    minute = title.split()[2].split('T')[1][2:4]
    second = title.split()[2].split('T')[1][4:6]
    track_time = np.datetime64(year + '-' + month + '-' + day +
                               'T' + hour + ':' + minute + ':' + second)
    return track_time
