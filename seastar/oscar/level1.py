# -*- coding: utf-8 -*-
"""Functions to compute Level-1 (L1) data products."""
import numpy as np
import xarray as xr
import scipy as sp
import seastar
import re
import warnings
from datetime import datetime as dt


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
    antenna_list = [fore_id, mid_id, aft_id]

    for antenna_1 in antenna_list:
        for antenna_2 in [a for a in antenna_list if a not in [antenna_1]]:
            for var in ds_dict[antenna_1].data_vars:
                if var not in ds_dict[antenna_2].data_vars:
                    ds_dict[antenna_2][var] = xr.DataArray(data=np.NaN)

    return ds_dict


def merge_beams(ds_dict, antenna_id):
    """
    Merge three beams into single dataset.

    Generate a combined-look-direction OSCAR L1 dataset from separate
    look directions and add a corresponding 'Antenna' dimension and coordinate.

    Adds `longitude` and `latitude` coordinates to merged dataset.

    Requires a ``dict`` of OSCAR L1 ``xarray.Dataset``, containing the
    ``xarray.DataArray``s `LonImage` and `LatImage`. These arrays must be 2D,
    with only two dimensions.

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
            [ds_dict[key_list[0]].LatImage
             .dropna(dim=ds_dict[key_list[0]].LatImage.dims[0])
             .dropna(dim=ds_dict[key_list[0]].LatImage.dims[1]),
             ds_dict[key_list[1]].LatImage
             .dropna(dim=ds_dict[key_list[1]].LatImage.dims[0])
             .dropna(dim=ds_dict[key_list[1]].LatImage.dims[1]),
             ds_dict[key_list[2]].LatImage
             .dropna(dim=ds_dict[key_list[2]].LatImage.dims[0])
             .dropna(dim=ds_dict[key_list[2]].LatImage.dims[1])],
            ).LatImage
    ds_level1.coords['longitude'] = xr.merge(
            [ds_dict[key_list[0]].LonImage
             .dropna(dim=ds_dict[key_list[0]].LonImage.dims[0])
             .dropna(dim=ds_dict[key_list[0]].LonImage.dims[1]),
             ds_dict[key_list[1]].LonImage
             .dropna(dim=ds_dict[key_list[1]].LonImage.dims[0])
             .dropna(dim=ds_dict[key_list[1]].LonImage.dims[1]),
             ds_dict[key_list[2]].LonImage
             .dropna(dim=ds_dict[key_list[2]].LonImage.dims[0])
             .dropna(dim=ds_dict[key_list[2]].LonImage.dims[1])],
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
    ds.SigmaSLCMaster.attrs['long_name'] = 'SLC Master'
    ds.SigmaSLCMaster.attrs['description'] =\
        'Single Look Complex image, Master antenna'
    ds.SigmaSLCMaster.attrs['units'] = ''
    if 'SigmaImageSingleLookRealPartSlave' in ds.data_vars:
        ds['SigmaSLCSlave'] = (ds.SigmaImageSingleLookRealPartSlave + 1j *
                               ds.SigmaImageSingleLookImaginaryPartSlave)
        ds.SigmaSLCSlave.attrs['long_name'] = 'SLC Slave'
        ds.SigmaSLCSlave.attrs['long_name'] =\
            'Single Look Complex image, Slave antenna'
        ds.SigmaSLCSlave.attrs['units'] = ''
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
    ds.CentralWavenumber.attrs['units'] = 'rad / m'
    return ds


def compute_multilooking_Master_Slave(ds, window=3, vars_to_send=['Intensity', 'Interferogram', 'Coherence']):
    """
    Compute  multilooking Master/Slave L1b image products.

    Computes multilooking ATI variables from L1 variables present in an
    ``xarray.Dataset``. As a minimum must contain the following 2D
    ``xarray.DataArray``s:
        - `SigmaImageSingleLookRealPart`
        - `SigmaImageSingleLookImaginaryPart`
    Optionally, `ds` must include:
        - `SigmaImageSingleLookRealPartSlave`
        - `SigmaImageSingleLookImaginaryPartSlave`

    Parameters
    ----------
    ds : ``xarray.Dataset``
        OSCAR L1a dataset
    window : ``int``
        Integer averaging window size. The default is 3.
    vars_to_send:  list
        default: vars_to_send = ['Intensity, Interferogram', 'Coherence']
        can in addition take among: 'IntensityAvgComplexMasterSlave',
            'IntensityAvgMaster', 'IntensityAvgSlave'

    Returns
    -------
    ds_out : ``xarray.Dataset``
        Dataset containing computed L1b variables

    Raises
    ------
    Exception
        Raises exception if `vars_to_send` is not within:
            ['Intensity', 'Interferogram', 'Coherence',
             'IntensityAvgComplexMasterSlave', 'IntensityAvgMaster',
             'IntensityAvgSlave']
    Exception
        Raises exception if `SigmaImageSingleLookRealPart` is not a 2D variable
    """
    list_vars_to_send = set(['Intensity', 'Interferogram', 'Coherence',
                             'IntensityAvgComplexMasterSlave',
                            'IntensityAvgMaster', 'IntensityAvgSlave'])
    vars_to_send = set(vars_to_send)
    if len(vars_to_send.difference(list_vars_to_send)) > 0:
        raise Exception("vars_to_send should be within the following variables"
                        "'Intensity', 'Interferogram', 'Coherence',"
                        "'IntensityAvgComplexMasterSlave', 'IntensityAvgMaster', 'IntensityAvgSlave'")

    if len(ds.SigmaImageSingleLookRealPart.dims) > 2:
        raise Exception("The variable SigmaImageSingleLookRealPart is not a"
                        "2D variable. Please check this variable's dimensions")
    ds_out = xr.Dataset()
    if 'SigmaSLCMaster' not in ds.data_vars:
        ds = compute_SLC_Master_Slave(ds)
    if 'SigmaSLCSlave' not in ds.data_vars:
        ds_out['IntensityAvgComplexMasterSlave'] = (ds.SigmaSLCMaster ** 2)\
            .rolling({ds.SigmaSLCMaster.dims[1]: window}).mean()\
            .rolling({ds.SigmaSLCMaster.dims[0]: window}).mean()
    else:
        ds_out['IntensityAvgComplexMasterSlave'] =\
            (ds.SigmaSLCMaster * np.conjugate(ds.SigmaSLCSlave))\
            .rolling({ds.SigmaSLCMaster.dims[1]: window}).mean()\
            .rolling({ds.SigmaSLCMaster.dims[0]: window}).mean()
    ds_out.IntensityAvgComplexMasterSlave.attrs['long_name'] = \
        'Average intensity of Master/Slave SLC'
    ds_out.IntensityAvgComplexMasterSlave.attrs['units'] = ''
    ds_out['Intensity'] = np.abs(ds_out.IntensityAvgComplexMasterSlave)
    ds_out.Intensity.attrs['long_name'] = 'SLC Intensity'
    ds_out.Intensity.attrs['description'] =\
        'Average absolute single look complex image intensity (|M.S^*| with ^* complex conjugate, if S missing => |M^2|)'
    ds_out.Intensity.attrs['units'] = ''
    if 'SigmaSLCSlave' not in ds.data_vars:
        ds_out['Interferogram'] = xr.DataArray(data=np.NaN)
        ds_out.Interferogram.attrs['description'] =\
            'Interferogram between master/slave antenna pair.'\
            ' Values set to NaN as no Slave data present in beam dataset'
    else:
        ds_out['Interferogram'] = (
            ds.SigmaSLCMaster.dims,
            np.angle(ds_out.IntensityAvgComplexMasterSlave, deg=False)
            )
        ds_out.Interferogram.attrs['description'] =\
            'Interferogram between master/slave antenna pair.'
    ds_out.Interferogram.attrs['long_name'] = 'Interferogram'
    ds_out.Interferogram.attrs['units'] = 'rad'

    ds_out['IntensityAvgMaster'] = (np.abs
        (ds.SigmaSLCMaster) ** 2) \
            .rolling({ds.SigmaSLCMaster.dims[1]: window}).mean() \
            .rolling({ds.SigmaSLCMaster.dims[0]: window}).mean()

    ds_out.IntensityAvgMaster.attrs['long_name'] = \
        'Intensity Master'
    ds_out.IntensityAvgMaster.attrs['description'] = \
        'Average absolute single look complex image intensity for Master SLC (|M^2|)'
    ds_out.IntensityAvgMaster.attrs['units'] = ''

    if 'SigmaSLCSlave' in ds.data_vars:
        ds_out['IntensityAvgSlave'] = (np.abs
            (ds.SigmaSLCSlave) ** 2)\
                .rolling({ds.SigmaSLCSlave.dims[1]: window}).mean()\
                .rolling({ds.SigmaSLCSlave.dims[0]: window}).mean()

        ds_out['Coherence'] =\
            ds_out.Intensity / np.sqrt(ds_out.IntensityAvgMaster
                                       * ds_out.IntensityAvgSlave)
    else:
        # ds_out['IntensityAvgMaster'] = xr.DataArray(data=np.NaN)
        ds_out['IntensityAvgSlave'] = xr.DataArray(data=np.NaN)
        ds_out['Coherence'] = xr.DataArray(data=np.NaN)

    ds_out.IntensityAvgSlave.attrs['long_name'] = \
        'Intensity Slave'
    ds_out.IntensityAvgSlave.attrs['description'] = \
        'Average absolute single look complex image intensity for Slave SLC (|S^2|)'
    ds_out.IntensityAvgSlave.attrs['units'] = ''
    ds_out.Coherence.attrs['long_name'] = \
        'Coherence'
    ds_out.Coherence.attrs['description'] = \
        'Coherence between master/slave antenna pair'
    ds_out.Coherence.attrs['units'] = ''

    return ds_out[list(vars_to_send)]


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

    Raises
    ------
    Exception
        Raises exception if `antenna` not in `['Fore', 'Aft', 'Mid']`
    Exception
        Raises exception if `SquintImage` not present in input `ds`

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

    Raises
    ------
    Exception
        Raises exception if `options` not in the form 'from_SAR_time',
        'from_aircraft_velocity'
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


def compute_radial_surface_velocity(ds_ml):
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

    Raises
    ------
    Exception
        Raises exception if `IncidenceAngleImage` not present in input dataset
    """
    if 'CentralWavenumber' not in ds_ml.data_vars:
        ds_ml = add_central_electromagnetic_wavenumber(ds_ml)
    if 'IncidenceAngleImage' not in ds_ml:
        raise Exception('WARNING: IncidenceAngleImage not present in dataset.'
                        'Computation not possible. please check dataset used'
                        'for input')
    RadialSurfaceVelocity = ds_ml.Interferogram /\
        (ds_ml.TimeLag * ds_ml.CentralWavenumber
         * np.sin(np.radians(ds_ml.IncidenceAngleImage)))
    RadialSurfaceVelocity.attrs['long_name'] =\
        'Radial Surface Velocity'
    RadialSurfaceVelocity.attrs['units'] = 'm/s'

    return RadialSurfaceVelocity


def compute_radial_surface_current(level1, aux, gmf='mouche12'):
    """
    Compute radial surface current (RSC).

    Compute radial surface current (RSC) from radial surface velocity (RSV)
    and the wind artifact surface velocity (WASV) from:
        RSC = RSV - WASV

    Parameters
    ----------
    level1 : ``xarray.Dataset``
        L1 dataset
    aux : ``xarray.Dataset``
        Dataset containing geophysical wind data
    gmf : ``str``, optional
        Choice of geophysical model function to compute the WASV.
        The default is 'mouche12'.

    Returns
    -------
    rsc : ``xarray.DataArray``
        Radial Surface Current

    """
    dswasv = seastar.gmfs.doppler.compute_wasv(level1, aux, gmf)
    rsc = level1.RadialSurfaceVelocity - dswasv

    rsc.attrs['long_name'] =\
        'Radial Surface Current'
    rsc.attrs['description'] =\
        'Radial Surface Current (RSC) along antenna beam direction, corrected'\
        'for Wind Artifact Surface Velocity (WASV)'
    rsc.attrs['units'] = 'm/s'

    return rsc


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
        seastar.performance.scene_generation\
            .generate_wind_field_from_single_measurement(u10,
                                                    wind_direction,
                                                    level1)
    aux = xr.Dataset()
    aux['EarthRelativeWindSpeed'] = WindSpeed
    aux['EarthRelativeWindDirection'] = WindDirection

    return aux


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
    datestr = title.split()[2]
    track_time = np.datetime64(dt.strptime(datestr, '%Y%m%dT%H%M%S'))
    return track_time
