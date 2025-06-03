# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr
import os
import scipy as sp
import seastar
import re
import glob
import warnings
from datetime import datetime as dt
from datetime import timezone

from _version import __version__
from _logger import logger


def fill_missing_variables(ds_dict, antenna_list):
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
    antenna_list : ``list``
        List of antenna identifiers in the form ['Fore', 'Mid', 'Aft'],
        corresponding to the data and keys stored in `ds`

    Returns
    -------
    ds_dict : ``dict``
        OSCAR data stored as a dict with antenna number as keys and loaded
        data in ``xarray.Dataset`` format as values

    """
    fore_id = list(ds_dict.keys())[antenna_list.index('Fore')]
    mid_id = list(ds_dict.keys())[antenna_list.index('Mid')]
    aft_id = list(ds_dict.keys())[antenna_list.index('Aft')]
    antenna_list2 = [fore_id, mid_id, aft_id]

    for antenna_1 in antenna_list2:
        for antenna_2 in [a for a in antenna_list2 if a not in [antenna_1]]:
            for var in ds_dict[antenna_1].data_vars:
                if var not in ds_dict[antenna_2].data_vars:
                    ds_dict[antenna_2][var] = xr.DataArray(data=np.nan)

    return ds_dict


def merge_beams(ds_dict, antenna_list):
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
    antenna_list : ``list``
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
    ds_level1 = ds_level1.assign_coords(Antenna=('Antenna', antenna_list))
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


def compute_multilooking_Master_Slave(ds, window=3,
                                      vars_to_send=['Intensity',
                                                    'Interferogram',
                                                    'Coherence']):
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
        raise Exception(f"vars_to_send should be within the following variables: {list_vars_to_send}")

    if len(ds.SigmaImageSingleLookRealPart.dims) > 2:
        raise Exception("The variable SigmaImageSingleLookRealPart is not a"
                        "2D variable. Please check this variable's dimensions")
    ds_out = xr.Dataset()
    ds_out.attrs = ds.attrs.copy()          # Copy the attrs from ds to ds_out
    if 'SigmaSLCMaster' not in ds.data_vars:
        ds = compute_SLC_Master_Slave(ds)

    ds_out['IntensityAvgMaster'] = (np.abs
        (ds.SigmaSLCMaster) ** 2) \
            .rolling({ds.SigmaSLCMaster.dims[1]: window}).mean() \
            .rolling({ds.SigmaSLCMaster.dims[0]: window}).mean()

    if 'SigmaSLCSlave' in ds.data_vars:
        ds_out['IntensityAvgSlave'] = (np.abs(ds.SigmaSLCSlave) ** 2)\
                .rolling({ds.SigmaSLCSlave.dims[1]: window}).mean()\
                .rolling({ds.SigmaSLCSlave.dims[0]: window}).mean()

        ds_out['IntensityAvgComplexMasterSlave'] =\
            (ds.SigmaSLCMaster * np.conjugate(ds.SigmaSLCSlave))\
            .rolling({ds.SigmaSLCMaster.dims[1]: window}).mean()\
            .rolling({ds.SigmaSLCMaster.dims[0]: window}).mean()

        ds_out['Intensity'] = np.abs(ds_out.IntensityAvgComplexMasterSlave)

        ds_out['Interferogram'] = (
            ds.SigmaSLCMaster.dims,
            np.angle(ds_out.IntensityAvgComplexMasterSlave, deg=False)
            )

        ds_out['Coherence'] =\
            ds_out.Intensity / np.sqrt(ds_out.IntensityAvgMaster
                                       * ds_out.IntensityAvgSlave)

    else:
        ds_out['IntensityAvgSlave'] = xr.DataArray(data=np.nan)

        ds_out['IntensityAvgComplexMasterSlave'] = xr.DataArray(data=np.nan)

        ds_out['Intensity'] = ds_out.IntensityAvgMaster 

        ds_out['Interferogram'] = xr.DataArray(data=np.nan)

        ds_out['Coherence'] = xr.DataArray(data=np.nan)

    ds_out.IntensityAvgMaster.attrs['long_name'] = \
        'Intensity Master'
    ds_out.IntensityAvgMaster.attrs['description'] = \
        'Average absolute single look complex image intensity for Master SLC (|M^2|)'
    ds_out.IntensityAvgMaster.attrs['units'] = ''

    ds_out.IntensityAvgSlave.attrs['long_name'] = \
        'Intensity Slave'
    ds_out.IntensityAvgSlave.attrs['units'] = ''

    ds_out.IntensityAvgComplexMasterSlave.attrs['long_name'] = \
        'Average intensity of Master/Slave SLC'
    ds_out.IntensityAvgComplexMasterSlave.attrs['units'] = ''

    ds_out.Intensity.attrs['long_name'] = 'SLC Intensity'
    ds_out.Intensity.attrs['description'] =\
        'Average absolute single look complex image intensity (|M.S^*| with ^* complex conjugate, if S missing => |M^2|)'
    ds_out.Intensity.attrs['units'] = ''

    ds_out.Interferogram.attrs['description'] =\
        'Interferogram between master/slave antenna pair.'
    ds_out.Interferogram.attrs['long_name'] = 'Interferogram'
    ds_out.Interferogram.attrs['units'] = 'rad'

    ds_out.Coherence.attrs['long_name'] = \
        'Coherence'
    ds_out.Coherence.attrs['description'] = \
        'Coherence between master/slave antenna pair'
    ds_out.Coherence.attrs['units'] = ''
    
    # Addition of the resolution attribute
    ds_out.attrs['MultiLookCrossRangeEffectiveResolution'] = window * ds.attrs['SingleLookCrossRangeGridResolution']
    ds_out.attrs['MultiLookGroundRangeEffectiveResolution'] = window * ds.attrs['SingleLookGroundRangeGridResolution']
    
    # Update of the processing level attribute
    ds_out.attrs['ProcessingLevel'] = "L1B"
    
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
            TimeLag = xr.DataArray(data=np.nan)
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
    level2.attrs = level1.attrs.copy()
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
    aux['WindSpeed'] = WindSpeed
    aux['WindDirection'] = WindDirection

    return aux


def replace_dummy_values(ds, dummy_val=-9999, replace=np.nan):
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


def track_title_to_datetime(start_time):
    """
    Track start date to datetime conversion.

    Converts the StartTime attribute of an OSCAR .netcdf dataset to
    numpy.datetime64 format.

    Parameters
    ----------
    start_time : ``str``
        Dataset start time in the form "YYYYMMDDTHHMM"
    """
    return np.datetime64(dt.strptime(start_time, '%Y%m%dT%H%M'))


def processing_OSCAR_L1AP_to_L1B(L1AP_folder, campaign, acq_date, track, dict_L1B_process=dict(), write_nc=False):
    """
    Processing chain from L1AP tp L1B.
    
    This function processes the OSCAR data from L1AP to L1B. It needs a triplet of L1AP file to generate a L1B file with data of the three antennas.
    
    Parameters
    ----------
        L1AP_folder : ``str``
            Folder of the L1AP files.
        campaign : ``str``
            Campaign name as defined in config/Campaign_name_lookup.ini file. Example of campaign name: 202205_IroiseSea, 202305_MedSea.
        acq_date : ``str``
            Date of the data acquisition with the format "YYYYMMDD".
        track : ``str``
            Track name as defined in config/XXX_TrackNames.ini file. Format should be "Track_x" for tracks over ocean and 
            "Track_Lx" for tracks over land with "x" the number of the track.
        dict_L1B_process : ``dict``
            Dictionnary containing information about the window for the rolling mean for the multilooking computation,
            the vars to keep (vars_to_keep) from L1AP to L1B file and the vars to provide (vars_to_send) after the multilooking computation.
            window default value is 3
            vars_to_keep default list is: ['LatImage', 'LonImage', 'IncidenceAngleImage',
                                           'LookDirection', 'SquintImage', 'CentralFreq', 'OrbitHeadingImage']
            vars_to_send default list is: ['Intensity', 'Interferogram', 'Coherence']
        write_nc (bool, optional): 
            Argument to write the data in a netcdf file. Defaults to False.

    Returns:
    ----------
        ds_L1B: ``xr.Dataset``
            Xarray dataset of the L1B OSCAR data.
    """

    # checking acq_date format:
    if seastar.oscar.tools.is_valid_acq_date(acq_date):
        logger.info("'acq_date' format is okay")
    else: 
        logger.error("'acq_date' should be a valid date string in 'YYYYMMDD' format ")
        raise ValueError("'acq_date' should be a valid date string in 'YYYYMMDD' format ")

    # Checking campaign name:
    valid_campaign = ["202205_IroiseSea", "202305_MedSea"]
    if campaign in valid_campaign:
        logger.info("Camapaign name has valid value.")
    else:
        logger.error(f"Unexpected campaign value: {campaign}")
        raise ValueError(f"Unexpected campaign value: {campaign}. Shall be {valid_campaign}.")
    

    # Getting the date for every tracks in the dict.
    track_names_dict = seastar.utils.readers.read_config_OSCAR('track', {"campaign" : campaign, "flight" : acq_date})  #read_OSCAR_track_names_config(campaign, acq_date)
    
    # Getting the date for the track we are interested in
    if track in track_names_dict.values():
        date_of_track = [key for key, v in track_names_dict.items() if v == track][0]
        logger.info(f"Track '{track}' found, acquisition time: {date_of_track}")
    else:
        logger.warning(f"Track '{track}' not found in track_names_dict. The code will crash.")
        
    # Loading the file names of the files corresponding to date_of_track (triplet of file - one per antenna)
    L1AP_file_names = [os.path.basename(file) for file in sorted(glob.glob(L1AP_folder + "/*" + date_of_track + "*.nc"))]

    #-----------------------------------------------------------
    #               L1B PROCESSING
    #-----------------------------------------------------------
    
    vars_to_keep = dict_L1B_process.get('vars_to_keep',['LatImage','LonImage','IncidenceAngleImage',
                                                        'LookDirection','SquintImage','CentralFreq','OrbitHeadingImage'])

    logger.info(f"Opening track: {track} on day: {acq_date}")

    ds_dict = seastar.oscar.tools.load_L1AP_OSCAR_data(L1AP_folder, L1AP_file_names) 

    # Getting the antenna identificators
    antenna_list = list(ds_dict.keys())
    logger.info(f"Antenna: {antenna_list}")

    ds_ml = dict()
    for i, antenna in enumerate(antenna_list):
        logger.info(f"Begining of the processing of {L1AP_file_names[i]}")
        ds_dict[antenna] = seastar.oscar.level1.replace_dummy_values(
                ds_dict[antenna], dummy_val=float(ds_dict[antenna].Dummy.data))
        ds_ml[antenna] = seastar.oscar.level1.compute_multilooking_Master_Slave(ds_dict[antenna], dict_L1B_process['window'])
        ds_ml[antenna]['Polarization'] = seastar.oscar.level1.check_antenna_polarization(ds_dict[antenna])
        ds_ml[antenna]['AntennaAzimuthImage'] = seastar.oscar.level1.compute_antenna_azimuth_direction(ds_dict[antenna], antenna=antenna)
        ds_ml[antenna]['TimeLag'] = seastar.oscar.level1.compute_time_lag_Master_Slave(ds_dict[antenna], options='from_SAR_time')         # Time difference between Master and Slave
        #Rolling median to smooth out TimeLag errors
        if not np.isnan(ds_ml[antenna].TimeLag).all():
            ds_ml[antenna]['TimeLag'] = ds_ml[antenna].TimeLag\
                .rolling({'CrossRange': 5}).median()\
                .rolling({'GroundRange': 5}).median()
                
        ds_ml[antenna][vars_to_keep] = ds_dict[antenna][vars_to_keep]
        ds_ml[antenna]['TrackTime'] = seastar.oscar.level1.track_title_to_datetime(ds_ml[antenna].StartTime)
        ds_ml[antenna]['Intensity_dB'] = seastar.utils.tools.lin2db(ds_ml[antenna].Intensity)
        ds_ml[antenna]['RadialSurfaceVelocity'] = seastar.oscar.level1.compute_radial_surface_velocity(ds_ml[antenna])
        
        
    ds_ml = seastar.oscar.level1.fill_missing_variables(ds_ml, antenna_list)
    
    #-----------------------------------------------------------
    
    # Building L1 dataset
    logger.info(f"Build L1 dataset for :  {track} of day: {acq_date}")
    
    ds_L1B = seastar.oscar.level1.merge_beams(ds_ml, antenna_list)
    del ds_ml   
    ds_L1B = ds_L1B.drop_vars(['LatImage', 'LonImage'], errors='ignore')

    #Updating of the CodeVersion in the attrs:
    ds_L1B.attrs["CodeVersion"] = __version__

    # Updating of the history in the attrs:
    current_history = ds_L1B.attrs.get("History", "")                                          # Get the current history or initialize it
    str_time = dt.now(timezone.utc).strftime('%d-%b-%Y %H:%M:%S')
    new_entry = f"{str_time} L1B processing."                                                  # Create a new history entry
    updated_history = f"{current_history}\n{new_entry}" if current_history else new_entry           # Append to the history
    ds_L1B.attrs["History"] = updated_history                                                       # Update the dataset attributes

    # Defining filename for datafile
    filename = seastar.oscar.tools.formatting_filename(ds_L1B)

    # Write the data in a NetCDF file
    if write_nc: 
        path_new_data = os.path.join(os.path.dirname(L1AP_folder).replace("L1AP", "L1B"), os.path.basename(L1AP_folder))
        if not os.path.exists(path_new_data):
            os.makedirs(path_new_data, exist_ok=True)
            logger.info(f"Created directory {path_new_data}")
        else: logger.info(f"Directory {path_new_data} already exists.")
        
        logger.info(f"Writing in {os.path.join(path_new_data, filename)}")
        ds_L1B.to_netcdf(os.path.join(path_new_data, filename)) 

    return ds_L1B


def apply_calibration(ds_L1B, ds_calibration, calib):
    """
    Apply calibration to L1B data.
    
    Applies calibration curves contained in a calibration dataset to an OSCAR
    L1B dataset. Option `calib` to switch between `Sigma0` or `Interferogram`
    calibration

    Parameters
    ----------
    ds_L1B : ``xr.Dataset``
        OSCAR L1B dataset containing `Sigma0` or `Interferogram` data
    ds_calibration : ``xr.Dataset``
        Calibration dataset
    calib : ``str``
        Option for `Sigma0` or `Interferogram` calibration. Case insensitive

    Raises
    ------
    Exception
        Raises an exception if `calib` is not in the form `Sigma0` or `Interferogram`

    Returns
    -------
    da_out : ``xr.DataArray``
        Calibrated L1B data of the variable chosen by `calib`
    CalImage : ``xr.DataArray``
        Calibration image of the variable chosen by `calib`

    """
    valid_calib = ['sigma0', 'interferogram']
    if calib.lower() not in valid_calib:
        raise Exception('Calibration option of ' + calib + ' is not valid. Please select from Sigma0 or Interferogram')

    if calib.lower() == 'sigma0':
        CalImage = xr.concat([xr.DataArray(data=np.interp(ds_L1B.IncidenceAngleImage.sel(Antenna=ant),
                                                     ds_calibration.IncidenceAngle.data,
                                                     ds_calibration.Sigma0.sel(Antenna=ant).data),
                                      coords=ds_L1B.Intensity.sel(Antenna=ant).coords,
                                    dims=ds_L1B.Intensity.sel(Antenna=ant).dims)
                               for ant in ds_L1B.Antenna], dim='Antenna', join='outer')
        CalImage.attrs['long_name'] = 'Sigma0 Calibration'
        CalImage.attrs['units'] = ''
        CalImage.attrs['description'] = 'Sigma0 bias with GMF from OceanPattern calibration in linear units '
        da_out = seastar.utils.tools.db2lin(seastar.utils.tools.lin2db(ds_L1B.Intensity) - CalImage)
        da_out.attrs['long_name'] = 'Sigma0'
        da_out.attrs['units'] = ''
        da_out.attrs['description'] = 'Calibrated NRCS using ' + ds_calibration.NRCSGMF + ' and over-ocean OSCAR data'
    elif calib.lower() == 'interferogram':
        if ds_calibration.Calibration == 'LandCalib':
            Interferogram_calib = ds_calibration.InterferogramSmoothed
        else:
            Interferogram_calib = ds_calibration.Interferogram
        CalImage = xr.concat([xr.DataArray(data=np.interp(ds_L1B.IncidenceAngleImage.sel(Antenna=ant),
                                                     ds_calibration.IncidenceAngle.data,
                                                     Interferogram_calib.sel(Antenna=ant).data),
                                      coords=ds_L1B.Interferogram.sel(Antenna=ant).coords,
                                    dims=ds_L1B.Interferogram.sel(Antenna=ant).dims)
                       for ant in ds_L1B.Antenna],
                      dim='Antenna',
                      join='outer')
        CalImage.attrs['long_name'] = 'Interferogram Calibration'
        CalImage.attrs['units'] = 'rad'
        CalImage.attrs['description'] = 'Interferogram bias from ' + ds_calibration.Calibration + ' in radians'
        da_out = ds_L1B.Interferogram - CalImage
        da_out.attrs['long_name'] = 'Interferogram'
        da_out.attrs['units'] = 'rad'
        da_out.attrs['description'] = 'Interferometric phase calibrated using ' + ds_calibration.Calibration + ' OSCAR data'
    
    return da_out, CalImage

def processing_OSCAR_L1B_to_L1C(L1B_folder, campaign, acq_date, track, calib_dict, write_nc=False):
    """
    L1B to L1C processing chain.
    
    Processes OSCAR L1B data to L1C by applying one or more calibration files.

    Parameters
    ----------
    L1B_folder : ``str``
        Path to folder on disk containing OSCAR L1B files to process
    campaign : ``str``
        OSCAR campaign name
    acq_date : ``str``
        Acquisition date of the data in the form YYYYMMDD
    track : ``str``
        Track name of data to process in the form of e.g., `Track_1`
    calib_dict : ``dict``
        Dict containing the following (name:content):
            {'Sigma0_calib_file': full filename including path for Sigma0 calib file,
             'Interferogram_calib_file': full filename including path for Interferogram calib file,
             }
    write_nc : ``bool``, optional
        Option to write L1C file to disk. The default is False.
    
    Raises
    ------
    Exception
        Raises an exception if not all required entries found in calib_dict

    Returns
    -------
    ds_L1C : ``xr.Dataset``
        Calibrated OSCAR L1C dataset

    """
    #Checking calib_dict
    valid_dict_keys = ['Sigma0_calib_file', 'Interferogram_calib_file']
    if not all([i in calib_dict.keys() for i in valid_dict_keys]):
        pattern = re.compile(r'^(' + '|'.join(map(re.escape, calib_dict.keys())) + r')$')
        missing_keys = [entry for entry in valid_dict_keys if not pattern.match(entry)]
        raise Exception(str(missing_keys) + ' missing from calib_dict')
    
    
    # checking acq_date format:
    if seastar.oscar.tools.is_valid_acq_date(acq_date):
        logger.info("'acq_date' format is okay")
    else: 
        logger.error("'acq_date' should be a valid date string in 'YYYYMMDD' format ")
        raise ValueError("'acq_date' should be a valid date string in 'YYYYMMDD' format ")

    # Checking campaign name:
    valid_campaign = ["202205_IroiseSea", "202305_MedSea"]
    if campaign in valid_campaign:
        logger.info("Campaign name has valid value.")
    else:
        logger.error(f"Unexpected campaign value: {campaign}")
        raise ValueError(f"Unexpected campaign value: {campaign}. Shall be {valid_campaign}.")
    

    # Getting the date for every tracks in the dict.
    track_names_dict = seastar.utils.readers.read_config_OSCAR('track', {"campaign" : campaign, "flight" : acq_date})  #read_OSCAR_track_names_config(campaign, acq_date)
    
    # Getting the date for the track we are interested in
    if track in track_names_dict.values():
        date_of_track = [key for key, v in track_names_dict.items() if v == track][0]
        logger.info(f"Track '{track}' found, acquisition time: {date_of_track}")
    else:
        logger.warning(f"Track '{track}' not found in track_names_dict. The code will crash.")
        
    #-----------------------------------------------------------
    #               L1C PROCESSING
    #-----------------------------------------------------------

    # Loading calib files
    Interferogram_calib_file = calib_dict.get('Interferogram_calib_file')
    Sigma0_calib_file = calib_dict.get('Sigma0_calib_file')
    
    logger.info(f"Loading Interferogram calibration file: {Interferogram_calib_file}")
    Interferogram_calib_file_path, Interferogram_calib_file_name = os.path.split(Interferogram_calib_file)
    ds_Interferogram_calib = xr.open_dataset(Interferogram_calib_file)
    logger.info(f"Loading Sigma0 calibration file: {Sigma0_calib_file}")
    Sigma0_calib_file_path, Sigma0_calib_file_name = os.path.split(Sigma0_calib_file)
    ds_Sigma0_calib = xr.open_dataset(Sigma0_calib_file)
    # Loading data
    logger.info(f"Opening track: {track} on day: {acq_date}")
    L1B_file_name = seastar.oscar.tools.find_file_by_track_name(os.listdir(L1B_folder), track=track)
    ds_L1B = xr.open_dataset(os.path.join(L1B_folder, L1B_file_name))
    ds_L1C = ds_L1B.copy(deep=True)
    ds_L1C.attrs['ProcessingLevel'] = 'L1C'
    
    # Radiometric Calibration
    logger.info(f"Calibrating Sigma0 for :  {track} of day: {acq_date}")
    ds_L1C['Sigma0'], ds_L1C['Sigma0CalImage'] = apply_calibration(ds_L1B, ds_Sigma0_calib, 'Sigma0')
    ds_L1C['Sigma0'].attrs['calibration_file_name'] = Sigma0_calib_file_name
    ds_L1C['Sigma0'].attrs['calibration_file_short_name'] = seastar.utils.readers.short_file_name_from_md5(
        seastar.utils.readers.md5_checksum_from_file(Sigma0_calib_file)
    )
    ds_L1C['Sigma0CalImage'].attrs['calibration_file_name'] = Sigma0_calib_file_name
    ds_L1C['Sigma0CalImage'].attrs['calibration_file_short_name'] = seastar.utils.readers.short_file_name_from_md5(
        seastar.utils.readers.md5_checksum_from_file(Sigma0_calib_file)
    )
    
    # Interferometric Calibration
    logger.info(f"Calibrating Interferogram for :  {track} of day: {acq_date}")
    ds_L1C['Interferogram'], ds_L1C['InterferogramCalImage'] = apply_calibration(ds_L1B, ds_Interferogram_calib, 'Interferogram')
    ds_L1C['Interferogram'].attrs['calibration_file_name'] = Interferogram_calib_file_name
    ds_L1C['Interferogram'].attrs['calibration_file_short_name'] = seastar.utils.readers.short_file_name_from_md5(
        seastar.utils.readers.md5_checksum_from_file(Interferogram_calib_file)
    )
    ds_L1C['InterferogramCalImage'].attrs['calibration_file_name'] = Interferogram_calib_file_name
    ds_L1C['InterferogramCalImage'].attrs['calibration_file_short_name'] = seastar.utils.readers.short_file_name_from_md5(
        seastar.utils.readers.md5_checksum_from_file(Interferogram_calib_file)
    )
    
    #Updating of the CodeVersion in the attrs:
    ds_L1C.attrs["CodeVersion"] = __version__
    
    # Updating of the history in the attrs:
    current_history = ds_L1C.attrs.get("History", "")                                               # Get the current history or initialize it
    new_entry = f"{dt.now(timezone.utc).strftime("%d-%b-%Y %H:%M:%S")} L1C processing."             # Create a new history entry
    updated_history = f"{current_history}\n{new_entry}" if current_history else new_entry           # Append to the history
    ds_L1C.attrs["History"] = updated_history                                                       # Update the dataset attributes
    ds_L1C.attrs['OceanPatternCalibrationFileName'] = Sigma0_calib_file_name
    ds_L1C.attrs['OceanPatternCalibrationFileShortName'] = seastar.utils.readers.short_file_name_from_md5(
        seastar.utils.readers.md5_checksum_from_file(Sigma0_calib_file)
    )
    ds_L1C.attrs['LandCalibFileName'] = Interferogram_calib_file_name
    ds_L1C.attrs['LandCalibFileShortName'] = seastar.utils.readers.short_file_name_from_md5(
        seastar.utils.readers.md5_checksum_from_file(Interferogram_calib_file)
    )
    ds_L1C.attrs['NRCSGMF'] = ds_Sigma0_calib.attrs['NRCSGMF']
    ds_L1C.attrs['Calibration'] = ' '.join(['NRCS calibrated using OceanPattern calibration. OceanPattern process uses data from a star pattern of multiple acquisitions taken at different headings',
    'relative to the wind direction. Median along-track data are then grouped by incidence angle and antenna look direction to produce data variables relative to azimuth for a discrete',
    'set of incidence angles. Curves are fitted to these points. Similar curves are generated using the NRCS GMF (',
                                           ds_Sigma0_calib.attrs['NRCSGMF'],
                                           ') and the difference',
    'between the fitted curves and the GMF averaged over azimuth are taken as the NRCS calibration bias for a given incidence angle. Interferograms calibrated with LandCalib. LandCalib',
    'process uses an OSCAR acquisition over land and finds land pixes using the GSHHS global coastline dataset to generate a land mask. Applying this land mask to OSCAR L1B Interferogram imagery',
    'the along-track median is taken to generate Interferogram bias wrt the cross range (incidence angle) dimension. These data are then smoothed with a polynomial fit to generate Interferogram',
    'bias curves for the Fore and Aft antenna directions (Mid is set to zero)'])

    filename = seastar.oscar.tools.formatting_filename(ds_L1C)

    # Write the data in a NetCDF file
    if write_nc: 
        path_new_data = os.path.join(os.path.dirname(L1B_folder).replace("L1B", "L1C"), os.path.basename(L1B_folder))
        if not os.path.exists(path_new_data):
            os.makedirs(path_new_data, exist_ok=True)
            logger.info(f"Created directory {path_new_data}")
        else: logger.info(f"Directory {path_new_data} already exists.")
        
        logger.info(f"Writing in {os.path.join(path_new_data, filename)}")
        ds_L1C.to_netcdf(os.path.join(path_new_data, filename)) 

    return ds_L1C
