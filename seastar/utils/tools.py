#!/usr/bin/env python
# coding=utf-8

import numpy as np
import xarray as xr

from scipy import interpolate
from collections import defaultdict
from os import listdir
import cartopy.feature as cfeature
from shapely.geometry import Point
from scipy.io import loadmat
from scipy.ndimage import binary_erosion as erode
# import seastar

def currentVelDir2UV(vel, cdir):
    """
    Compute current vector components from direction and magnitude.

    Parameters
    ----------
    vel : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        Magnitude of current (m/s).
    cdir : `float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        Direction of current (degrees N) in oceanographic convention.

    Returns
    -------
    u : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        u component (positive East) of current vector.
    v : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        v component (positive North) of current vector..

    """
    z = vel * np.exp(-1j * (cdir - 90) / 180 * np.pi)
    u = z.real
    v = z.imag
    return u, v


def currentUV2VelDir(u, v):
    """
    Convert u,v vector components to velocity and direction.

    Parameters
    ----------
    u : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        u vector component in positive East direction
    v : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        v vector component in positive North direction

    Returns
    -------
    vel : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        Magnitude of the converted vector with same units as `u` and `v`
    cdir : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        Direction of the converted vector (degrees)
    """
    tmp = u + 1j * v
    vel = np.abs(tmp)
    cdir = np.mod(90 - np.angle(tmp, deg=True), 360)

    return vel, cdir


def windSpeedDir2UV(wspd, wdir):
    """
    Convert wind speed and direction to u,v vector components.

    Parameters
    ----------
    wspd : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        Magnitude of wind (m/s).
    wdir : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        Direction of wind (degrees N) in wind convention.

    Returns
    -------
    u : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        u component (positive East) of wind vector (m/s)
    v : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        v component (positive North) of wind vector (m/s)

    Notes
    -----
    Wind input direction (``wdir``) is in wind convention format, i.e. the
    direction the wind is blowing from in degrees from North. A ``wdir``
    value of 0 corresponds to a wind blowing from North.
    """
    z = wspd * np.exp(-1j * (wdir + 90) / 180 * np.pi)
    u = z.real
    v = z.imag
    return u, v


def windUV2SpeedDir(u, v):
    """
    Convert wind u,v vector components to wind speed and direction.

    Parameters
    ----------
    u : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        u component (positive East) of wind vector (m/s)
    v : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        v component (positive North) of wind vector (m/s)

    Returns
    -------
    wspd : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        Magnitude of wind (m/s).
    wdir : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        Direction of wind (degrees N) in wind convention.

    Notes
    -----
    Wind input direction (``wdir``) is in wind convention format, i.e. the
    direction the wind is blowing from in degrees from North. A ``wdir``
    value of 0 corresponds to a wind blowing from North.
    """
    tmp = u + 1j * v
    wspd = np.abs(tmp)
    wdir = np.mod(-90 - np.angle(tmp) * 180 / np.pi, 360)
    return wspd, wdir

def windCurrentUV2all(mydict):
    """
    Convert a dictionary with ['u'], ['v'], ['c_u'], ['c_v'] elements to
    a full dictionary with vis_u, vis_v, vis_wspd, vis_wdir, c_vel, c_dir

    Parameters
    ----------
    mydict : dict or dotdict
        a dictionary with ['u'], ['v'], ['c_u'], ['c_v'] elements
    Returns
    -------
    mydict : dict or dotdict
        a dictionary with ['u'], ['v'], ['c_u'], ['c_v'] elements
    """

    mydict['vis_u'] = mydict['u'] - mydict['c_u']
    mydict['vis_v'] = mydict['v'] - mydict['c_v']
    mydict['vis_wspd'], mydict['vis_wdir'] = \
        windUV2SpeedDir(
            mydict['vis_u'], mydict['vis_v']
        )
    mydict['c_vel'], mydict['c_dir'] = \
        currentUV2VelDir(
            mydict['c_u'], mydict['c_v']
        )
    return mydict

def wavenumber2wavelength(wavenumber):
    """
    Convert wavenumber to wavelength.

    Parameters
    ----------
    wavenumber : ``float``
        Wavenumber (rad / m)

    Returns
    -------
    wavelength : ``float``
        Wavelength (m)

    """
    wavelength = 2 * np.pi / wavenumber
    return wavelength


def compute_relative_wind_direction(windDirection, lookDirection):
    """
    Compute relative wind direction.

    Compute angle between radar beam and wind direction (degrees)
    0 degrees = up-wind
    180 degrees = down-wind
    90, -90 degrees = cross-wind

    Parameters
    ----------
    windDirection : array, ``float``
        Wind direction (degrees N) in wind convention.
    lookDirection : array, ``float``
        Antenna look direction (degrees N) in oceanographic convention

    Returns
    -------
    relative_wind_direction : array, ``float``
        Angle between radar beam and wind direction (degrees)

    Notes
    -----
    Be aware of the difference in direction convention between the wind in
    ``windDirection`` and antenna look direction in ``lookDirection``.

    The wind direction is in wind convention (degrees N), i.e. the
    direction from where the wind is blowing: e.g., a wind direction of
    0 degrees corresponds to a wind blowing from the North.

    The antenna look direction is in oceanographic convention(degrees N),
    i.e., the direction to where the antenna is looking: e.g., a look direction
    of 0 degrees is looking North and with a wind direction of 0 degrees is
    looking Up-Wind.
    """
    relative_wind_direction = \
        np.abs(
            np.mod(
                windDirection - lookDirection + 180,
                360
            ) - 180
        )
    return relative_wind_direction


def import_incidence_angle_and_squint_to_dataset(filename,ds):
    data = loadmat(filename)
    data_vars = list(data.keys())
    for var_name in data_vars:
        var_data = data[var_name]
        if isinstance(var_data, np.ndarray):
            ds[var_name] = xr.DataArray(
                data=np.degrees(var_data),
                coords={'GroundRange': ds.GroundRange,
                        'CrossRange': ds.CrossRange},
                dims=('GroundRange', 'CrossRange')
                ).T


    return ds


def  wgs2utm_v3(lat, lon, utmzone, utmhemi):
    """
    Convert WGS84 coordinates into UTM coordinates.

    Parameters
    ----------
    lat : ``float``
        WGS84 Latitude scalar, vector or array in decimal degrees.
    lon : ``float``
        WGS84 Longitude scalar, vector or array in decimal degrees.
    utmzone : ``int``
        UTM longitudinal zone. Scalar or same size as ``Lat`` and ``Lon``.
    utmhemi : `str`
        UTM hemisphere as a single character, 'N' or 'S', or array of 'N' or
        'S' characters of same size as ``Lat`` and ``Lon``.

    Returns
    -------
    x : ``float``
        UTM Easting (m)
    y : ``float``
        UTM Northing (m)

    Notes
    -----
    Author: |br| Alexandre Schimel, MetOcean Solutions Ltd. New Zealand
    |br| Adapted to python by: |br| David McCann, National Oceanography Centre, U.K.
    |br| From the original author:
    |br| I downloaded and tried deg2utm.m from Rafael Palacios but found
    differences of up to 1m with my reference converters in southern
    hemisphere so I wrote my own code based on "Map Projections - A
    Working Manual" by J.P. Snyder (1987). Quick quality control performed
    only by comparing with LINZ converter and Chuck Taylor's on a
    few test points, so use results with caution. Equations not suitable
    for a latitude of +/- 90 deg.
    """
    lat = np.radians(lat)
    lon = np.radians(lon)
    a = 6378137
    b = 6356752.3142
    e = np.sqrt(1 - (b / a) ** 2)
    lon0 = 6 * utmzone - 183
    lon0 = np.radians(lon0)
    k0 = 0.9996
    FE = 500000
    FN = float(utmhemi == 'S')*10000000
    eps = e ** 2 / (1 - e ** 2)
    N = a / np.sqrt(1 - e ** 2 * np.sin(lat) ** 2)
    T = np.tan(lat) ** 2
    C = ((e ** 2) / (1 - e ** 2)) * (np.cos(lat)) ** 2
    A = (lon - lon0) * np.cos(lat)
    M = a * ((1 - e ** 2 / 4 - 3 * e ** 4/64 - 5 * e ** 6/256) *
             lat - (3 * e ** 2/8 + 3 * e ** 4/32 + 45 * e ** 6/1024) *
             np.sin(2 * lat) +
             (15 * e ** 4/256 + 45 * e ** 6/1024) * np.sin(4 * lat) -
             (35 * e ** 6/3072) * np.sin(6 * lat))
    x = FE + k0 * N * (A +
                       (1 - T + C) * A ** 3/6 +
                       (5 - 18 * T + T ** 2 + 72 * C - 58 * eps) *
                       A ** 5 / 120)
    y = FN + k0 * M + k0 * N * np.tan(lat) * (A ** 2 / 2 +
                                              (5 - T + 9 * C + 4 * C ** 2) *
                                              A ** 4 / 24 +
                                              (61 - 58 * T + T ** 2 + 600 * C -
                                               330 * eps) * A ** 6/720)
    return x, y


def identify_antenna_location(ds):
    """
    Identify ATI antenna location.

    Identifies the antenna direction in an OSCAR dataset by interrogating the
    minimum and maximum processed Doppler values.

    Parameters
    ----------
    ds : ``xarray.Dataset``
        OSCAR dataset containing MinProcessedDoppler and MaxProcessedDoppler
        variables

    Returns
    -------
    antenna_location : ``str``
        Antenna location ('Fore', 'Aft', 'Mid')

    """
    doppler_mean = np.mean([ds.MinProcessedDoppler, ds.MaxProcessedDoppler])
    if np.abs(doppler_mean) < 500:
        antenna_location = 'Mid'
    elif doppler_mean < 0:
        antenna_location = 'Aft'
    elif doppler_mean > 0:
        antenna_location = 'Fore'

    return antenna_location


def list_duplicates(seq):
    """
    Find indices of duplicate entries in list.

    Finds the indicies of duplicate list entries and returns a generator object
    that can be sorted to create a list of [(seq, index)].

    Suggested use: sorted(list_duplicates(seq))

    Parameters
    ----------
    seq : ``list``
        DESCRIPTION.

    Returns
    -------
    key : same as in `seq`
        List of entries in `seq`
    locs : ``list`` of ``int``
        List of indexes of duplicate entries of `key` in `seq`

    """
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return ((key, locs) for key, locs in tally.items()
            if len(locs) > 1)


def find_file_triplets(file_path):
    """
    Find file list indices of matching antenna files.

    Sorts a file list generated by `os.listdir(file_path)` and creates a list
    `file_time_triplets` of file aquisition times (YYYYMMDD'T'HHMMSS) along the
    1st column and the corresponding indices of the Fore/Mid/Aft antenna
    data files associated with each aquisition time in the 2nd column.

    Parameters
    ----------
    file_path : ``str``
        Path containing OSCAR NetCDF files

    Returns
    -------
    file_time_triplets : ``list``
        List of times of data aquisition and the corresponding indices of
        matching files in the Fore/Mid/Aft antenna triplet in the file list
        generated by `os.listdir(file_path)`
    """
    file_list = listdir(file_path)
    num_files = len(file_list)
    file_time = list()
    for file in range(num_files):
        file_info = str.split(file_list[file], '_')
        file_time.append(file_info[2])
    file_time_triplets = sorted(list_duplicates(file_time))
    return file_time_triplets


def antenna_idents(ds):
    """
    Build antenna identifier list.

    Generates list of antenna identifiers of the form ['Fore','Mid','Aft'] to
    correspond with keys in the dataset dictionary generated by
    `seastar.utils.readers.load_OSCAR_data`

    Parameters
    ----------
    ds : ``dict``
        OSCAR data stored as a dict with antenna number as keys and loaded
        data in ``xarray.Dataset`` format as values

    Returns
    -------
    antenna_ident : ``list``
        List of antenna identifiers in the form ['Fore', 'Mid', 'Aft'],
        corresponding to the data and keys stored in `ds`

    """
    antenna_id = list()
    for i in list(ds.keys()):
        
        antenna_id.append(identify_antenna_location(ds[i]))
    return antenna_id

def identify_antenna_location_from_filename(file_path, file_time_triplets):
    antenna_identifiers = {'0': 'Mid', '3': 'Fore', '7': 'Aft'}
    file_list = listdir(file_path)
    antenna_id = list()
    for i in range(len(file_time_triplets)):
        file_name = file_list[file_time_triplets[i]]
        antenna_id.append(antenna_identifiers[file_name.split('_')[5][0]])
    return antenna_id


def find_coincident_looks(ds_l1_star, star_pattern_tracks, file_time_triplets,
                          rounding=10, d_precision=5):
    """
    Find coincident looks in star pattern.

    Function to find the file list indices of antenna datasets containing
    data in coincident look directions within the SEASTARex star pattern
    on May 22nd 2022.

    Parameters
    ----------
    ds_l1_star : ``dict``
        Dictionary of antenna L1 datasets.
    star_pattern_tracks : ``dict``
        Dictionary of track names and their associated file indices
    rounding : ``int``, optional
        Look direction rounding number in degrees. The default is 10.
    d_precision : ``int``, optional
        Precision of look direction comparison in degrees. The default is 5.

    Returns
    -------
    look_files : ``dict``
        Dictionary of look direction (degrees) in the star pattern as keys and
        matching tuples of (Track, Antenna)

    """
    antennas = {'Fore': 0, 'Mid': 1, 'Aft': 2}
    looks = np.zeros((len(star_pattern_tracks.keys()), 3))
    for track in star_pattern_tracks.keys():
        look_fore = np.round(
            np.mean(ds_l1_star[track]
                    .sel(Antenna='Fore')
                    .AntennaAzimuthImage)
            .data / rounding) * rounding
        look_mid = np.round(
            np.mean(ds_l1_star[track]
                    .sel(Antenna='Mid')
                    .AntennaAzimuthImage)
            .data / rounding) * rounding
        look_aft = np.round(
            np.mean(ds_l1_star[track]
                    .sel(Antenna='Aft')
                    .AntennaAzimuthImage)
            .data / rounding) * rounding
        looks[list(star_pattern_tracks.values())
              .index(star_pattern_tracks[track]),
              :] = [look_fore, look_mid, look_aft]
    look_files = dict()
    for d in range(rounding, 360 + rounding, rounding):
        look_range = np.abs(looks - d) <= d_precision
        if np.any(look_range) and np.count_nonzero(look_range) > 1:
            antenna_info = list()
            for i in range(len(np.where(look_range)[0])):
                ii = np.where(look_range)[0][i]
                jj = np.where(look_range)[1][i]
                antenna_info.append(
                    (list(star_pattern_tracks.keys())[ii],
                     list(antennas.keys())[jj])
                    )
            look_files[d] = antenna_info
    return look_files


def polarizationStr2Val(da):
    '''
    Transform Polarization string ('VV' or 'HH') towards values (1, 2)

    Parameters
    ----------
    da: `DataArray`
    Returns
    -------
    out: `DataArray`
    '''

    keys = {'VV': 1, 'HH': 2}
    data = np.vectorize(keys.get)(da.data)
    out = xr.DataArray(
        data=data,
        dims=da.dims,
    )

    return out

def polarizationVal2Str(da):
    '''
    Transform Polarization values (1, 2) to string ('VV' or 'HH')

    Parameters
    ----------
    da: `DataArray`
    Returns
    -------
    out: `DataArray`
    '''

    keys = {1: 'VV', 2: 'HH'}
    data = np.vectorize(keys.get)(da.data)
    out = xr.DataArray(
        data=data,
        dims=da.dims,
    )

    return out


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def compute_land_mask_from_GSHHS(da, boundary=None, skip=1/1000, erosion=False,
                                      erode_scale = 3, coastline_selection=0,):
    """
    Compute land mask.

    Computes a boolean mask with size, coords and dims corresponding to a
    supplied xarray.DataArray, with mask values = True where the point lies
    within the coastline from the GSHHS global coastline dataset.

    Parameters
    ----------
    da : ``xarray.DataArray``
        xarray.DataArray to compute the land mask for. Must contain
        `longitude` and `latitude` coordinates as well as
        coords and dims to align the new `mask` to.
    boundary : ``list``, optional
        Optional boundary to check for the presence of coastlines in the GSHHS
        dataset. Supply `boundary` in the form:
        `[min(long), max(long), min(lat), max(lat)]`.
        The default boundary will be set to the minimum and maximum extent of
        the `longitude` and `latitude` data present in the coords of `da`.
    skip : ``float``, optional
        Speed-up factor in degrees longitude / latitude. The default is 1/1000.
        Lower than 1/250 resolution will result in a coarsening of the computed
        `mask`.
    erosion : ``bool``, optional
        Boolean switch to trigger binary erosion of the resulting `mask`.
        The default is False.
    erode_scale : ``int``, ``float``, optional
        Scale for the array-like structure used for optional binary erosion.
        When default is used but `erosion=True` then a
        `erode_scale` of 3 is assumed.
    coastline_selection : ``int``, ``list``, optional
        Manual choice of which identified coastlines present within the 
        `boundary` are used in the generation of the `mask`. 
        Choice is a key or list of keys of type ``int``. The default is 0. 
        The default behaviour is the first identified coastline within 
        `boundary` is used to generate the `mask`, corresponding to the largest
        coastline within the `boundary` by internal area.

    Raises
    ------
    Exception
        'longitude and latitude missing from input dataset'.
        'coastline_selection out of bounds of identified coastlines within
        boundary'

    Returns
    -------
    mask : ``bool``, ``array``, ``xr.DataArray``
        An xr.DataArray containing an array-like boolean mask of land pixels.
    """
    if 'longitude' not in da.coords or 'latitude' not in da.coords:
        raise Exception('longitude and latitude missing from input dataset')
    if not boundary:
        boundary = [np.min(da.longitude.data),
                    np.max(da.longitude.data),
                    np.min(da.latitude.data),
                    np.max(da.latitude.data)]
    lon_skip, lat_skip = np.meshgrid(np.arange(boundary[0], boundary[1], skip),
                                 np.arange(boundary[2], boundary[3], skip))
    if erosion:
        erode_scale = int(np.round(erode_scale))
        erode_structure = np.full((erode_scale, erode_scale), True)
    coast_polygons = dict()
    m, n = lon_skip.shape
    mask = np.full((m, n), False)
    print('Scanning GSHHS dataset for coastlines within boundary...')
    coast = cfeature.GSHHSFeature(scale='full')\
        .intersecting_geometries(boundary)
    for k, polygon in enumerate(coast):
        coast_polygons[k] = polygon
    if bool(coast_polygons):
        print(coast_polygons)
        if type(coastline_selection) is int:
            coastline_selection = [coastline_selection]
        if np.max(coastline_selection) > k:
            raise Exception('Selected coastline(s)',
                           coastline_selection,
                           ' different to coastlines identified within boundary ',
                           list(coast_polygons.keys()),
                           '. Please try a different coastline_selection (default=0)'
                           )
        coast_polygons = {key: coast_polygons[key]
                         for key in coastline_selection}
    
        
        count = 0
        print('Performing search...')
        for i in range(m):
            for j in range(n):
                count = count + 1
                if not int(np.mod(count, ((m*n) / 10))):
                    print(int((count / (m*n)) * 100), '% complete')
                for k in coast_polygons.keys():
                    mask[i, j] = mask[i, j] or\
                       Point(lon_skip[i, j], lat_skip[i, j])\
                       .within(coast_polygons[k])
        print('...done')
    if erosion:
        mask = erode(mask, structure=erode_structure)
    mask = xr.DataArray(data=mask.astype(int))
    mask = mask.assign_coords(longitude=(['dim_0', 'dim_1'], lon_skip),
                   latitude=(['dim_0', 'dim_1'], lat_skip),
                   )

    new_data = interpolate.griddata(
                            points=(np.ravel(lon_skip),
                                    np.ravel(lat_skip)),
                            values=(np.ravel(mask)),
                            xi=(da.longitude.values,
                                da.latitude.values),
                            method='nearest'
                            )
    mask = xr.DataArray(data=new_data.T,
                        dims=da.dims,
                        coords=da.coords)
    return mask
