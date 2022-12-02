#!/usr/bin/env python
# coding=utf-8

import numpy as np
import xarray as xr

from scipy import interpolate
from collections import defaultdict
from os import listdir

def currentVelDir2UV(vel, cdir):
    """
    Compute current vector components from direction and magnitude.

    Parameters
    ----------
    vel : array, float
        Magnitude of current (m/s).
    cdir : array, float
        Direction of current (degrees N) in oceanographic convention.

    Returns
    -------
    u : array, float
        u component (positive East) of current vector.
    v : array, float
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
    u : ``float``
        u vector component in positive East direction
    v : ``float``
        v vector component in positive North direction

    Returns
    -------
    vel : ``float``
        Magnitude of the converted vector with same units as `u` and `v`
    cdir : ``float``
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
    wspd : array, ``float``
        Magnitude of wind (m/s).
    wdir : ``float``
        Direction of wind (degrees N) in wind convention.

    Returns
    -------
    u : array, ``float``
        u component (positive East) of wind vector (m/s)
    v : array, ``float``
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
    u : array, ``float``
        u component (positive East) of wind vector (m/s)
    v : array, ``float``
        v component (positive North) of wind vector (m/s)

    Returns
    -------
    wspd : array, ``float``
        Magnitude of wind (m/s).
    wdir : ``float``
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
