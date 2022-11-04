#!/usr/bin/env python
# coding=utf-8

import numpy as np
import xarray as xr
from scipy.io import loadmat
from scipy import interpolate

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


def colocate_xband_marine_radar_data(filename, dsl2):
    """
    Colocate X-band data from matlab to SAR lat/long.

    Parameters
    ----------
    filename : ``str``
        Filename of the X-band matlab .mat data file
    dsl2 : ``xarray.Dataset``
        Dataset containing coordinates and dimensions to colocate to

    Raises
    ------
    Exception
        Exeption raised if ``latitude`` and ``longitude`` variables are not
        present in the radar .mat file

    Returns
    -------
    ds_out : ``xarray.Dataset``
        Dataset containing colocated X-band radar data

    Notes
    -----
    This function written to be as agnostic as possible but is designed
    primarily to colocate X-band radar data as supplied for the SEASTARex
    project.

    """
    ds_out = xr.Dataset()
    data = loadmat(filename)
    data_vars = list(data.keys())
    if 'longitude' in data_vars and 'latitude' in data_vars:
        for var_name in data_vars:
            var_data = data[var_name]
            if var_name in ['__header__', '__version__', '__globals__']:
                ds_out.attrs[var_name] = var_data
            if isinstance(var_data, np.ndarray):
                if var_data.shape == data['longitude'].shape:
                    ds_out[var_name] = xr.DataArray(
                        data=interpolate.griddata(
                            points=(np.ravel(data['longitude']),
                                    np.ravel(data['latitude'])),
                            values=(np.ravel(var_data)),
                            xi=(dsl2.longitude.values,
                                dsl2.latitude.values)
                            ),
                        dims=dsl2.dims,
                        coords=dsl2.coords
                        )
                elif var_data.shape == (1, 1):
                    ds_out[var_name] = float(var_data)
    else:
        raise Exception(
            'longitude and latitude not present in Xband .mat file'
            )
    ds_out.coords['longitude'] = dsl2.longitude
    ds_out.coords['latitude'] = dsl2.latitude
    return ds_out


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
