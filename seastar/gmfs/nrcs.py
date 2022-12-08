#!/usr/bin/env python
# coding=utf-8

import numpy as np
import xarray as xr
from scipy.interpolate import interpn
import seastar


def compute_nrcs(L1_combined, aux_geo, gmf):
    """
    Compute Normalized Radar Cross Section (nrcs).

    Compute NRCS (Sigma0) using incidence angle, antenna polarization and
    windspeed/direction data.

    Parameters
    ----------
    L1_combined : ``xr.Dataset``
        Dataset containing IncidenceAngleImage and antenna polarization data
    aux_geo : ``xr.Dataset``
        Geophysical parameter dataset containing wind speed and direction data

    Returns
    -------
    nrcs : ``xr.Dataset``
        Dataset of computed Normalized Radar Cross Section DataArrays,
        arranged along a dimension corresponding to antenna position

    """
    nrcs = xr.Dataset()
    for antenna in L1_combined.Antenna.data:
        L1 = L1_combined.sel(Antenna=antenna)
        L1, aux_geo = xr.align(L1, aux_geo, join="outer")
        relative_wind_direction =\
            seastar.utils.tools.compute_relative_wind_direction(
                aux_geo.WindDirection,
                L1.AntennaAzimuthImage
                )
        # ind = {'VV': 1, 'HH': 2}
        # pol_val = np.full(L1.IncidenceAngleImage.values.shape,
        #                   ind[str(L1
        #                           .Polarization.data)]
        #                   )
        pol_val = seastar.utils.tools.polarizationStr2Val(L1.Polarization)

        if gmf.name == 'nscat4ds':
            nrcs[antenna] = xr.DataArray(
                nscat4ds(
                    aux_geo.WindSpeed.values,
                    relative_wind_direction.values,
                    L1.IncidenceAngleImage.values,
                    pol_val
                ),
                coords=L1.IncidenceAngleImage.coords,
                dims=L1.IncidenceAngleImage.dims
            )
        else:
            raise Exception('Error, unknown gmf, gmf.name should be nscat4ds')

    nrcs = nrcs.to_array(dim='Antenna')
    nrcs.attrs['long_name'] = 'Normalized radar cross section Sigma 0'
    nrcs.attrs['units'] = ['']
    return nrcs

def nscat4ds(u10, phi, inc, pol):
    """
    Compute sigma0 due to wind.
    admartin@noc.ac.uk adapted from
    (c) 2017 Anton Verhoef, KNMI Royal Netherlands Meteorological Institute

    Input parameters u10, phi, inc and pol must be in
    identical sizes and formats.

    Parameters
    ----------
    u10 : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        Wind speed (m/s) at 10m above sea surface.
        GMF built for wind speed ranging from 0.2 m/s to 50 m/s.
    phi : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        Angle between wind and look directions (degrees) in range 0 (upwind) :
            90 (crosswind) : 180 (downwind).
    inc : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        Incidence angle of radar beam (degrees from nadir).
        GMF built for incidence angle ranging from 16° to 66°.
    pol : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        Polarisation of radar beam (1 for VV, 2 for HH).
        Any value between 1 and 2 will be a linear interpolation

    Raises
    ------
    Exception
        Exception for inconsistency in sizes of input parameters.
        Exception for polarisation out of range of (1,2) for (VV,HH).

    Returns
    -------
    sigma0 : ``float``, ``numpy.array``, ``xarray.DataArray``
        NRCS due to geophysical and geometric conditions of
        the same size as the input.

    """
    # Check inputs
    sizes = np.array([np.size(inc), np.size(u10), np.size(phi)])
    size = sizes.max()
    if ((sizes != size) & (sizes != 1)).any():
        raise Exception('Inputs sizes do not agree.')
    for ii in np.unique(pol):
        if ii not in [1, 2]:
            raise Exception('Polarisation should be 1 (VV) or 2 (HH), '
                        'other values will be a linear interpolation '
                        'and is not allowed yet. Given polarisation: ' + pol)
    varin = np.stack([u10, phi, inc, pol],axis=-1)

    # Construct Path
    dirpath = seastar.gmfs.nrcs.__file__[:-7]
    # fname_big = 'nscat4ds_250_73_51_vv.dat_big_endian'
    fname_vv = dirpath + 'nscat4ds_250_73_51_vv.dat_little_endian'
    fname_hh = dirpath + 'nscat4ds_250_73_51_hh.dat_little_endian'

    # Dimensions of GMF table
    m = 250  # wind speed min/max = 0.2-50 (step 0.2) [m/s] --> 250 pts
    n = 73   # dir min/max = 0-180 (step 2.5) [deg]   -->  73 pts
    p = 51   # inc min/max = 16-66 (step 1) [deg]     -->  51 pts
    q = 2    # polarisation = 1, 2 --> 2 pts

    wspd = np.linspace(0.2, 50, m)
    rdir = np.linspace(0, 180, n)
    inci = np.linspace(16, 66, p)
    pol  = np.linspace(1, 2, q)

    gmf_table_vv = np.fromfile(fname_vv, dtype=np.float32)
    gmf_table_hh = np.fromfile(fname_hh, dtype=np.float32)

    # Remove head and tail
    gmf_table_vv = gmf_table_vv[1:-1]
    gmf_table_hh = gmf_table_hh[1:-1]

    # To access the table as a three-dimensional Fortran-ordered m x n x p matrix,
    # reshape it
    gmf_table_vv = gmf_table_vv.reshape((m, n, p), order="F")
    gmf_table_hh = gmf_table_hh.reshape((m, n, p), order="F")

    gmf_table = np.stack([gmf_table_vv, gmf_table_hh], axis=-1)

    points = (wspd, rdir, inci, pol)

    s0 = interpn(points, gmf_table, varin,
                method='linear',
                bounds_error=False,
                fill_value=np.nan)

    if isinstance(u10, xr.core.dataarray.DataArray):
        s0 = xr.DataArray(
            data=s0,
            dims=u10.dims,
        )
    return(
        s0
    )

