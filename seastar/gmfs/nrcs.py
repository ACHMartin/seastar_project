#!/usr/bin/env python
# coding=utf-8
"""Functions to compute Normalised Radar Cross Section data."""
import numpy as np
import xarray as xr
from scipy.interpolate import interpn
import seastar
from os.path import abspath, dirname, join


def _load_gmf_table(fname, shape):
    """Loads and reshapes a GMF table from a binary file."""
    gmf_table = np.fromfile(fname, dtype=np.float32)
    # Remove head and tail
    # To access the table as a three-dimensional Fortran-ordered m x n x p matrix (shape),
    # reshape it
    gmf_table = gmf_table[1:-1].reshape(shape, order="F")
    return gmf_table

def get_gmf_table_and_points(gmf_model):
    """
    Load the GMF table and corresponding points for a given GMF model.

    Parameters
    ----------
    gmf_model : str
        The GMF model to load. Supported options are 'nscat4ds' or 'cmod7'.

    Returns
    -------
    tuple
        A tuple containing:
        
        - numpy.ndarray: The GMF table.
        - tuple of numpy.ndarray: The corresponding points, which include arrays for
        `wspd`, `rdir`, `inci`, and optionally `pol`.
    """
    dirpath = abspath(dirname(__file__))

    # Common dimensions of GMF table
    m = 250  # wind speed min/max = 0.2-50 (step 0.2) [m/s] --> 250 pts
    n = 73   # dir min/max = 0-180 (step 2.5) [deg]   -->  73 pts
    p = 51   # inc min/max = 16-66 (step 1) [deg]     -->  51 pts

    wspd = np.linspace(0.2, 50, m)
    rdir = np.linspace(0, 180, n)
    inci = np.linspace(16, 66, p)

    if gmf_model == 'nscat4ds':
        q = 2 # polarisation = 1, 2 --> 2 pts
        pol = np.linspace(1, 2, q)
        fname_vv = join(dirpath, 'nscat4ds_250_73_51_vv.dat_little_endian')
        fname_hh = join(dirpath, 'nscat4ds_250_73_51_hh.dat_little_endian')
        gmf_table_vv = _load_gmf_table(fname_vv, (m, n, p))
        gmf_table_hh = _load_gmf_table(fname_hh, (m, n, p))
        gmf_table = np.stack([gmf_table_vv, gmf_table_hh], axis=-1)
        points = (wspd, rdir, inci, pol)
    elif gmf_model == 'cmod7':
        fname_vv = join(dirpath, 'cmod7_vv.dat_little_endian')
        gmf_table = _load_gmf_table(fname_vv, (m, n, p))
        points = (wspd, rdir, inci)
    else:
        raise ValueError(f"Unknown GMF model: {gmf_model}. Available models are 'nscat4ds' and 'cmod7'.")

    return gmf_table, points

def compute_nrcs(L1_combined, aux_geo, gmf):
    """
    Compute Normalized Radar Cross Section (nrcs).

    Compute NRCS (Sigma0) using incidence angle, antenna polarization and
    OceanSurfaceWindSpeed/direction data.

    Parameters
    ----------
    L1_combined : ``xr.Dataset``
        Dataset containing IncidenceAngleImage and antenna polarization data
    aux_geo : ``xr.Dataset``
        Geophysical parameter dataset containing OceanSurfaceWindSpeed and
        Direction data

    Returns
    -------
    nrcs : ``xr.Dataset``
        Dataset of computed Normalized Radar Cross Section DataArrays,
        arranged along a dimension corresponding to antenna position

    """
    # Initialisation / test
    if 'OceanSurfaceWindSpeed' not in aux_geo.keys():
        import logging
        aux_geo['OceanSurfaceWindSpeed'] = aux_geo['WindSpeed']
        aux_geo['OceanSurfaceWindDirection'] = aux_geo['WindDirection']
        logging.warning('"WindSpeed" and "WindDirection" fields are deprecated.'
                        'You should use "OceanSurfaceWindSpeed" and "OceanSurfaceWindDirection" '
                        'instead in order to remove this warning.\n'
                        '"Wind" are been used here as "OceanSurfaceWind" i.e. relative to the surface motion'
                        )

    nrcs = xr.Dataset()
    for antenna in L1_combined.Antenna.data:
        L1 = L1_combined.sel(Antenna=antenna)
        L1, aux_geo = xr.align(L1, aux_geo, join="outer")
        relative_wind_direction =\
            seastar.utils.tools.compute_relative_wind_direction(
                aux_geo.OceanSurfaceWindDirection,
                L1.AntennaAzimuthImage
                )
        # ind = {'VV': 1, 'HH': 2}
        # pol_val = np.full(L1.IncidenceAngleImage.values.shape,
        #                   ind[str(L1
        #                           .Polarization.data)]
        #                   )
        pol_val = seastar.utils.tools.polarizationStr2Val(L1.Polarization)

        if gmf['name'] == 'nscat4ds':
            nrcs_data = nscat4ds(
                aux_geo.OceanSurfaceWindSpeed.values,
                relative_wind_direction.values,
                L1.IncidenceAngleImage.values,
                pol_val.values
            )
        elif gmf['name'] == 'cmod7':
            # check pol_val are only in VV (values of 1)
            unique_pol = np.unique(pol_val.values)
            # more than one value for pol or not 1 # VV
            if len(unique_pol) > 1 or not (unique_pol == np.ones(unique_pol.shape)).all():
                raise Exception('Error, cmod7 gmf only accept VV polarisation with values of "1"')
            # compute with cmod7
            nrcs_data = cmod7(
                aux_geo.OceanSurfaceWindSpeed.values,
                relative_wind_direction.values,
                L1.IncidenceAngleImage.values,
                1 # VV
            )

        else:
            raise Exception('Error, unknown gmf, gmf["name"] should be nscat4ds or cmod7')

        if len(nrcs_data) == 1: # bug if nrcs_data is of len = 1 for DataArray creation
            nrcs_data = nrcs_data[0]

        nrcs[antenna] = xr.DataArray(
            data=nrcs_data,
            coords=L1.IncidenceAngleImage.coords,
            dims=L1.IncidenceAngleImage.dims
        )

    nrcs = nrcs.to_array(dim='Antenna')
    nrcs.attrs['long_name'] = 'Normalized radar cross section Sigma 0'
    nrcs.attrs['units'] = ['']
    return nrcs

gmf_table_global_nscat4ds, points_global_nscat4ds = get_gmf_table_and_points('nscat4ds')

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

    gmf_table, points = gmf_table_global_nscat4ds, points_global_nscat4ds

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

gmf_table_global_cmod7, points_global_cmod7 = get_gmf_table_and_points('cmod7')

def cmod7(u10, phi, inc, pol=1):
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
        Polarisation of radar beam (1 for VV). No HH polarisation.

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
        if ii not in [1]:
            raise Exception('Polarisation should be 1 (VV),'
                        'CMOD7 does not accept HH pol.'
                        'Given polarisation: ' + pol)
    varin = np.stack([u10, phi, inc],axis=-1)

    gmf_table, points = gmf_table_global_cmod7, points_global_cmod7

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
