#!/usr/bin/env python
# coding=utf-8
"""Module to compute Doppler shift variables from airborne SAR imagery."""

import numpy as np
import pandas as pd
import xarray as xr
# import pdb
from scipy.constants import c  # speed of light in vacuum
import seastar
from typing import Union, Optional
from os.path import abspath, dirname, join
from numpy.polynomial import Legendre as L

def compute_total_surface_motion(L1, aux_geo, gmf, **kwargs):
    """
    Compute the total surface motion (or RVL, ...? TODO need to agree on the naming)
    including Wind-wave Artefact Surface Velocity (WASV)
    + surface current.

    Parameters
    ----------
    L1 : xarray.Dataset
        A Dataset containing IncidenceAngleImage, AntennaAzimuthImage and
        Polarization.
    aux_geo : xarray.Dataset
        A Dataset containing OceanSurfaceWindSpeed, OceanSurfaceWindDirection, CurrentVelocity, CurrentDirection
    gmf : str
        Option for geophysical model function. 'mouche12', 'yurovsky19'
        with kwargs for options.
    **kwargs : TYPE
        Optional arguments.

    Raises
    ------
    Exception
        Exception raised if LookDirection is not in the form of a 2D array.

    Returns
    -------
    da : ``xarray.DataArray``
        A DataArray containing the Radial Velocity for
        the given geophysical and geometric conditions.
    """

    wasv = seastar.gmfs.doppler.compute_wasv(L1, aux_geo, gmf, **kwargs)

    # TODO test aux_geo and L1 are aligned

    relative_current_direction = np.mod(aux_geo.CurrentDirection - L1.AntennaAzimuthImage, 360)
    # relative_direction_influence = np.cos(np.radians(relative_current_direction))
    # radial_current = aux_geo.CurrentVelocity * relative_direction_influence
    u_rsc, u_r = compute_radial_current(aux_geo.CurrentVelocity,
                                        relative_current_direction,
                                        L1.IncidenceAngleImage)

    da = xr.DataArray(
        data=wasv + u_rsc,
        dims=wasv.dims,
    )

    return da

def compute_radial_current(current_vel, rel_current_dir, inci):
    """
    Compute Radial and Radial Surface Current from known Current and acquisition geometry.

    Input parameters u10, phi and inc must be in identical sizes and formats.

    Parameters
    ----------
    current_vel : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        Current Velocity (m/s)
    rel_current_dir : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        Angle between current and look directions (degrees)
    inci : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        Incidence angle of radar beam (degrees from nadir).

    Returns
    -------
    u_rsc : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        Radial Surface Current (m/s)
    u_r : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        Radial Current (m/s) in the slant direction directly towards the radar.

        """
    relative_direction_influence = np.cos(np.radians(rel_current_dir))
    radial_surface_current = current_vel * relative_direction_influence
    radial_current = radial_surface_current * np.sin(np.radians(inci))
    return radial_surface_current, radial_current



def compute_wasv(L1, aux_geo, gmf, **kwargs):

    """
    Compute the Wind-wave Artefact Surface Velocity (WASV).

    Compute the Wind-wave Artefact Surface Velocity (WASV) for 'L1' data
    as function of geophysical conditions.

    Parameters
    ----------
    L1 : xarray.Dataset
        A Dataset containing IncidenceAngleImage, AntennaAzimuthImage and
        Polarization
    aux_geo : xarray.Dataset
        A Dataset containing OceanSurfaceWindSpeed, OceanSurfaceWindDirection.
    gmf : str
        Option for geophysical model function. 'mouche12', 'yurovsky19'
        with kwargs for options.
    **kwargs : TYPE
        Optional arguments.

    Raises
    ------
    Exception
        Exception raised if LookDirection is not in the form of a 2D array.

    Returns
    -------
    ds_wa : xarray.Dataset
        A Dataset containing the Wind Artefact Surface Velocity (WASV) for
        the given geophysical and geometric conditions.
    """

    # Initialisation
    if len(L1.AntennaAzimuthImage.shape) > 2:
        raise Exception('L1.AntennaAzimuthImage need to be a 2D field. \n'
                        'Use e.g. L1.sel(Antenna="Fore") to reduce to '
                        'a 2D field.')
    # Aligned L1 and aux_geo dataset => 'outer' Union of the two indexes
    # TODO I really think we might be able just to test L1 and geo are aligned and whatever the dimension
    L1, aux_geo = xr.align(L1, aux_geo, join="outer")

    if 'OceanSurfaceWindSpeed' not in aux_geo.keys():
        import logging
        aux_geo['OceanSurfaceWindSpeed'] = aux_geo['WindSpeed']
        aux_geo['OceanSurfaceWindDirection'] = aux_geo['WindDirection']
        logging.warning('"WindSpeed" and "WindDirection" fields are deprecated. '
                        'You should use "OceanSurfaceWindSpeed" and "OceanSurfaceWindDirection" '
                        'instead in order to remove this warning.\n'
                        '"Wind" are been used here as "OceanSurfaceWind" i.e. relative to the surface motion'
                        )

    relative_wind_direction =\
        seastar.utils.tools.compute_relative_wind_direction(
            aux_geo.OceanSurfaceWindDirection,
            L1.AntennaAzimuthImage
        )

    ind = dict()
    #if L1.Polarization.size > 1:
    for pol_str in ['VV', 'HH']:
        ind[pol_str] = (L1.Polarization == pol_str).values

    if gmf == 'mouche12':
        if L1.Polarization.size == 1:
            dop_c = mouche12(
                aux_geo.OceanSurfaceWindSpeed.values,
                relative_wind_direction.values,
                L1.IncidenceAngleImage.values,
                L1.Polarization.data,
            )
        else:
            dop_c = np.full(L1.IncidenceAngleImage.shape, np.nan)
            for pol_str in ('VV', 'HH'):
                if ind[pol_str].any():
                    dop_c[ind[pol_str]] = mouche12(
                        aux_geo.OceanSurfaceWindSpeed.values[ind[pol_str]],
                        relative_wind_direction.values[ind[pol_str]],
                        L1.IncidenceAngleImage.values[ind[pol_str]],
                        pol_str,
                    )


        # Convert Doppler Shift of C-band (5.5 GHz) to CentralFrequency
        f_c = 5.5 * 10 ** 9
        dop_Hz = dop_c * L1.CentralFreq.data / f_c
        [wasv_losv, wasv_rsv] = convertDoppler2Velocity(
            L1.CentralFreq.data / 1e9,
            dop_Hz,
            L1.IncidenceAngleImage
        )

    elif gmf == 'yurovsky19':
        if 'CentralWavenumber' not in L1:
            L1 = seastar.level1.add_central_electromagnetic_wavenumber(L1) # TODO should return a dataArray not the dataSet, should directly calculate the wavelength
        central_wavelength = seastar.utils.tools.wavenumber2wavelength(
            L1.CentralWavenumber
        )
        dc = dict()
        [dc['VV'], dc['HH']] = yurovsky19(
            L1.IncidenceAngleImage,
            relative_wind_direction,
            aux_geo.OceanSurfaceWindSpeed,
            lambdar=central_wavelength,
            **kwargs
        )
        wasv_rsv = np.full(L1.IncidenceAngleImage.shape, np.nan)
        for pol_str in ['VV', 'HH']:
            wasv_rsv[ind[pol_str]] = dc[pol_str].values[ind[pol_str]]

    elif gmf[:5] == 'oscar':
        wasv_rsv = oscar_empirical_wasv(
            aux_geo.OceanSurfaceWindSpeed,
            relative_wind_direction,
            L1.IncidenceAngleImage,
            L1.Polarization,
            gmf=gmf
        )

    else:
        raise Exception(
            'Error, unknown gmf, should be yurovsky19 or mouche 12'
            )

    ds_wa = xr.DataArray(
        data=wasv_rsv.data,
        coords=L1.AntennaAzimuthImage.coords,
        dims=L1.AntennaAzimuthImage.dims
    )
    ds_wa.attrs['long_name'] = 'Wind Artifact Surface Velocity (WASV)'
    ds_wa.attrs['units'] = ['m/s']
    
    # Addition of the DopplerGMF attribute
    ds_wa.attrs['DopplerGMF'] = gmf
    
    return ds_wa

def oscar_empirical_wasv(
        u10: float or np.ndarray or xr.DataArray,
        phi: float or np.ndarray or xr.DataArray,
        inc: float or np.ndarray or xr.DataArray,
        pol: float or np.ndarray or xr.DataArray,
        gmf: str) -> float or np.ndarray or xr.DataArray:

    """
    Compute Doppler shift due to the empirical geophysical model function 
    derived from oscar measurements.

    Parameters
    ----------
    u10 : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        Wind speed (m/s) at 10m above sea surface.
    phi : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        Angle between wind and look directions (degrees) in range 0 (upwind) :
            90 (crosswind) : 180 (downwind).
    inc : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        Incidence angle of radar beam (degrees from nadir).
    pol : str
        Polarisation of radar beam (VV).
    gmf : str
        Valid GMF:
        'oscar20220522T11-18_v20250318'

    Raises
    ------
    Exception
        Exception raised if unknown gmf. It should be one of the described above.

    Returns
    -------
    wasv_rsv: ``float``, ``numpy.ndarray``, ``xarray.DataArray``
        Radial surface velocity 
    uncerty_rsv: ``float``, ``numpy.ndarray``, ``xarray.DataArray``
        Uncertainty on the rsv as infered from the fit
    """

    if gmf == 'oscar20220522T11-18_v20250318':
        wasv_rsv = get_second_harmonic_inci_legendre(phi, inc, gmf=gmf)
    else:
        raise Exception(
            'Error, unknown gmf. See documentation.'
            )
    
    return wasv_rsv


def _load_second_harmonic_inci_legendre(gmf: str) -> list:
    """
    Load empirical GMF following a second harmonic fit in azimuth, 
    with Legendre polymonial fit in incidence angle

    The curve is of the form: An + Bn*cos(phi) + Cn*cos(2phi)
    With, An, Bn, Cn Legendre polymonial of order 'n' over the normalised 
    domain in incidence angle: x = 2*(inci - inci_centre)/inci_range 

    Parameters
    ----------
    gmf : str
        One of the following string format:
        "oscar20220522T11-18_v20250318"

    Raises
    ------
    Exception
        Exception raised if unknown gmf. It should be one of the described above.

    Returns
    -------
        List of parameters

    """
    dirpath = abspath(dirname(__file__))

    if gmf == 'oscar20220522T11-18_v20250318':
        fname = join(dirpath, 'GMF_OSCAR_20220522_T11-18_v20250318.csv')
    else:
        raise Exception(
            'Error, unknown gmf. See documentation.'
            )
    
    df = pd.read_csv(fname)
    # a = L(list(df['A']), domain=np.array([-0.5,0.5]), window=np.array([-1,1]))
    # b = L(list(df['B']), domain=np.array([-0.5,0.5]), window=np.array([-1,1]))
    # c = L(list(df['C']), domain=np.array([-0.5,0.5]), window=np.array([-1,1]))
    a = L(list(df['A']))
    b = L(list(df['B']))
    c = L(list(df['C']))
    inci_angle_centre = np.unique(df['centre'])
    inci_angle_range = np.unique(df['range'])

    if len(inci_angle_centre) > 1:
        raise Exception(
            'Error, centre in the CSV table should be a constant value.'
            )
    if len(inci_angle_range) > 1:
        raise Exception(
            'Error, range in the CSV table should be a constant value.'
        )
    return(list([a,b,c,inci_angle_centre,inci_angle_range]))

def get_second_harmonic_inci_legendre(phi, inc, gmf: str) -> list:
    """
    Load empirical GMF following a second harmonic fit in azimuth, 
    with Legendre polymonial fit in incidence angle

    The curve is of the form: An + Bn*cos(phi) + Cn*cos(2phi)
    With, An, Bn, Cn Legendre polymonial of order 'n' over the normalised 
    domain in incidence angle: x = 2*(inci - inci_centre)/inci_range 

    Parameters
    ----------
    phi : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        Angle between wind and look directions (degrees) in range 0 (upwind) :
            90 (crosswind) : 180 (downwind).
    inc : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        Incidence angle of radar beam (degrees from nadir).
    gmf : str
        One of the following string format:
        "oscar20220522T11-18_v20250318"

    Returns
    -------
        

    """
    [a,b,c,centre,range] = _load_second_harmonic_inci_legendre(gmf)

    x = 2*(inc - centre)/range
    wasv_rsv = a(x) + b(x)*np.cos(np.radians(phi)) + c(x)*np.cos(np.radians(2*phi))
    return(wasv_rsv)


def convertDoppler2Velocity(freq_GHz, dop, inci):
    """
    Convert Doppler shift to surface velocity.

    Convert Doppler shift (Hz) to surface velocity (m/s)

    Parameters
    ----------
    freq_GHz : float
        Central electromagnetic frequency (GHz).
    dop : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        Doppler shift (Hz).
    inci : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        Incidence angle (degrees from nadir).

    Raises
    ------
    Exception
        Exception for central frequency not being in GHz.

    Returns
    -------
    los_vel : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        Line-of-sight velocity (m/s).
    surf_vel : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        Surface velocity (m/s).

    """
    # Do not change, freq_GHz need to be in GHz ADMARTIN,
    # used in CEASELESS 2022
    if 100 < freq_GHz < 0.1:
        raise Exception('Inputs freq_GHz should be in GHz, e.g. C-band: 5.5 '
                        'X-band 9.55; Ku-band 13.6')
    n = 1.000293
    c_air = c / n
    wavelength = c_air / freq_GHz / 1e9
    los_vel = - dop * wavelength / 2
    surf_vel = los_vel / np.sin(np.radians(inci))
    return los_vel, surf_vel


def mouche12(u10, phi, inc, pol):
    """
    Compute Doppler shift due to wind from Mouche (2012).

    Compute the Doppler shift due to the wind using the geophysical model
    function of Mouche (2012). Input parameters u10, phi and inc must be in
    identical sizes and formats.

    Parameters
    ----------
    u10 : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        Wind speed (m/s) at 10m above sea surface.
    phi : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        Angle between wind and look directions (degrees) in range 0 (upwind) :
            90 (crosswind) : 180 (downwind).
    inc : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        Incidence angle of radar beam (degrees from nadir).
    pol : str
        Polarisation of radar beam (HH or VV).

    Raises
    ------
    Exception
        Exception for inconsistency in sizes of input parameters.
        Exception for polarisation out of range of (HH,VV).

    Returns
    -------
    dop : ``float``, ``numpy.array``, ``numpy.ndarray``, ``xarray.DataArray``
        Doppler shift (GHz) due to geophysical and geometric conditions.

    """
    def cdop_func(x):
        """Exponential function for mouche12."""
        return 1. / (1. + np.exp(-x))

    # Check inputs
    sizes = np.array([np.size(inc), np.size(u10), np.size(phi)])
    size = sizes.max()
    if ((sizes != size) & (sizes != 1)).any():
        raise Exception('Inputs sizes do not agree.')
    if pol not in ['VV', 'HH']:
        raise Exception('Unknown polarisation : ' + pol)
    # NN coefficients (W=weights and B=biases)
    # (coefficient names in mouche2012 are given)
    if pol == 'VV':
        # lambda[0:2,1]
        B1 = np.array([-0.343935744939, 0.108823529412, 0.15],
                      dtype='float32')
        # lambda[0:2,0]
        W1 = np.array([0.028213254683, 0.0411764705882, .00388888888889],
                      dtype='float32')
        # omega[i,0]
        B2 = np.array([14.5077150927, -11.4312028555, 1.28692747109,
                       -1.19498666071, 1.778908726, 11.8880215573,
                       1.70176062351, 24.7941267067, -8.18756617111,
                       1.32555779345, -9.06560116738],
                      dtype='float32')
        # omega[i,[3,2,1]]
        W2 = np.array([[19.7873046673, 22.2237414308, 1.27887019276],
                       [2.910815875, -3.63395681095, 16.4242081101],
                       [1.03269004609, 0.403986575614, 0.325018607578],
                       [3.17100261168, 4.47461213024, 0.969975702316],
                       [-3.80611082432, -6.91334859293, -0.0162650756459],
                       [4.09854466913, -1.64290475596, -13.4031862615],
                       [0.484338480824, -1.30503436654, -6.04613303002],
                       [-11.1000239122, 15.993470129, 23.2186869807],
                       [-0.577883159569, 0.801977535733, 6.13874672206],
                       [0.61008842868, -0.5009830671, -4.42736737765],
                       [-1.94654022702, 1.31351068862, 8.94943709074]],
                      dtype='float32')
        # gamma[0]
        B3 = np.array(4.07777876994, dtype='float32')
        # gamma[1:11]
        W3 = np.array([7.34881153553, 0.487879873912, -22.167664703,
                       7.01176085914, 3.57021820094, -7.05653415486,
                       -8.82147148713, 5.35079872715, 93.627037987,
                       13.9420969201, -34.4032326496],
                      dtype='float32')
        # beta
        B4 = np.array(-52.2644487109, dtype='float32')
        # alpha
        W4 = np.array(111.528184073, dtype='float32')
    elif pol == 'HH':
        # lambda[0:2,1]
        B1 = np.array([-0.342097701547, 0.118181818182, 0.15],
                      dtype='float32')
        # lambda[0:2,0]
        W1 = np.array([0.0281843837385, 0.0318181818182, 0.00388888888889],
                      dtype='float32')
        # omega[i,0]
        B2 = np.array([1.30653883096, -2.77086154074, 10.6792861882,
                       -4.0429666906, -0.172201666743, 20.4895916824,
                       28.2856865516, -3.60143441597, -3.53935574111,
                       -2.11695768022, -2.57805898849],
                      dtype='float32')
        # omega[i,[3,2,1]]
        W2 = np.array([[-2.61087309812, -0.973599180956, -9.07176856257],
                       [-0.246776181361, 0.586523978839, -0.594867645776],
                       [17.9261562541, 12.9439063319, 16.9815377306],
                       [0.595882115891, 6.20098098757, -9.20238868219],
                       [-0.993509213443, 0.301856868548, -4.12397246171],
                       [15.0224985357, 17.643307099, 8.57886720397],
                       [13.1833641617, 20.6983195925, -15.1439734434],
                       [0.656338134446, 5.79854593024, -9.9811757434],
                       [0.122736690257, -5.67640781126, 11.9861607453],
                       [0.691577162612, 5.95289490539, -16.0530462],
                       [1.2664066483, 0.151056851685, 7.93435940581]],
                      dtype='float32')
        # gamma[0]
        B3 = np.array(2.68352095337, dtype='float32')
        # gamma[1:11]
        W3 = np.array([-8.21498722494, -94.9645431048, -17.7727420108,
                       -63.3536337981, 39.2450482271, -6.15275352542,
                       16.5337543167, 90.1967379935, -1.11346786284,
                       -17.57689699, 8.20219395141],
                      dtype='float32')
        # beta
        B4 = np.array(-66.9554922921, dtype='float32')
        # alpha
        W4 = np.array(136.216953823, dtype='float32')
    # Make inputs as a single matrix (and clip phi in [0,180])
    inputs = np.zeros((3, size), dtype='float32')
    for ivar, var in enumerate((inc, u10, phi)):
        if sizes[ivar] == 1:
            inputs[ivar, :] = np.repeat(var, size)
        else:
            inputs[ivar, :] = np.ravel(var)
        if ivar == 2:
            inputs[ivar, :] = np.abs(((inputs[ivar, :] + 180) % 360) - 180)
        inputs[ivar, :] *= W1[ivar]
        inputs[ivar, :] += B1[ivar]
    # Compute CDOP
    B2 = np.tile(B2.reshape((11, 1)), (1, size))
    dop = W4 * cdop_func(np.dot(W3, cdop_func(np.dot(W2, inputs) + B2)) +
                         B3) + B4
    # Reshape output
    # (using the shape of input which have the maximum ndim)
    ndims = np.array([np.ndim(inc), np.ndim(u10), np.ndim(phi)])
    tmp = np.where(sizes == size)[0]
    ivar = tmp[ndims[tmp].argmax()]
    shp = np.shape((inc, u10, phi)[ivar])
    dop = dop.reshape(shp)

    if isinstance(u10, xr.core.dataarray.DataArray):
        dop = xr.DataArray(
            data=dop,
            dims=u10.dims,
        )

    return dop


def yurovsky19(theta, phi, u, swh_sw=0, omega_sw=0, phi_sw=0, drift=0.015,
               lambdar=0.008, beta_ws=0.20, beta_sw=0.0625,
               zerocrosswind=False, **kwargs):
    """
    Compute Doppler shift due to wind using Yurovsky (2019).

    Compute Doppler shift due to the geophysical model function of
    Yurovsky (2019).

    KaDOP computes sea surface Doppler spectrum centroid for VV and HH
    polarizations, DCvv and DChh, based on the empirical MTF
    [10.1109/TGRS.2017.2787459] DC is expressed in m/s,
    DC[m/s]= DC[Hz]*lambda/2, where lambda is the radar wavelength.
    By default, the DC is estimated for Pierson-Moskowitz (PM),
    [10.1029/JZ069i024p05181]
    Wind sea spectrum without swell. If significant wave height (SWH)
    and peak frequency (omega) are known, they can be specified explicitly.
    Swell can be added independently by specifying its SWH and omega,
    no swell by default.

    Parameters
    ----------
    theta : ``float``, ``numpy.ndarray``, ``xarray.DataArray``
        Incidence angle (degrees from nadir) in the range (0:60).
    phi : ``float``, ``numpy.ndarray``, ``xarray.DataArray``
        Angle between wind and look directions (degrees) in range 0 (upwind) :
            90 (crosswind) : 180 (downwind).
    u : ``float``, ``numpy.ndarray``, ``xarray.DataArray``
        Wind speed (m/s) at 10m above sea surface.
    swh_sw : TYPE, optional
        Swell significant wave hight (m). The default is 0.
    omega_sw : TYPE, optional
        Swell peak radial frequency (rad/s). The default is 0.
    phi_sw : TYPE, optional
        Radar-to-swell relative direction (degrees). The default is 0.
    drift : TYPE, optional
        Wind-drift coefficient. The default is 0.015 (1.5% U).
    lambdar : TYPE, optional
        Radar wavelength (m). The default is 0.008 (Ka-band).
    beta_ws : TYPE, optional
        3rd moment of Wind Sea spectrum (Pierson-Moskowitz) parameter.
        The default is 0.20.
    beta_sw : TYPE, optional
        3rd moment of Swell spectrum parameter. The default is 0.0625.
    zerocrosswind : bool, optional
        If this parameter set to 1, the MTFwindsea is replaced by
        MTFswell (crosswind is zeroed). The default is False.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    VV : ``float``, ``numpy.ndarray``, ``xarray.DataArray``
        Sea surface Doppler Velocity for VV polarization
    HH : ``float``, ``numpy.ndarray``, ``xarray.DataArray``
        Sea surface Doppler Velocity centroid for HH polarization


    """
    cosd = lambda x: np.cos(x * np.pi / 180)
    sind = lambda x: np.sin(x * np.pi / 180)
    acosd = lambda x: np.arccos(x) * 180 / np.pi
    sech = lambda x: 1 / np.cosh(x * np.pi / 180)

    # =======================================================================
    def MTF(theta, phi, u, C):
        """
        Evaluate complex MTF using C-coefficients.

        Evaluate complex MTF using C-coefficients. 1st column of C-matrix
        is Bijk, 2nd column is Cijk from Equation (A1),Table A1 or A2

        Parameters
        ----------
        theta : float, numpy.array, xarray.DataArray
            Incidence angle (degrees from nadir) in the range (0:60).
        phi : float, numpy.array, xarray.DataArray
            Angle between wind and look directions (degrees) in range 0
            (upwind) :
            90 (crosswind) : 180 (downwind).
        u : TYPE
            DESCRIPTION.
        C : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        def evalfun(theta, phi, u, C):
            """Evaluate magnitude or phasor of MTF."""
            # evaluate magnitude or phasor of MTF
            # C is column, either Bijk or Cijk

            pp = C[0] + \
                C[1] * theta ** 1 + \
                C[2] * theta ** 2 + \
                C[3] * theta ** 3 + \
                C[4] * cosd(1 * phi) + \
                C[5] * theta ** 1 * cosd(1 * phi) + \
                C[6] * theta ** 2 * cosd(1 * phi) + \
                C[7] * theta ** 3 * cosd(1 * phi) + \
                C[8] * cosd(2 * phi) + \
                C[9] * theta ** 1 * cosd(2 * phi) + \
                C[10] * theta ** 2 * cosd(2 * phi) + \
                C[11] * theta ** 3 * cosd(2 * phi) + \
                C[12] * np.log(u) ** 1 + \
                C[13] * theta ** 1 * np.log(u) ** 1 + \
                C[14] * theta ** 2 * np.log(u) ** 1 + \
                C[15] * theta ** 3 * np.log(u) ** 1 + \
                C[16] * cosd(1 * phi) * np.log(u) ** 1 + \
                C[17] * theta ** 1 * cosd(1 * phi) * np.log(u) ** 1 + \
                C[18] * theta ** 2 * cosd(1 * phi) * np.log(u) ** 1 + \
                C[19] * theta ** 3 * cosd(1 * phi) * np.log(u) ** 1 + \
                C[20] * cosd(2 * phi) * np.log(u) ** 1 + \
                C[21] * theta ** 1 * cosd(2 * phi) * np.log(u) ** 1 + \
                C[22] * theta ** 2 * cosd(2 * phi) * np.log(u) ** 1 + \
                C[23] * theta ** 3 * cosd(2 * phi) * np.log(u) ** 1

            return pp

        Mabs = evalfun(theta, phi, u, C[:, 0])
        Mphs = evalfun(theta, phi, u, C[:, 1])
        M = np.exp(Mabs) * Mphs / abs(Mphs)

        return M

    g = 9.8  # Gravitational constant [m/s^2]
    gamma = 7.3e-5  # Surface tension [N/m]

    if 'swh' not in kwargs.keys():
        # Wind sea Significant Wave Height [m], PM by default
        swh = 0.22 * u ** 2 / g
    else:
        swh = kwargs.pop('swh')

    if 'omega' not in kwargs.keys():
        # Wind sea peak radial frequency [rad/s], PM by default
        omega = 0.83 * g / u
    else:
        omega = kwargs.pop('omega')

    # Set of coefficients, Table A1 and A2.
    # 1st column is Bijk, 2nd column is Cijk from Equation (A1)

    coefs = {}
    coefs["VVws"] = np.array([[+2.037368e+00, -9.991774e-01 - 1.859445e-03j],
                              [-9.956181e-03, +9.995403e-02 - 3.728707e-02j],
                              [+1.733240e-03, -9.495314e-04 + 5.073520e-04j],
                              [-2.110994e-05, -1.742060e-06 + 2.930913e-06j],
                              [-1.704388e-02, -2.062522e-03 + 4.317005e-03j],
                              [-4.002570e-02, -2.021244e-02 + 1.328154e-01j],
                              [+2.213287e-03, +1.037791e-03 - 5.526796e-03j],
                              [-1.778161e-05, -1.183648e-05 + 4.932378e-05j],
                              [-2.933537e-02, -5.651327e-05 + 1.289564e-03j],
                              [+2.755026e-02, +7.638659e-02 + 7.101499e-02j],
                              [+1.382417e-03, -3.141920e-03 - 2.127452e-03j],
                              [-2.811759e-05, +3.360741e-05 + 1.363174e-05j],
                              [-2.637003e-01, -1.300697e-03 + 6.335937e-04j],
                              [+2.457828e-02, -1.060972e-02 + 4.969400e-03j],
                              [-1.537867e-03, -2.108491e-05 - 1.405381e-05j],
                              [+1.667354e-05, +2.373730e-06 - 1.623276e-06j],
                              [+1.342060e-02, +4.740406e-04 - 8.386239e-04j],
                              [+1.791006e-02, +9.982368e-03 - 1.343944e-02j],
                              [-1.048575e-03, -4.634691e-04 + 1.129914e-03j],
                              [+9.158551e-06, +5.153546e-06 - 1.134140e-05j],
                              [+1.809446e-02, +2.879613e-04 - 3.980226e-04j],
                              [+8.255341e-03, -2.309667e-02 - 1.347916e-02j],
                              [-1.286835e-03, +9.359817e-04 + 5.873901e-04j],
                              [+1.827908e-05, -1.056345e-05 - 5.154716e-06j]])

    coefs["HHws"] = np.array([[+2.038368e+00, -9.999579e-01 - 2.003675e-03j],
                              [+6.742867e-02, +1.401092e-01 - 3.822135e-02j],
                              [-1.544673e-03, -2.832742e-03 + 6.391936e-04j],
                              [+1.167191e-05, +1.755927e-05 - 1.325959e-06j],
                              [-1.716876e-02, -2.510170e-03 + 5.669125e-03j],
                              [-2.064313e-02, -1.886127e-03 + 1.301061e-01j],
                              [+1.172491e-03, +2.217910e-04 - 5.440821e-03j],
                              [-6.111610e-06, -2.769183e-06 + 5.317919e-05j],
                              [-2.939264e-02, +1.738649e-03 + 1.255492e-03j],
                              [+4.007160e-03, +3.758102e-02 + 7.395083e-02j],
                              [+1.482772e-03, -1.072406e-03 - 2.254102e-03j],
                              [-2.163604e-05, +8.151756e-06 + 1.559167e-05j],
                              [-2.643806e-01, -8.840229e-04 + 6.209692e-04j],
                              [-1.240919e-02, -3.155538e-02 + 3.907412e-03j],
                              [+2.162084e-04, +8.937600e-04 - 1.544636e-05j],
                              [-3.482596e-07, -6.512207e-06 - 4.914423e-07j],
                              [+1.347741e-02, +7.416105e-04 - 1.536552e-03j],
                              [+7.223413e-03, -2.172061e-03 - 1.458223e-02j],
                              [-5.037439e-04, +1.053785e-04 + 1.203955e-03j],
                              [+2.889241e-06, -9.978940e-07 - 1.415368e-05j],
                              [+1.812623e-02, -6.400749e-04 - 4.329797e-04j],
                              [+2.313635e-02, -5.070167e-03 - 1.231709e-02j],
                              [-1.569241e-03, -5.514080e-06 + 5.292689e-04j],
                              [+1.795667e-05, +8.560235e-07 - 4.894367e-06j]])

    coefs["VVsw"] = np.array([[+2.037368e+00, -1.047849e+00 - 1.086382e-03j],
                              [-9.956181e-03, +9.779865e-02 + 9.409557e-03j],
                              [+1.733240e-03, -9.521228e-04 - 1.330189e-03j],
                              [-2.110994e-05, -8.936468e-07 + 1.921637e-05j],
                              [-1.704388e-02, -2.054076e-02 + 2.380576e-02j],
                              [-4.002570e-02, +4.046633e-02 + 1.544692e-01j],
                              [+2.213287e-03, -1.395978e-03 - 5.769671e-03j],
                              [-1.778161e-05, +1.340544e-05 + 4.688263e-05j],
                              [-2.933537e-02, -4.552795e-03 - 3.923333e-03j],
                              [+2.755026e-02, +2.273467e-02 + 1.289799e-02j],
                              [+1.382417e-03, -8.407162e-04 + 1.345284e-05j],
                              [-2.811759e-05, +9.080283e-06 - 3.645146e-06j],
                              [-2.637003e-01, +4.449188e-03 + 1.717938e-03j],
                              [+2.457828e-02, -1.171622e-02 - 2.045575e-03j],
                              [-1.537867e-03, +9.499907e-05 + 4.015526e-04j],
                              [+1.667354e-05, +8.816342e-07 - 5.631314e-06j],
                              [+1.342060e-02, +5.159466e-03 - 6.475855e-03j],
                              [+1.791006e-02, -9.459894e-03 - 1.412467e-02j],
                              [-1.048575e-03, +3.075467e-04 + 9.873627e-04j],
                              [+9.158551e-06, -3.260269e-06 - 8.840548e-06j],
                              [+1.809446e-02, +1.029965e-03 + 1.201244e-03j],
                              [+8.255341e-03, -3.648071e-03 - 5.884530e-03j],
                              [-1.286835e-03, +1.828698e-06 + 7.071967e-05j],
                              [+1.827908e-05, +1.276843e-07 + 8.061632e-08j]])

    coefs["HHsw"] = np.array([[+2.038368e+00, -1.070596e+00 + 4.617718e-04j],
                              [+6.742867e-02, +1.422845e-01 + 4.036745e-03j],
                              [-1.544673e-03, -2.882753e-03 - 1.021860e-03j],
                              [+1.167191e-05, +1.838410e-05 + 1.433121e-05j],
                              [-1.716876e-02, -1.404714e-02 + 2.765220e-02j],
                              [-2.064313e-02, +2.884548e-02 + 1.580035e-01j],
                              [+1.172491e-03, -6.833107e-04 - 6.044204e-03j],
                              [-6.111610e-06, +4.112504e-06 + 5.471463e-05j],
                              [-2.939264e-02, +1.196099e-02 - 5.905559e-03j],
                              [+4.007160e-03, -6.952809e-03 + 1.881372e-02j],
                              [+1.482772e-03, +3.991268e-04 - 2.664768e-04j],
                              [-2.163604e-05, -4.235270e-06 - 1.228258e-06j],
                              [-2.643806e-01, +1.676822e-02 + 5.227076e-05j],
                              [-1.240919e-02, -3.573475e-02 - 7.998733e-04j],
                              [+2.162084e-04, +1.083750e-03 + 3.168845e-04j],
                              [-3.482596e-07, -8.535620e-06 - 4.213366e-06j],
                              [+1.347741e-02, +3.305453e-03 - 8.652656e-03j],
                              [+7.223413e-03, -6.991652e-03 - 1.631189e-02j],
                              [-5.037439e-04, +1.321311e-04 + 1.143512e-03j],
                              [+2.889241e-06, -5.730351e-07 - 1.266315e-05j],
                              [+1.812623e-02, -7.689661e-03 + 1.684635e-03j],
                              [+2.313635e-02, +1.171194e-02 - 6.082012e-03j],
                              [-1.569241e-03, -6.270342e-04 + 9.248031e-05j],
                              [+1.795667e-05, +6.716300e-06 - 1.181313e-08j]])

    if zerocrosswind is True:
        coefs["VVws"] = coefs["VVsw"]
        coefs["HHws"] = coefs["HHsw"]

    G = lambda theta, phi: cosd(phi) * sind(theta) - 1j * cosd(theta)
    spread = lambda phi: sech(1.0 * acosd(cosd(phi))) ** 2
    Braggsum = lambda phi: (spread(phi) - spread(phi + 180)) /\
        (spread(phi) + spread(phi + 180))
    vbr = lambda theta, lambdar: np.sqrt(g *
                                         sind(theta) /
                                         (4 * np.pi / lambdar) +
                                         gamma *
                                         (4 * np.pi /
                                          lambdar * sind(theta) ** 3))
    if (np.size(theta) == np.size(phi)) & (np.size(phi) == np.size(u)):
        THETA = theta
        PHI = phi
        U = u
        OMEGA = omega
        SWH = swh
        OMEGA_sw = omega_sw
        SWH_sw = swh_sw
        PHI_sw = phi_sw
    else:
        [THETA, PHI, U] = np.meshgrid(theta, phi, u)
        [_, _, OMEGA] = np.meshgrid(theta, phi, omega)
        [_, _, SWH] = np.meshgrid(theta, phi, swh)

        [_, _, OMEGA_sw] = np.meshgrid(theta, phi, omega_sw)
        [_, _, SWH_sw] = np.meshgrid(theta, phi, swh_sw)
        [_, _, PHI_sw] = np.meshgrid(theta, phi, phi_sw)

    nonpolDoppler = vbr(THETA, lambdar) * Braggsum(PHI) +\
        drift * U * cosd(PHI) * sind(THETA)

    VV = nonpolDoppler + \
        beta_ws * np.real(G(THETA, PHI) *
                          MTF(THETA, PHI, U, coefs.get("VVws"))) / g * \
        SWH ** 2 * OMEGA ** 3 + \
        beta_sw * np.real(G(THETA, PHI_sw) *
                          MTF(THETA, PHI_sw, U, coefs.get("VVsw"))) / g * \
        SWH_sw ** 2 * OMEGA_sw ** 3

    HH = nonpolDoppler + \
        beta_ws * np.real(G(THETA, PHI) *
                          MTF(THETA, PHI, U, coefs.get("HHws"))) / g * \
        SWH ** 2 * OMEGA ** 3 + \
        beta_sw * np.real(G(THETA, PHI_sw) *
                          MTF(THETA, PHI_sw, U, coefs.get("HHsw"))) / g * \
        SWH_sw ** 2 * OMEGA_sw ** 3

    if isinstance(theta, xr.core.dataarray.DataArray):
        VV = xr.DataArray(
            data=VV,
            dims=theta.dims,
        )
        HH = xr.DataArray(
            data=HH,
            dims=theta.dims,
        )

    return [-VV, -HH]  # convention different ADMARTIN 24/03/2021
