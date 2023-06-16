#!/usr/bin/env python
# coding=utf-8
import numpy as np
import xarray as xr
import pytest
import seastar
from seastar.utils.tools import dotdict

from seastar.retrieval import cost_function

@pytest.fixture
# TODO it is the same function/constant as in test_doppler, it would be better to have it save somewhere and load here
def level1_geo_dataset(gmf_mouche):
    level1 = xr.Dataset(
        data_vars=dict(
            CentralWavenumber=([], 270),
            CentralFreq=([], 13.5 * 10 ** 9),
            IncidenceAngleImage=(['across', 'along', 'Antenna'], np.full([5, 6, 4], 30)),
            AntennaAzimuthImage=(['across', 'along', 'Antenna'],
                                 np.stack((np.full([5, 6], 45),
                                           np.full([5, 6], 90),
                                           np.full([5, 6], 90),
                                           np.full([5, 6], 135)
                                           ), axis=-1
                                          )
                                 ),
            Polarization=(['across', 'along', 'Antenna'],
                          np.stack((np.full([5, 6], 'VV'),
                                    np.full([5, 6], 'VV'),
                                    np.full([5, 6], 'HH'),
                                    np.full([5, 6], 'VV')
                                    ), axis=-1
                                   )
                          ),
            #             Sigma0=( ['across','along','Antenna'], np.full([5, 6,4], 1.01) ),
            #             dsig0=( ['across','along','Antenna'], np.full([5, 6,4], 0.05) ),
            #             RSV=( ['across','along','Antenna'], np.full([5, 6,4], 0.5) ),
            #             dRSV=( ['across','along','Antenna'], np.full([5, 6,4], 0.01) ),
        ),
        coords=dict(
            across=np.arange(0, 5),
            along=np.arange(0, 6),
            Antenna=['Fore', 'MidV', 'MidH', 'Aft'],
        ),
    )

    level1 = level1.set_coords([
        'CentralWavenumber',
        'CentralFreq',
        'IncidenceAngleImage',
        'AntennaAzimuthImage',
        'Polarization',
    ])

    geo = xr.Dataset(
        data_vars=dict(
            WindSpeed=(['across', 'along'], np.full([5, 6], 10)),
            WindDirection=(['across', 'along'], np.full([5, 6], 150)),
            CurrentVelocity=(['across', 'along'], np.full([5, 6], 1)),
            CurrentDirection=(['across', 'along'], np.full([5, 6], 150)),
        ),
        coords=dict(
            across=np.arange(0, 5),
            along=np.arange(0, 6),
        ),
    )
    geo['U'], geo['V'] = seastar.utils.tools.windSpeedDir2UV(geo.WindSpeed, geo.WindDirection)
    [geo['C_U'], geo['C_V']] = seastar.utils.tools.currentVelDir2UV(geo.CurrentVelocity, geo.CurrentDirection)

    # TODO warning should be relative wind, not absolute wind
    level1['Sigma0'] = seastar.gmfs.nrcs.compute_nrcs(level1, geo, gmf_mouche.nrcs) * 1.001
    level1['RSV'] = xr.concat([
        seastar.gmfs.doppler.compute_total_surface_motion(
            level1.sel(Antenna=ant),
            geo,
            gmf=gmf_mouche.doppler.name
        ) for ant in level1.Antenna.data],
        'Antenna',
        join='outer')
    # Add NaN for RSV for the mid antennas
    level1.RSV[1, :, :] = np.full([5, 6], np.nan)
    level1.RSV[2, :, :] = np.full([5, 6], np.nan)
    # Noise
    noise = level1.drop_vars([var for var in level1.data_vars])
    noise['Sigma0'] = level1.Sigma0 * 0.05
    noise['RSV'] = level1.RSV * 0.05

    # TODO, we shouldn't do this, it should be fixed value, without calling a function
    return dict({'level1': level1, 'geo': geo, 'noise': noise})

@pytest.fixture
def gmf_mouche():
    gmf = dotdict({'nrcs': dotdict({'name': 'nscat4ds'})})
    gmf['doppler'] = dotdict({'name': 'mouche12'})
    return gmf


def test_fun_residual(level1_geo_dataset, gmf_mouche):
    level1 = level1_geo_dataset['level1']
    geo = level1_geo_dataset['geo']
    noise = level1_geo_dataset['noise']

    # Test on full xr.DataSet
    results = cost_function.fun_residual([geo.U, geo.V, geo.C_U, geo.C_V], level1, noise, gmf_mouche)
    np.testing.assert_allclose(
        results[:,0,0],
        np.array([-3.0, -3.2, -3.2, -3.7,
                  -7.4,  0. ,  0. , -2.4 ]),
        rtol=0.1,
    )  # cost values for Sigma0, then RSV


def test_find_minima(level1_geo_dataset, gmf_mouche):
    level1 = level1_geo_dataset['level1']
    geo = level1_geo_dataset['geo']
    noise = level1_geo_dataset['noise']

    ssL1 = level1.isel(across=0, along=0)
    ssN = noise.isel(across=0, along=0)


    # Test on full xr.DataSet
    results = cost_function.find_minima(ssL1, ssN, gmf_mouche)
    margin_noise_floor = 2
    assert (results.cost < 1*margin_noise_floor).any() # one of the 4 minima should be below 1 (noise floor)
