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
def level1_geo_dataset():
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
            #             RVL=( ['across','along','Antenna'], np.full([5, 6,4], 0.5) ),
            #             drvl=( ['across','along','Antenna'], np.full([5, 6,4], 0.01) ),
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
    # TODO, we shouldn't do this, it should be fixed value, without calling a function
    geo['U'], geo['V'] = seastar.utils.tools.windSpeedDir2UV(geo.WindSpeed, geo.WindDirection)
    [geo['C_U'], geo['C_V']] = seastar.utils.tools.currentVelDir2UV(geo.CurrentVelocity, geo.CurrentDirection)
    return dict({'level1': level1, 'geo': geo})

@pytest.fixture
def gmf_mouche():
    gmf = dotdict({'nrcs': dotdict({'name': 'nscat4ds'})})
    gmf['doppler'] = dotdict({'name': 'mouche12'})
    return gmf


def test_fun_residual(level1_geo_dataset, gmf_mouche):
    level1 = level1_geo_dataset['level1']
    geo = level1_geo_dataset['geo']
    level1['Sigma0'] = seastar.gmfs.nrcs.compute_nrcs(level1, geo, gmf_mouche.nrcs) * 1.001
    level1['RVL'] = xr.concat([
        seastar.gmfs.doppler.compute_total_surface_motion(
            level1.sel(Antenna=ant),
            geo,
            gmf=gmf_mouche.doppler.name
        ) for ant in level1.Antenna.data],
        'Antenna',
        join='outer')
    # Add NaN for RVL for the mid antennas
    level1.RVL[1, :, :] = np.full([5, 6], np.nan)
    level1.RVL[2, :, :] = np.full([5, 6], np.nan)
    # Noise
    noise = level1.drop_vars([var for var in level1.data_vars])
    noise['Sigma0'] = level1.Sigma0 * 0.05
    noise['RVL'] = level1.RVL * 0.05

    # Test on full xr.DataSet
    results = cost_function.fun_residual([geo.U, geo.V, geo.C_U, geo.C_V], level1, noise, gmf_mouche)
    np.testing.assert_allclose(
        results[:,0,0],
        np.array([-3.0, -3.2, -3.2, -3.7,
                  -7.4,  0. ,  0. , -2.4 ]),
        rtol=0.1,
    )  # cost values for Sigma0, then RVL