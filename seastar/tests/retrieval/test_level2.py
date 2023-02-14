#!/usr/bin/env python
# coding=utf-8
import numpy as np
import xarray as xr
import pytest
import seastar
from seastar.utils.tools import dotdict

from seastar.retrieval import level2

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



    geo['u'], geo['v'] = seastar.utils.tools.windSpeedDir2UV(geo.WindSpeed, geo.WindDirection)
    [geo['c_u'], geo['c_v']] = seastar.utils.tools.currentVelDir2UV(geo.CurrentVelocity, geo.CurrentDirection)
    geo['vis_u'] = geo['u'] - geo['c_u']
    geo['vis_v'] = geo['v'] - geo['c_v']
    geo['vis_wspd'], vis_wdir = \
        seastar.utils.tools.windUV2SpeedDir(
            geo['vis_u'], geo['vis_v']
        )
    geo['vis_wdir'] = (geo.vis_wspd.dims, vis_wdir)

    # TODO geo should use the RelativeWind
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

    # TODO, we shouldn't do this, it should be fixed value, without calling a function
    return dict({'level1': level1, 'geo': geo, 'noise': noise})

@pytest.fixture
def gmf_mouche():
    gmf = dotdict({'nrcs': dotdict({'name': 'nscat4ds'})})
    gmf['doppler'] = dotdict({'name': 'mouche12'})
    return gmf


def test_wind_current_retrieval(level1_geo_dataset, gmf_mouche):
    level1 = level1_geo_dataset['level1']
    geo = level1_geo_dataset['geo']
    noise = level1_geo_dataset['noise']

    sL1 = level1.isel(across=slice(0,2), along=0)
    sN = noise.isel(across=slice(0,2), along=0)

    ssL1 = level1.isel(across=0, along=0)
    ssN = noise.isel(across=0, along=0)
    ambiguity = 'sort_by_cost'

    # Test on full xr.DataSet but single pixel
    ssl2, sslmout = level2.wind_current_retrieval(ssL1, ssN, gmf_mouche, ambiguity)
    ssds = xr.Dataset(
        data_vars=dict(
            x_variables=(['x_variables'], [-4.50, 7.80, 0.50, -0.86]),
            CurrentU=([], 0.50),
            CurrentV=([], -0.86),
            WindU=([], -4.50),
            WindV=([], 7.80),
            CurrentVelocity=([], 1.00),
            CurrentDirection=([], 150.00),
            WindSpeed=([], 9.00), # TODO should be 10 if AbsoluteWind, 9 if RelativeWind
            WindDirection=([], 150.00),
        ),
    )

    xr.testing.assert_allclose(
        ssl2,
        ssds,
        rtol=0.01
    )
