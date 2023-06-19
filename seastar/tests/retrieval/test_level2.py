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
            EarthRelativeWindSpeed=(['across', 'along'], np.full([5, 6], 10)),
            EarthRelativeWindDirection=(['across', 'along'], np.full([5, 6], 150)),
            CurrentVelocity=(['across', 'along'], np.full([5, 6], 1)),
            CurrentDirection=(['across', 'along'], np.full([5, 6], 150)),
        ),
        coords=dict(
            across=np.arange(0, 5),
            along=np.arange(0, 6),
        ),
    )

    geo = seastar.utils.tools.EarthRelativeSpeedDir2all(geo)

    # geo['u'], geo['v'] = seastar.utils.tools.windSpeedDir2UV(geo.EarthRelativeWindSpeed, geo.EarthRelativeWindDirection)
    # [geo['c_u'], geo['c_v']] = seastar.utils.tools.currentVelDir2UV(geo.CurrentVelocity, geo.CurrentDirection)
    # geo['vis_u'] = geo['u'] - geo['c_u']
    # geo['vis_v'] = geo['v'] - geo['c_v']
    # geo['vis_wspd'], vis_wdir = \
    #     seastar.utils.tools.windUV2SpeedDir(
    #         geo['vis_u'], geo['vis_v']
    #     )
    # geo['vis_wdir'] = (geo.vis_wspd.dims, vis_wdir)

    level1['Sigma0'] = seastar.gmfs.nrcs.compute_nrcs(level1, geo, gmf_mouche.nrcs) * 1.0001
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
    noise['Sigma0'] = level1.Sigma0 * 0.001
    noise['RSV'] = level1.RSV * 0.001

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

    L1 = level1.isel(across=slice(0, 2), along=slice(0, 2))
    N = noise.isel(across=slice(0, 2), along=slice(0, 2))
    ambiguity = {'name': 'closest_truth',
                 'truth': geo.isel(across=slice(0, 2), along=slice(0, 2)),
                 'method': 'current'}

    sL1 = level1.isel(across=slice(0,2), along=0)
    sN = noise.isel(across=slice(0,2), along=0)

    ssL1 = level1.isel(across=0, along=0)
    ssN = noise.isel(across=0, along=0)
    # ambiguity = {'name': 'sort_by_cost'}
    ssambiguity = {'name': 'closest_truth', 'truth': geo.isel(across=0, along=0), 'method': 'current'}

    # Test on full xr.DataSet but single pixel
    ssl2 = level2.wind_current_retrieval(ssL1, ssN, gmf_mouche, ssambiguity)
    ssds = xr.Dataset(
        data_vars=dict(
            x=(['x_variables'], [-5.00, 8.66, 0.50, -0.86]),
            CurrentU=([], 0.50),
            CurrentV=([], -0.86),
            EarthRelativeWindU=([], -5.00),
            EarthRelativeWindV=([], 8.66),
            CurrentVelocity=([], 1.00),
            CurrentDirection=([], 150.00),
            EarthRelativeWindSpeed=([], 10.00),
            EarthRelativeWindDirection=([], 150.00),
        ),
        coords=dict(x_variables=['u','v','c_u','c_v']),
    )

    xr.testing.assert_allclose(
        ssl2.reset_coords(drop=True)[list(ssds.keys())],
        ssds,
        rtol=0.1
    )

    # Test on full xr.DataSet with 1 dimension
    # sl2, slmout = level2.wind_current_retrieval(sL1, sN, gmf_mouche, ambiguity)
    # sds = xr.Dataset(
    #     data_vars=dict(
    #         x=(['x_variables'], [-4.50, 7.80, 0.50, -0.86]),
    #         CurrentU=([], 0.50),
    #         CurrentV=([], -0.86),
    #         WindU=([], -4.50),
    #         WindV=([], 7.80),
    #         CurrentVelocity=([], 1.00),
    #         CurrentDirection=([], 150.00),
    #         WindSpeed=([], 9.00),  # TODO should be 10 if AbsoluteWind, 9 if RelativeWind
    #         WindDirection=([], 150.00),
    #     ),
    # )

    # Test on full xr.DataSet with 2 dimension
    l2 = level2.wind_current_retrieval(L1, N, gmf_mouche, ambiguity)
    # ds = xr.Dataset(
    #     data_vars=dict(
    #         x=( ['x_variables', 'across', 'along'], [-4.50, 7.80, 0.50, -0.86]),
    #         CurrentU=([], 0.50),
    #         CurrentV=([], -0.86),
    #         WindU=([], -4.50),
    #         WindV=([], 7.80),
    #         CurrentVelocity=([], 1.00),
    #         CurrentDirection=([], 150.00),
    #         WindSpeed=([], 9.00),  # TODO should be 10 if AbsoluteWind, 9 if RelativeWind
    #         WindDirection=([], 150.00),
    #     ),
    # ) # Don't work as the global minimal is not always found.
    # should change the kind of test I do here TODO