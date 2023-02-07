#!/usr/bin/env python
# coding=utf-8
import numpy as np
import xarray as xr
import pytest
from seastar.utils.tools import dotdict

from seastar.gmfs import doppler

@pytest.fixture
def point_wind_speed():
    return(7) # in m/s

@pytest.fixture
def point_relative_dir():
    return(3) # in degree

@pytest.fixture
def point_inci_angle():
    return(30) # in degree from nadir

@pytest.fixture
def point_polarization():
    return 'VV' # in degree

@pytest.fixture
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


    return dict({'level1': level1, 'geo': geo})

@pytest.fixture
def gmf_mouche():
    gmf = dotdict({'nrcs': dotdict({'name': 'nscat4ds'})})
    gmf['doppler'] = dotdict({'name': 'mouche12'})
    return gmf

# TODO might be ideal to have a table of input and expected output


def test_mouche12(point_wind_speed, point_relative_dir, point_inci_angle, point_polarization):

    assert doppler.mouche12(point_wind_speed, point_relative_dir, point_inci_angle, point_polarization) \
           == pytest.approx(24, 1) # 1 +- 5 => error on pol in mouche12, with pol expected as a str not a function
    # Test points
    assert doppler.mouche12(7, 3, 30, 'VV') \
            == pytest.approx(24, 1)  # in Hz 24 +- 1
    assert doppler.mouche12(7, 3, 30, 'HH') \
           == pytest.approx(24, 1)  # in Hz 24 +- 1

    # Test numpy array
    np.testing.assert_allclose(
        doppler.mouche12(
            np.array([3, 7, 20]),
            np.array([0, 3, 200]),
            np.array([20, 30, 40]),
            'VV' #np.array(['VV', 'VV', 'VV']),
        ),
        np.array([ 19.6,  24.1, -23.8 ]),
        rtol=0.1,
    ) # in Hz

    # Test xr.DataArray
    t, inci2D = np.mgrid[-10:10:5, 20:50:10]
    wdir2D = 150 + 15 * np.random.random_sample(inci2D.shape)
    wspd2D = 7 + 2 * np.random.random_sample(inci2D.shape)
    ds = xr.Dataset(
        data_vars=dict(
            incidenceImage=(['x', 'y'], inci2D),
            wdir=(['x', 'y'], wdir2D),
            wspd=(['x', 'y'], wspd2D),
        ),
    )
    xr.testing.assert_allclose(
        doppler.mouche12(
            ds.wspd,
            ds.wdir,
            ds.incidenceImage,
            'VV',
        ),
        xr.DataArray(
            data=np.array([[-21, -16, -12],
                           [-20, -15, -12],
                           [-21, -16, -12],
                           [-21, -16, -12]]),
            dims=(['x', 'y'])
        ),
        rtol=1
    )

def test_compute_wasv(level1_geo_dataset, gmf_mouche):
    level1 = level1_geo_dataset['level1']
    L1ant = level1.sel(Antenna='Fore')
    geo = level1_geo_dataset['geo']

    # Test dataset with .sel(Antenna)
    xr.testing.assert_allclose(
        doppler.compute_wasv(
            L1ant,
            geo,
            gmf=gmf_mouche.doppler.name
        ).drop_vars(['CentralWavenumber', 'CentralFreq', 'IncidenceAngleImage', 'AntennaAzimuthImage',
                     'Polarization','Antenna','along','across']),
        xr.DataArray(
            data=np.full( L1ant.AntennaAzimuthImage.shape, 0.32),
            dims=( L1ant.dims)
        ),
        rtol=0.01
    )




    # with pytest.raises(ValueError):
    #     adult_or_child(-10)