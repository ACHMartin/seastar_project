#!/usr/bin/env python
# coding=utf-8
import numpy as np
import xarray as xr
import pytest

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

# TODO might be ideal to have a table of input and expected output



def test_mouche12():
    # print(type(str(point_polarization)))
    # print(str(point_polarization))
    # print(type(point_inci_angle))
    # assert doppler.mouche12(point_wind_speed, point_relative_dir, point_inci_angle, str(point_polarization)) \
    #        == pytest.approx(1, 50) # 1 +- 5
    # Test points
    assert doppler.mouche12(7, 3, 30, 'VV') \
            == pytest.approx(24, 1)  # in Hz 24 +- 1
    assert doppler.mouche12(7, 3, 30, 'HH') \
           == pytest.approx(24, 1)  # in Hz 24 +- 1

    # Test numpy array
    assert  doppler.mouche12(
                np.array([3, 7, 20]),
                np.array([0, 3, 200]),
                np.array([20, 30, 40]),
                'VV' #np.array(['VV', 'VV', 'VV']),
            ) \
            == pytest.approx(24, 50)  # in Hz 24 +- 1

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





    # with pytest.raises(ValueError):
    #     adult_or_child(-10)