#!/usr/bin/env python
# coding=utf-8
import numpy as np
import xarray as xr
import pytest
import seastar
from seastar.utils.tools import dotdict

from seastar.gmfs import nrcs

@pytest.fixture # TODO 
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
    return 1 # 1 for VV 


def test_cmod7(point_wind_speed, point_relative_dir, point_inci_angle, point_polarization):

    assert nrcs.cmod7(point_wind_speed, point_relative_dir, point_inci_angle, point_polarization) \
           == pytest.approx(0.0795, abs=1e-4)
    # Test points
    np.isclose(nrcs.cmod7(7, 300, 30), np.nan, equal_nan=True) # Nan for value > 180°
    
    # Test numpy array
    np.testing.assert_allclose(
        nrcs.cmod7(
            np.array([3, 7, 20]),
            np.array([0, 3, 200]),
            np.array([20, 30, 40]),
            1 # 'VV'
        ),
        np.array([ 0.273,  0.079, np.nan]),
        rtol=1e-1, equal_nan=True
    )

    # Test xr.DataArray
    t, inci2D = np.mgrid[-10:10:5, 20:50:10]
    suite = np.arange(inci2D.shape[0] * inci2D.shape[1]).reshape(inci2D.shape)
    wdir2D = -10 + 21 * suite
    wspd2D = 0.3 + 2 * suite
    ds = xr.Dataset(
        data_vars=dict(
            incidenceImage=(['x', 'y'], inci2D),
            wdir=(['x', 'y'], wdir2D),
            wspd=(['x', 'y'], wspd2D),
        ),
    )
    xr.testing.assert_allclose(
        nrcs.cmod7(
            ds.wspd,
            ds.wdir,
            ds.incidenceImage,
            1,
        ),
        xr.DataArray(
            data=np.array([[np.nan, 0.023, 0.010],
                           [0.434, 0.057, 0.017],
                           [0.646, 0.157, 0.096],
                           [1.410, np.nan, np.nan]]),
            dims=(['x', 'y'])
        ),
        rtol=1e-1
    )


