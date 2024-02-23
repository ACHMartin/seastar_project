import pytest
import seastar.tests.retrieval.test_spatial_ambiguity_selection as test_spatial_ambiguity_selection
import numpy as np
import xarray as xr


@pytest.fixture
def L2():
    # Create a sample L2 OSCAR dataset
    values = np.array([[3, 3, 3], [3, 3, 3], [3, 3, 3]])
    return xr.Dataset(
        {
            "CurrentU": xr.DataArray(
                values, dims=("CrossRange", "GroundRange")
            ),
            "CurrentV": xr.DataArray(
                values, dims=("CrossRange", "GroundRange")
            ),
            "EarthRelativeWindU": xr.DataArray(
                values, dims=("CrossRange", "GroundRange")
            ),
            "EarthRelativeWindV": xr.DataArray(
                values, dims=("CrossRange", "GroundRange")
            ),
        }
    )


@pytest.fixture
def lmout_single():
    # Create a sample L2 OSCAR dataset with 1 cell and it's ambiguities
    values = np.array([1, 2, 3, 4])
    return xr.Dataset(
        {
            "CurrentU": xr.DataArray(
                values, dims=("Ambiguities")
            ),
            "CurrentV": xr.DataArray(
                values, dims=("Ambiguities")
            ),
            "EarthRelativeWindU": xr.DataArray(
                values, dims=("Ambiguities")
            ),
            "EarthRelativeWindV": xr.DataArray(
                values, dims=("Ambiguities")
            ),
        }
    )


@pytest.fixture
def initial():
    # Create a sample L2 OSCAR dataset
    values = np.zeros((5, 4))
    return xr.Dataset(data_vars=dict(CurrentU=(["CrossRange", "GroundRange"], values),
                                     CurrentV=(
                                         ["CrossRange", "GroundRange"], values),
                                     EarthRelativeWindU=(
                                         ["CrossRange", "GroundRange"], values),
                                     EarthRelativeWindV=(["CrossRange", "GroundRange"], values)),
                      coords=dict(CrossRange=np.arange(5), GroundRange=np.arange(4)))


@pytest.fixture
def lmout():
    # Create a sample L2 OSCAR dataset
    values = np.zeros((4, 5, 4))
    values[2, :, :] = np.ones((5, 4))
    return xr.Dataset(data_vars=dict(CurrentU=(["Ambiguities", "CrossRange", "GroundRange"], values),
                                     CurrentV=(
                                         ["Ambiguities", "CrossRange", "GroundRange"], values),
                                     EarthRelativeWindU=(
                                         ["Ambiguities", "CrossRange", "GroundRange"], values),
                                     EarthRelativeWindV=(["Ambiguities", "CrossRange", "GroundRange"], values)),
                      coords=dict(Ambiguities=np.arange(4), CrossRange=np.arange(5), GroundRange=np.arange(4)))


def test_squared_Euclidian_distance(L2, lmout_single):
    # Test the squared Euclidian distance cost function
    cost = test_spatial_ambiguity_selection.squared_Euclidian_distance(
        lmout_single, L2, 1)
    assert (cost == [144., 36., 0., 36.]).all()


def test_Euclidian_distance(L2, lmout_single):
    # Test the Euclidian distance cost function
    cost = test_spatial_ambiguity_selection.Euclidian_distance(
        lmout_single, L2, 1)
    assert (cost == [54., 18., 0., 18.]).all()


def cost(L2_sel, L2_neighbours, weight):
    return np.array([3, 2, 0, 1])


@pytest.mark.parametrize("i_x, i_y", [(0, 0), (3, 2), (3, 3), (4, 2), (4, 3)])
def test_select_ambiguity_with_lowest_cost(lmout, initial, i_x, i_y):
    # Test the selection of ambiguity with the lowest cost
    selected_ambiguity = test_spatial_ambiguity_selection.select_ambiguity_with_lowest_cost(
        lmout, initial, i_x, i_y, cost, weight=2, box_size=3)
    assert selected_ambiguity == 2


def test_solve_ambiguity(lmout, initial):
    # Test the solve ambiguity function
    L2_solved = test_spatial_ambiguity_selection.solve_ambiguity(
        lmout, initial, cost, pass_number=1, weight=5, box_size=3)
    ones = np.ones((5, 4))
    assert (L2_solved.CurrentU.values == ones).all()