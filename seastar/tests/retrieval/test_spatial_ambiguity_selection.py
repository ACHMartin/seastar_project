"""Testing of spatial_ambiguity_selection"""

import pytest
import numpy as np
import xarray as xr
import numpy.testing as npt
import seastar.retrieval.spatial_ambiguity_selection as spatial_ambiguity_selection


@pytest.fixture
def L2_small2D():
    """Create a sample L2 OSCAR dataset"""
    values = np.array([[3, 3, 3], [3, 3, 3], [3, 3, 3]])
    return xr.Dataset(
        data_vars={
            "CurrentU": (["CrossRange", "GroundRange"], values),
            "CurrentV": (["CrossRange", "GroundRange"], values),
            "EarthRelativeWindU": (["CrossRange", "GroundRange"], values),
            "EarthRelativeWindV": (["CrossRange", "GroundRange"], values),
        },
        coords={"CrossRange": np.arange(3), "GroundRange": np.arange(3)},
    )


@pytest.fixture
def initial():
    """Create a sample L2 OSCAR dataset"""
    values = np.full((5, 4), 2)
    return xr.Dataset(
        data_vars={
            "CurrentU": (["CrossRange", "GroundRange"], values),
            "CurrentV": (["CrossRange", "GroundRange"], values),
            "EarthRelativeWindU": (["CrossRange", "GroundRange"], values),
            "EarthRelativeWindV": (["CrossRange", "GroundRange"], values),
        },
        coords={"CrossRange": np.arange(5), "GroundRange": np.arange(4)},
    )


@pytest.fixture
def lmout():
    """Create a sample L2 OSCAR dataset"""
    values = np.zeros((4, 5, 4))
    values[1, :, :] = np.full((5, 4), 1)
    values[2, :, :] = np.full((5, 4), 2)
    values[3, :, :] = np.full((5, 4), 3)
    return xr.Dataset(
        data_vars={
            "CurrentU": (["Ambiguities", "CrossRange", "GroundRange"], values),
            "CurrentV": (["Ambiguities", "CrossRange", "GroundRange"], values),
            "EarthRelativeWindU": (
                ["Ambiguities", "CrossRange", "GroundRange"],
                values,
            ),
            "EarthRelativeWindV": (
                ["Ambiguities", "CrossRange", "GroundRange"],
                values,
            ),
        },
        coords={
            "Ambiguities": np.arange(4),
            "CrossRange": np.arange(5),
            "GroundRange": np.arange(4),
        },
    )


def test_calculate_Euclidian_distance_to_neighbours(L2_small2D, lmout):
    """Test the Euclidian distance distance function without centre cell"""
    total_distance = (
        spatial_ambiguity_selection.calculate_Euclidian_distance_to_neighbours(
            lmout.isel(GroundRange=1, CrossRange=1),
            L2_small2D,
            windcurrentratio=1,
            include_centre=False,
            Euclidian_method="standard",
        )
    )
    npt.assert_array_almost_equal(
        total_distance, [67.88225100, 45.25483400, 22.62741700, 0.0]
    )


def test_calculate_Euclidian_distance_to_neigbours_and_centre(L2_small2D, lmout):
    """Test the Euclidian distance distance function with centre cell included"""
    total_distance = (
        spatial_ambiguity_selection.calculate_Euclidian_distance_to_neighbours(
            lmout.isel(GroundRange=1, CrossRange=1),
            L2_small2D,
            windcurrentratio=1,
            include_centre=True,
            Euclidian_method="standard",
        )
    )
    npt.assert_array_almost_equal(
        total_distance, [76.36753237, 50.91168825, 25.45584412, 0.0]
    )


def test_calculate_squared_Euclidian_distance_to_neighbours(L2_small2D, lmout):
    """Test the squared Euclidian distance distance function without centre cell"""
    total_distance = (
        spatial_ambiguity_selection.calculate_Euclidian_distance_to_neighbours(
            lmout.isel(GroundRange=1, CrossRange=1),
            L2_small2D,
            windcurrentratio=1,
            include_centre=False,
            Euclidian_method="squared",
        )
    )
    assert (total_distance == [288.0, 128.0, 32.0, 0.0]).all()


def test_calculate_squared_Euclidian_distance_to_neighbours_and_centre(
    L2_small2D, lmout
):
    """Test the squared Euclidian distance distance function with centre cell included"""
    total_distance = (
        spatial_ambiguity_selection.calculate_Euclidian_distance_to_neighbours(
            lmout.isel(GroundRange=1, CrossRange=1),
            L2_small2D,
            windcurrentratio=1,
            include_centre=True,
            Euclidian_method="squared",
        )
    )
    assert (total_distance == [324.0, 144.0, 36.0, 0.0]).all()


@pytest.mark.parametrize("i_x, i_y", [(0, 0), (3, 2), (3, 3), (4, 2), (4, 3)])
def test_single_cell_ambiguity_selection(lmout, initial, i_x, i_y):
    """Test the selection of ambiguity with the lowest distance"""
    selected_ambiguity = spatial_ambiguity_selection.single_cell_ambiguity_selection(
        lmout, initial, i_x, i_y, window=3
    )
    assert selected_ambiguity == 2


def test_solve_ambiguity_spatial_selection(lmout, initial):
    """Test the solve ambiguity function"""
    L2_solved = spatial_ambiguity_selection.solve_ambiguity_spatial_selection(
        lmout, initial, iteration_number=1, window=3
    )
    correct = np.full((5, 4), 2)
    assert (L2_solved.CurrentU.values == correct).all()
