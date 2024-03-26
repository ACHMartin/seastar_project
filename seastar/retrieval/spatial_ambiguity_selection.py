import numpy as np


def calculate_Euclidian_distance_to_neighbours(
    L2_sel,
    L2_neighbours,
    Euclidian_method="standard",
    method="windcurrent",
    windcurrentratio=10,
    include_centre=False,
):
    """
    Calculates distance using Euclidian distance or squared Euclidian distancee

    The distance is the Euclidian distance
    between each ambiguity of the cell and its neighbours:
    a sum of current distance*current_weight and wind distance
    ----------
    L2_sel : ``xarray.dataset``
        OSCAR L2 dataset containing the cell of interest and its ambiguities.
        Must have 'Ambiguities' dimensions,
        and 'CurrentU', 'CurrentV', 'EarthRelativeWindU', 'EarthRelativeWindV' data variables
    L2_neighbours : xarray dataset
        OSCAR L2 dataset containing the neighbours of the cell of interest
        Must have 'Ambiguities', 'CrossRange' and 'GroundRange' dimensions,
        and'CurrentU', 'CurrentV', 'EarthRelativeWindU', 'EarthRelativeWindV' data variables
        Additional keyword arguments to pass to the distance function
    Euclidian_method : ``str``, optional
        Method to calculate the Euclidian distance
        Must be 'standard' or 'squared'
        Default is 'standard'
    method : ``str``, optional
        Method to calculate the distance
        Must be 'windcurrent', 'wind' or 'current'
        Default is 'windcurrent'
    windcurrentratio : ``int``, optional
        Ratio of the weight of the current to the weight of the wind
        Default is 10
    include_centre : ``bool``, optional
        Whether to include the centre cell in the distance function
        Default is False
    Returns
    -------
    dif_squared.distsum : ``xarray.dataarray``
        Dataset containing the sum of the distances between the cell and its neighbours
    """
    if Euclidian_method == "standard":
        power = 0.5
    elif Euclidian_method == "squared":
        power = 1
    else:
        raise ValueError("Euclidian_method must be 'standard' or 'squared'")

    if method == "windcurrent":
        current_multiplier = windcurrentratio
        wind_multiplier = 1
    elif method == "wind":
        current_multiplier = 0
        wind_multiplier = 1
    elif method == "current":
        current_multiplier = 1
        wind_multiplier = 0
    else:
        raise ValueError("method must be 'windcurrent', 'wind' or 'current'")

    centre_cross = np.int_(L2_neighbours.CrossRange.sizes["CrossRange"] / 2)
    centre_ground = np.int_(L2_neighbours.GroundRange.sizes["GroundRange"] / 2)

    dif_squared = (L2_neighbours - L2_sel) ** 2
    dif_squared["dist"] = (
        current_multiplier * (dif_squared.CurrentU + dif_squared.CurrentV) ** power
        + wind_multiplier
        * (dif_squared.EarthRelativeWindU + dif_squared.EarthRelativeWindV) ** power
    )
    dif_squared["distsum"] = dif_squared.dist.sum(dim=("CrossRange", "GroundRange"))

    if not include_centre:
        dif_squared["distsum"] = dif_squared.distsum - dif_squared.dist.isel(
            CrossRange=centre_cross, GroundRange=centre_ground
        )

    return dif_squared.distsum


def single_cell_ambiguity_selection(lmout, initial, i_x, i_y, window, **kwargs):
    """
    Selects the ambiguity with the lowest distance function value
    based on a box around the cell

    Parameters
    ----------
    lmout : ``xarray.dataset``
        OSCAR L2 lmout dataset
        This dataset contains the ambiguities to be selected from.
        Must have 'Ambiguities', 'CrossRange' and 'GroundRange' dimensions,
        and 'CurrentU', 'CurrentV', 'EarthRelativeWindU', 'EarthRelativeWindV' data variables
    initial : ``xarray.dataset``
        OSCAR L2 dataset
        This dataset contains the initial solution to compare the ambiguities to.
        Must have 'CrossRange' and 'GroundRange' dimensions,
        and 'CurrentU', 'CurrentV', 'EarthRelativeWindU', 'EarthRelativeWindV' data variables
    i_x : ``int``
        Index of the `CrossRange` dimension
    i_y : ``int``
        Index of the `GroundRange` dimension
    window : ``int``, optional
        Size of the box around the cell
        Must be an odd number
    **kwargs : ``**kwargs``, optional
        Additional keyword arguments to pass to the distance function

    Returns
    -------
    selected_ambiguity: ``int``
        Index of the selected ambiguity
    """
    if window % 2 == 0:
        raise ValueError("Window size must be an odd number")
    radius = np.int_((window - 1) / 2)
    L2_sel = lmout.isel(CrossRange=i_x, GroundRange=i_y)
    if not np.isnan(L2_sel.isel(Ambiguities=0).CurrentU.values):
        if i_x - radius >= 0:
            CrossRange_slice = slice(i_x - radius, i_x + radius + 1)
        else:
            CrossRange_slice = slice(0, i_x + radius + 1)
        if i_y - radius >= 0:
            GroundRange_slice = slice(i_y - radius, i_y + radius + 1)
        else:
            GroundRange_slice = slice(0, i_y + radius + 1)
        total_distance = calculate_Euclidian_distance_to_neighbours(
            L2_sel,
            initial.isel(
                CrossRange=CrossRange_slice,
                GroundRange=GroundRange_slice,
            ),
            **kwargs
        )
        selected_ambiguity = total_distance.argmin()
    else:
        selected_ambiguity = np.nan
    return selected_ambiguity


def solve_ambiguity_spatial_selection(
    lmout, initial_solution, iteration_number=2, window=3, inplace=True, **kwargs
):
    """
    Solves the ambiguity of the L2_lmout dataset using the spatial selection method

    Parameters
    ----------
    lmout : ``xarray.Dataset``
        OSCAR L2 lmout dataset
        This dataset contains the ambiguities to be selected from.
        Must have 'Ambiguities', 'CrossRange' and 'GroundRange' dimensions,
        and 'CurrentU', 'CurrentV', 'EarthRelativeWindU', 'EarthRelativeWindV' data variables
    initial_solution : ``xarray.dataset``
        OSCAR L2 dataset
        This dataset contains the initial solution to compare the ambiguities to.
        Must have 'Ambiguities', 'CrossRange' and 'GroundRange' dimensions,
        and 'CurrentU', 'CurrentV', 'EarthRelativeWindU', 'EarthRelativeWindV' data variables
    distance_function : ``function``
        Function to calculate the distance of the ambiguities.
        Must take:
            single cell from `lmout`
            a box around it from `initial`
            any additional keyword arguments
        and return total distance for all 4 ambiguities
    pass_number : ``int``, optional
        Number of passes to iterate through the dataset
        Default is 2
    inplace : ``bool``, optional
        Whether to modify the input dataset in place
        Default is True
    **kwargs : ``**kwargs``, optional
        Additional keyword arguments to pass to the distance function
    Returns
    -------
    initial_copy : ``xarray.dataset``
        Dataset containing the selected ambiguities
    """

    def select_and_replace_ambiguity(i, j):
        # select ambiguity with the lowest distance
        selected_ambiguity = single_cell_ambiguity_selection(
            lmout, initial_copy, i, j, window=window, **kwargs
        )
        # replace with the selected ambiguity if it is not nan
        if not np.isnan(selected_ambiguity):
            initial_copy.loc[
                {
                    "CrossRange": initial_copy.CrossRange.isel(CrossRange=i),
                    "GroundRange": initial_copy.GroundRange.isel(GroundRange=j),
                }
            ] = lmout.isel(CrossRange=i, GroundRange=j, Ambiguities=selected_ambiguity)

    def verticalpass(direction):
        # iterate vertically in the given direction
        j = halfway_ground_range
        while j >= 0:  # iterate across track
            for i in range(0, cross_range_size, direction):  # iterate along track
                select_and_replace_ambiguity(i, j)
            if j == ground_range_size - 1:
                j = halfway_ground_range - 1
            elif j >= halfway_ground_range:
                j += 1
            elif j < halfway_ground_range:
                j -= 1

    def horizontalpass(direction):
        for i in range(0, cross_range_size):  # iterate along track
            for j in range(0, ground_range_size, direction):  # iterate across track
                select_and_replace_ambiguity(i, j)

    if inplace:
        initial_copy = initial_solution.copy(deep=False)
    else:
        initial_copy = initial_solution.copy(deep=True)

    # initialize arrays
    cross_range_size = lmout.CrossRange.sizes["CrossRange"]
    ground_range_size = lmout.GroundRange.sizes["GroundRange"]
    halfway_ground_range = np.round(ground_range_size / 2).astype(int)

    for n in range(iteration_number):  # repeat passes
        print("Pass", n + 1)
        # Pass A1: iterate vertically
        verticalpass(1)
        # Pass A2: iterate vertically back
        verticalpass(-1)
        # Pass B1: iterate horizontally
        horizontalpass(1)
        # Pass B2: iterate horizontally back
        horizontalpass(-1)
    return initial_copy
