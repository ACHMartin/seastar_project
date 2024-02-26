import numpy as np
import xarray as xr


def calculate_Euclidian_distance_to_neighbours(L2_sel, L2_neighbours, weight, method='standard'):
    """
    Calculates cost using Euclidian distance or squared Euclidian distancee

    The cost is the Euclidian distance
    between each ambiguity of the cell and its neighbours:
    a sum of current distance*weight and wind distance
    ----------
    L2_sel : ``xarray.dataset``
        OSCAR L2 dataset containing the cell of interest and its ambiguities.
        Must have 'Ambiguities' dimensions,
        and 'CurrentU', 'CurrentV', 'EarthRelativeWindU', 'EarthRelativeWindV' data variables
    L2_neighbours : xarray dataset
        OSCAR L2 dataset containing the neighbours of the cell of interest
        Must have 'Ambiguities', 'CrossRange' and 'GroundRange' dimensions,
        and 'CurrentU', 'CurrentV', 'EarthRelativeWindU', 'EarthRelativeWindV' data variables
    weight : float
        Weight for the current cost
    method: string
        Method to use for the cost function
        Default is 'standard'
        Options are 'standard' and 'squared'
    Returns
    -------
    TotalCost : ``numpy.array``
        Total cost for each ambiguity
    """
    if method == 'standard':
        power = 0.5
    elif method == 'squared':
        power = 1
    else:
        raise ValueError('Method must be either standard or squared')

    total_cost = np.array([0, 0, 0, 0])
    cross_range_size = L2_neighbours.CrossRange.sizes['CrossRange']
    ground_range_size = L2_neighbours.GroundRange.sizes['GroundRange']
    for i in range(cross_range_size):
        for j in range(ground_range_size):
            L2_comp_neighbour = L2_neighbours.isel(CrossRange=i, GroundRange=j)
            if not np.isnan(L2_comp_neighbour.CurrentU):
                # iterate through all 4 ambiguities
                for iambiguity in range(0, 4):
                    # find cost for current
                    current_distance = (
                        (L2_sel.sel(Ambiguities=iambiguity).CurrentU.values
                         - L2_comp_neighbour.CurrentU.values)**2
                        + (L2_sel.sel(Ambiguities=iambiguity).CurrentV.values
                            - L2_comp_neighbour.CurrentV.values)**2)**power
                    # find cost for wind
                    wind_distance = (
                        (L2_sel.sel(Ambiguities=iambiguity).EarthRelativeWindU.values
                         - L2_comp_neighbour.EarthRelativeWindU.values)**2
                        + (L2_sel.sel(
                            Ambiguities=iambiguity).EarthRelativeWindV.values
                            - L2_comp_neighbour.EarthRelativeWindV.values)**2)*power
                    total_cost[iambiguity] += weight*current_distance + \
                        wind_distance
    total_cost = xr.where(np.isnan(total_cost), np.inf, total_cost)
    return total_cost


def calculate_squared_Euclidian_distance_to_neighbours(L2_sel, L2_neighbours, weight):
    return calculate_Euclidian_distance_to_neighbours(L2_sel, L2_neighbours, weight, method='squared')

def single_cell_ambiguity_selection(lmout, initial, i_x, i_y, cost_function,
                                    weight, box_size):
    """
    Selects the ambiguity with the lowest cost function value based on a box around the cell

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
    i_j : ``int``
        Index of the `GroundRange` dimension
    cost_function : ``function``
        Function to calculate the cost of the ambiguities.
        Must take single cell from `lmout`, a box around it from `initial` as input, weight
        and return toal cost for all for ambiguities
    weight : ``int``, optional
        Weight for the cost function
        Default is 5
    box_size : ``int``, optional
        Size of the box around the cell
        Must be an odd number
        Default is 3

    Returns
    -------
    selected_ambiguity: ``int``
        Index of the selected ambiguity
    """
    if box_size % 2 == 0:
        raise ValueError('Box size must be an odd number')
    radius = np.int_((box_size-1)/2)
    L2_sel = lmout.isel(CrossRange=i_x, GroundRange=i_y)
    if not np.isnan(L2_sel.isel(Ambiguities=0).CurrentU.values):
        total_cost = cost_function(
            L2_sel, initial.isel(
                CrossRange=slice(i_x-radius, i_x+radius+1),
                GroundRange=slice(i_y-radius, i_y+radius+1)),
            weight)
        selected_ambiguity = total_cost.argmin()
    else:
        selected_ambiguity = np.nan
    return selected_ambiguity


def solve_ambiguity_spatial_selection(lmout, initial, cost_function,
                                      pass_number=2, weight=5, box_size=3):
    """
    Solves the ambiguity of the L2_lmout dataset using the spatial selection method

    Parameters
    ----------
    lmout : ``xarray.Dataset``
        OSCAR L2 lmout dataset
        This dataset contains the ambiguities to be selected from.
        Must have 'Ambiguities', 'CrossRange' and 'GroundRange' dimensions,
        and 'CurrentU', 'CurrentV', 'EarthRelativeWindU', 'EarthRelativeWindV' data variables
    initial : ``xarray.dataset``
        OSCAR L2 dataset
        This dataset contains the initial solution to compare the ambiguities to.
        Must have 'Ambiguities', 'CrossRange' and 'GroundRange' dimensions,
        and 'CurrentU', 'CurrentV', 'EarthRelativeWindU', 'EarthRelativeWindV' data variables
    cost_function : ``function``
        Function to calculate the cost of the ambiguities.
        Must take single cell from `lmout`, a box around it from `initial` as input, weight
        and return total cost for all for ambiguities
    pass_number : ``int``, optional
        Number of passes to iterate through the dataset
        Default is 2
    weight : ``int``, optional
        Weight for the cost function
        Default is 5
    box_size : ``int``, optional
        Size of the box around the cell
        Default is 3
    Returns
    -------
    L2: ``xarray.Dataset``
        OSCAR L2 dataset with solved ambiguities
    """
    def select_and_replace_ambiguity(i, j):
        # select ambiguity with the lowest cost
        selected_ambiguity = single_cell_ambiguity_selection(
            lmout,
            initial,
            i,
            j,
            cost_function=cost_function,
            weight=weight,
            box_size=box_size)
        # replace with the selected ambiguity if it is not nan
        if not np.isnan(selected_ambiguity):
            initial.loc[{
                'CrossRange': initial.CrossRange.isel(CrossRange=i),
                'GroundRange': initial.GroundRange.isel(GroundRange=j)
            }] = lmout.isel(
                CrossRange=i,
                GroundRange=j,
                Ambiguities=selected_ambiguity
            )

    # initialize arrays
    cross_range_size = lmout.CrossRange.sizes['CrossRange']
    ground_range_size = lmout.GroundRange.sizes['GroundRange']
    halfway_ground_range = np.round(ground_range_size/2).astype(int)

    for n in range(pass_number):  # repeat passes
        print('Pass', n+1)
        # Pass A: iterate vertically
        j = halfway_ground_range
        while j >= 0:  # iterate across track
            for i in range(0, cross_range_size):  # iterate along track
                select_and_replace_ambiguity(i, j)
            if j == ground_range_size-1:
                j = halfway_ground_range-1
            elif j >= halfway_ground_range:
                j += 1
            elif j < halfway_ground_range:
                j -= 1
        # Pass B: iterate horizontally
        for i in range(0, cross_range_size):  # iterate along track
            for j in range(0, ground_range_size):  # iterate across track
                select_and_replace_ambiguity(i, j)
    return initial
