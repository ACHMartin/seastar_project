# -*- coding: utf-8 -*-
"""
Created on December 2022

@author: admartin, dlmccann
"""

import numpy as np
import xarray as xr
from scipy.optimize import least_squares
import seastar
from seastar.utils.tools import dotdict

# import seastar.gmfs.doppler
import pdb

def wind_current_retrieval(level1):
    """
    Compute ocean surface WIND and CURRENT magnitude and direction
    by minimisation of a cost function.

    Compute Ocean Surface Vector Wind (OSVW) in (m/s) and
    direction (degrees N) in the meteorological convention (coming from).
    Assumed neutral wind at 10m.

    Compute Total Surface Current Vector (TSCV) in (m/s) and
    direction (degrees N) in the oceanographic convention (going to).

     Parameters
     ----------
     level1 : ``xarray.Dataset``
        L1 observable noisy dataset (NRCS, RVL, geometry)

     Returns
     -------
     level2 : xarray.Dataset
         L2 dataset with a new dimension with ambiguity
         L2.shape (ambiguity, x, y)
     level2.CurrentMagnitude : xarray.DataArray
         Magnitude of surface current vector (m/s)
     level2.CurrentDirection : xarray.DataArray
         Surface current direction (degrees N) in oceanographic convention (going to)
    level2.WindSpeed : xarray.DataArray
         Ocean Surface Wind Speed (m/s)
     level2.WindDirection : xarray.DataArray
         Ocean Surface Wind Direction (degrees N) in meteorologic convention (coming from)
     """

    # Wrap Up function for find_minima, should be similar input/output than compute_magnitude...
    print('To be done')

    return

def fun_residual(variables, level1, noise, gmf):
    """
    Function which computes the vector of residuals between
    the observations and the model divide by the noise

    Parameters
    ----------
    variables : ``numpy.array`` others...
        [u, v, c_u, c_v]; u, v, c_u, c_v being floats
    level1 : ``xarray.Dataset``
        level1 dataset with "Antenna" as a dimension, with the mandatory following fields:
         "IncidenceAngleImage", "RVL", "Sigma0"
    noise : ``xarray.Dataset``
        Noise DataSet with fields "RVL" and "Sigma0" of the same size as level1
    gmf: dict or dotdict
        dictionary with gmf['nrcs']['name'] and gmf['doppler']['name'] items.
        cf compute_nrcs and compute_wasv for the gmf input
    Returns
     -------
     out : ``numpy.array``
        numpy array of size level1.isel(Antenna=0).shape times the sum of Antenna (observation) dimension of Sigma0 + RVL.
        if 4 antennas (Fore, Aft, MidVV, MidHH) for Sigma0 and RVL => 8
        NaN are replaced by 0
    """
    # Initialisation
    u = variables[0]
    v = variables[1]
    c_u = variables[2]
    c_v = variables[3]

    #TODO if len(variables)==2; inversion of wind only and c_u=0 cf Matalb and find_simple_minimum
    vis_u = u - c_u
    vis_v = v - c_v

    [vis_wspd, vis_wdir] = seastar.utils.tools.windUV2SpeedDir(vis_u, vis_v)
    [c_vel, c_dir] = seastar.utils.tools.currentUV2VelDir(c_u, c_v)

    model = level1.drop_vars([var for var in level1.data_vars]) # to keep only the coordinates

    geo = xr.Dataset(
        data_vars=dict(
            WindSpeed=(level1.isel(Antenna=0).IncidenceAngleImage.dims, vis_wspd),
            WindDirection=(level1.isel(Antenna=0).IncidenceAngleImage.dims, vis_wdir),
            CurrentVelocity=(level1.isel(Antenna=0).IncidenceAngleImage.dims, c_vel),
            CurrentDirection=(level1.isel(Antenna=0).IncidenceAngleImage.dims, c_dir),
        ),
    )
    # propagate relevant Level1 coords to geo
    for dim in level1.isel(Antenna=0).IncidenceAngleImage.dims:
        geo.coords[dim] = level1.isel(Antenna=0).coords[dim]
    if 'Antenna' in geo.coords:
        geo = geo.drop_vars('Antenna')

    # paragraph below to be changed in future without the loop for Antenna
    model_rvl_list = [None] * level1.Antenna.size
    for aa, ant in enumerate(level1.Antenna.data):
        # print(aa, ant)
        model_rvl_list[aa] = seastar.gmfs.doppler.compute_total_surface_motion(level1.sel(Antenna=ant), geo, gmf=gmf['doppler']['name'])
        # print(model_rvl_list[aa])
    model['RVL'] = xr.concat(model_rvl_list, dim='Antenna')
    # in future it should be: model['RVL'] = seastar.gmfs.doppler.compute_total_surface_motion(level1, geo, gmf=gmf.doppler)

    model['Sigma0'] = seastar.gmfs.nrcs.compute_nrcs(level1, geo, gmf=gmf['nrcs'])

    res = ( level1 - model ) / noise # DataSet with RVL and Sigma0 fields

    sigma0_axis_num = level1.Sigma0.get_axis_num('Antenna')
    rvl_axis_num = level1.RVL.get_axis_num('Antenna')
    if sigma0_axis_num == rvl_axis_num:
        concat_axis = sigma0_axis_num
    else:
        raise Exception('Different axis in Antenna for Sigma0 and RVL')

    out = np.concatenate(
        (res.Sigma0.data, res.RVL.data),
        axis=concat_axis,
    )

    return np.where(np.isfinite(out), out, 0)

def find_minima(level1_pixel, noise_pixel, gmf):

    opt = {
        'method': 'trf', # Trust Region Reflective algorithm, particularly suitable for large sparse problems with bounds. Generally robust method.
        'xtol':  1e-3, # Tolerance for termination by the change of the independent variables
        'x_scale': [7,7,.5,.5], # Characteristic scale of each variable.
        'bounds': ([-30,-30,-5,-5], [30,30,5,5]),
    }

    init = [None] * 4
    init[0] = dotdict({
        'x0': [ 7*np.random.normal(), #u
                7*np.random.normal(), #v
                0, # c_u
                0 ]  # c_v
    })

    lmout = [None] * 4
    # find the first minimum with begin current = 0
    lmout[0] = least_squares(
        seastar.retrieval.cost_function.fun_residual,
        init[0].x0,
        args=(level1_pixel, noise_pixel, gmf),
        **opt
    )

    #nsol or init[1:3] =
    nsol = find_initial_values(lmout[0].x, inst, gmf=gmf)

    # find the 3 ambiguities and run along them to find them
    # level1_conf = level1_pixel.drop_vars([var for var in level1_pixel.data_vars])
    # find_initial_values( lmout[0].x,  level1_conf, gmf) # gmf??? or default Mouche?
    # loop of the 3 new initialisation

    # sort as function of the cost function

    # format the solution as output as u, v, c_u, c_v xarray for different ambiguities

    return lmout

def x2uvcucv(x):
    """
    Convert the array x = [u, v, c_u, c_v] to a dict with .u, .v, .c_u, .c_v elements

    Parameters
    ----------
    x : ``list`` ``numpy.array``
       array x = [u, v, c_u, c_v]
    Returns
    -------
    out : dotdict
        a dictionary with .u, .v, .c_u, .c_v elements
    """
    out = dotdict({
        'u': x[0],
        'v': x[1],
        'c_u': x[2],
        'c_v': x[3]
    })
    return out

def uvcucv2x(mydict):
    """
    Convert a dictionary with ['u'], ['v'], ['c_u'], ['c_v'] elements to the array x = [u, v, c_u, c_v]

    Parameters
    ----------
    out : dict or dotdict
        a dictionary with ['u'], ['v'], ['c_u'], ['c_v'] elements
    Returns
    -------
    x : ``numpy.array``
       array x = [u, v, c_u, c_v]
    """
    x = np.array([
        mydict['u'],
        mydict['v'],
        mydict['c_u'],
        mydict['c_v'],
    ])
    return x

def find_initial_values(sol1st_x, level1_inst, gmf):
    """
    Find the rough position of the ambiguities given a first solution.

    Parameters
    ----------
    sol1st : ``xarray.Dataset``??
       1st solution in term of U, V, C_U, C_V
    level1_inst : ``xarray.Dataset``??
       Instrument characteristics (geometry)
    gmf : dict or dotdict
        dictionary with gmf.nrcs.name and gmf.doppler.name fields
    Returns
    -------
    out : list of "x" array containing [u,v,c_u,c_v]
        list of the 3 new initial values to look for ambiguities
    """

    # Intern Function initialisation TODO to update using the original matlab code below
    WS = np.array([5,10])
    dte_coef = 0.1 # to update with matlab code below diff(WASV)/diff(WS)
    def smooth(x):
        if np.abs(x) < 3:
            return(x/3)
        else:
            return(x)
    dte = lambda x: smooth(x) * np.sign(x) * dte_coef * (np.abs(x) - WS[0])


    sol = x2uvcucv(sol1st_x)
    sol = seastar.utils.tools.windCurrentUV2all(sol)

    init_list = [None] * 3

    meas_cur = dotdict({
        'c_u': ( dte(sol['vis_u']) + sol['c_u'] ),
        'c_v': ( dte(sol['vis_v']) + sol['c_v'] ),
    })

    for ii in range(len(init_list)):
        init = dotdict({})
        init['vis_wspd'] = sol['vis_wspd']
        init['vis_wdir'] = np.mod( sol['vis_wdir'] + ii*90, 360)
        init['vis_u'], init['vis_v'] = \
            seastar.utils.tools.windSpeedDir2UV(
                init['vis_wspd'], init['vis_wdir']
            )
        init['c_u'] = meas_cur['c_u'] - dte(init['vis_u'])
        init['c_v'] = meas_cur['c_v'] - dte(init['vis_v'])
        init_list[ii] = uvcucv2x(init)

    # sol: 1st  solution, type vector: [u v c_u c_v]

#     inst = comb.inst(1);
#
#     % function    to    derive
#     global FREQ_Ku;
#     freq = FREQ_Ku;
#
#     c = 299792458; % m / s in vacuum
#     n = 1.000293;
#     c_air = c / n;
#     lambda_e = c_air / freq;
#
#     WS = [5 10];
#     WASV_up = my_kudop([
#         WS(1) 0 0   inst.inci inst.pol;
#         WS(2) 0 0    inst.inci  inst.pol
#         ], 'attrs', comb
#     ).*lambda_e / 2 / sind(30);
#     WASV_down = my_kudop(
#         [WS(1) 0 180 inst.inci inst.pol;
#         WS(2) 0 180 inst.inci inst.pol],
#         'attrs', comb
#     ).*lambda_e / 2 / sind(30);
#     WASV = (WASV_up - WASV_down) / 2;
#     dte_coef = diff(WASV). / diff(WS);
#
#     smooth =  @(x)[abs(x(~logical(round(x / 3 / 2))) / 3)  ones(logical(round(x / 3 / 2)))];
#     # lineaire  par morceau: y = x / 3( | x | < 3); else y = 1
#     dte =  @(x) smooth(x) * sign(x) * (WASV(1) + dte_coef * (abs(x) - WS(1))); # if x > 0;
#
#     sol.u = sol1st(1);
#     sol.v = sol1st(2);
#     sol.c_u = sol1st(3);
#     sol.c_v = sol1st(4);
#     sol.vis_u = sol.u - sol.c_u;
#     sol.vis_v = sol.v - sol.c_v;
#     [sol.vis_wspd, sol.vis_wdir] = u_v_to_wspd_wdir(sol.vis_u, sol.vis_v);
#
#     meas_cur.u = dte(sol.vis_u) + sol.c_u;
#     meas_cur.v = dte(sol.vis_v) + sol.c_v;
#
#     for ii=1:3
#     nsol(ii).vis_wspd = sol.vis_wspd;
#     nsol(ii).vis_wdir = mod(sol.vis_wdir + ii * 90, 360);
#     [nsol(ii).vis_u, nsol(ii).vis_v] = wind_speed_dir_to_u_v(nsol(ii).vis_wspd, nsol(ii).vis_wdir);
#     nsol(ii).c_u = meas_cur.u - dte(nsol(ii).vis_u);
#     nsol(ii).c_v = meas_cur.v - dte(nsol(ii).vis_v);
#     nsol(ii).u = nsol(ii).vis_u + nsol(ii).c_u;
#     nsol(ii).v = nsol(ii).vis_v + nsol(ii).c_v;
#     nsol(ii).out = [nsol(ii).u nsol(ii).v nsol(ii).c_u nsol(ii).c_v];
#
#
# end

    print("To Be Done")

    return init_list



# def run_find_minima():
#
#     level1 = xr.Dataset(
#         data_vars=dict(
#             IncidenceAngleImage=( ['across','along','Antenna'], np.full([10,10,4], 30) ),
#             LookDirection=(['across', 'along', 'Antenna'],
#                            np.stack((np.full([10, 10], 45),
#                                      np.full([10, 10], 90),
#                                      np.full([10, 10], 90),
#                                      np.full([10, 10], 135)
#                                      ), axis=-1
#                                     )
#                            ),
#             Polarization=(['across', 'along','Antenna'],
#                           np.stack((np.full([10, 10], 1),
#                                     np.full([10, 10], 1),
#                                     np.full([10, 10], 2),
#                                     np.full([10, 10], 1)
#                                     ), axis=-1
#                                    )
#                           ),
#             Sigma0=( ['across','along','Antenna'], np.full([10,10,4], 1.01) ),
#             dsig0=( ['across','along','Antenna'], np.full([10,10,4], 0.05) ),
#             RVL=( ['across','along','Antenna'], np.full([10,10,4], 0.5) ),
#             drvl=( ['across','along','Antenna'], np.full([10,10,4], 0.01) ),
#         )
#
#     )
#
#
#     return
