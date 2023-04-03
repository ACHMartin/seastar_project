# -*- coding: utf-8 -*-
"""
Created on December 2022

@author: admartin, dlmccann
"""

import numpy as np
import xarray as xr
from scipy.optimize import least_squares
import seastar
from seastar.utils.tools import dotdict, da2py

# import seastar.gmfs.doppler
# import pdb # pdb.set_trace() # where we want to start to debug


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
         "IncidenceAngleImage", "RSV", "Sigma0"
    noise : ``xarray.Dataset``
        Noise DataSet with fields "RSV" and "Sigma0" of the same size as level1
    gmf: dict or dotdict
        dictionary with gmf['nrcs']['name'] and gmf['doppler']['name'] items.
        cf compute_nrcs and compute_wasv for the gmf input
    Returns
    -------
    out : ``numpy.array``
        numpy array of size level1.isel(Antenna=0).shape times the sum of Antenna (observation) dimension of Sigma0 + RSV.
        if 4 antennas (Fore, Aft, MidVV, MidHH) for Sigma0 and RSV => 8
        NaN are replaced by 0
    """
    # Initialisation
    u = da2py(variables[0])
    v = da2py(variables[1])
    c_u = da2py(variables[2])
    c_v = da2py(variables[3])

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
    model_rsv_list = [None] * level1.Antenna.size
    for aa, ant in enumerate(level1.Antenna.data):
        # print(aa, ant)
        model_rsv_list[aa] = seastar.gmfs.doppler\
            .compute_total_surface_motion(level1.sel(Antenna=ant),
                                          geo,
                                          gmf=gmf['doppler']['name'])
        # print(model_rsv_list[aa])
    model['RSV'] = xr.concat(model_rsv_list, dim='Antenna')
    # in future it should be: model['RSV'] = seastar.gmfs.doppler.compute_total_surface_motion(level1, geo, gmf=gmf.doppler) without the loop on antennas

    model['Sigma0'] = seastar.gmfs.nrcs.compute_nrcs(level1, geo, gmf['nrcs'])

    res = ( level1 - model ) / noise # DataSet with RSV and Sigma0 fields

    sigma0_axis_num = level1.Sigma0.get_axis_num('Antenna')
    rsv_axis_num = level1.RSV.get_axis_num('Antenna')
    if sigma0_axis_num == rsv_axis_num:
        concat_axis = sigma0_axis_num
    else:
        raise Exception('Different axis in Antenna for Sigma0 and RSV')

    out = np.concatenate(
        (res.Sigma0.data, res.RSV.data),
        axis=concat_axis,
    )

    return np.where(np.isfinite(out), out, 0)

def find_minima(level1_pixel, noise_pixel, gmf):

    opt = {
        'method': 'trf', # Trust Region Reflective algorithm, particularly suitable for large sparse problems with bounds. Generally robust method.
        'xtol':  1e-3, # Tolerance for termination by the change of the independent variables
        'x_scale': [7,7,.5,.5], # Characteristic scale of each variable.
        'bounds': ([-30,-30,-5,-5], [30,30,5,5]),
    } # if modified => change optionLeastSquares2dataset() below

    init = [None] * 4
    init[0] = dotdict({
        'x0': [ 7*np.random.normal(), #u
                7*np.random.normal(), #v
                0, # c_u
                0 ]  # c_v
    })

    lmout = [None] * 4
    # find the first minimum with begin current = 0
    lmout_dict = least_squares(
        seastar.retrieval.cost_function.fun_residual,
        init[0].x0,
        args=(level1_pixel, noise_pixel, gmf),
        **opt
    )
    lmout[0] = optimizeResults2dataset(lmout_dict, init[0].x0, level1_pixel)

    # find the 3 ambiguities and run the minimisation to find the 3 minima
    init[1:3] = find_initial_values(lmout[0].x, level1_pixel, gmf)
    for ii in [1,2,3]:
        lmout_dict = least_squares(
            seastar.retrieval.cost_function.fun_residual,
            init[ii].x0,
            args=(level1_pixel, noise_pixel, gmf),
            **opt
        )
        lmout[ii] = optimizeResults2dataset(lmout_dict, init[ii].x0, level1_pixel)

    dslmout = xr.concat(lmout, dim='Ambiguities')
    dslmout = optionLeastSquares2dataset(opt, dslmout)
    dslmout = dslmout.assign_coords(level1_pixel.coords)

    return dslmout

def x2uvcucv(x):
    """
    Convert the array x = [u, v, c_u, c_v] to a dict with .u, .v, .c_u, .c_v elements

    Parameters
    ----------
    x : ``list`` ``numpy.array``
       array x = [u, v, c_u, c_v]

    Returns
    -------
    out : ``dotdict``
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
    out : ``dict`` or ``dotdict``
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

def optionLeastSquares2dataset(opt, dslmout):
    dslmout['method'] = opt['method']
    dslmout['xtol'] = opt['xtol']
    dslmout['x_scale'] = (('x_variables'), opt['x_scale'])
    dslmout['bounds'] = (('extrema','x_variables'), np.asarray(opt['bounds']))

    # description from
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
    dslmout.method.attrs['description'] = \
        'Algorithm to perform minimization. ' \
        '‘trf’ : Trust Region Reflective algorithm, particularly suitable for ' \
        'large sparse problems with bounds. Generally robust method.' \
        '‘dogbox’ : dogleg algorithm with rectangular trust regions, typical use ' \
        'case is small problems with bounds. Not recommended for problems with ' \
        'rank-deficient Jacobian.' \
        '‘lm’ : Levenberg-Marquardt algorithm as implemented in MINPACK. ' \
        'Doesn’t handle bounds and sparse Jacobians. Usually the most efficient ' \
        'method for small unconstrained problems'
    dslmout.xtol.attrs['description'] = \
        'Tolerance for termination by the change of the independent variables.' \
        ' Default is 1e-8. The exact condition depends on the method used:' \
        'For ‘trf’ and ‘dogbox’ : norm(dx) < xtol * (xtol + norm(x)).' \
        'For ‘lm’ : Delta < xtol * norm(xs), where Delta is a trust-region radius ' \
        'and xs is the value of x scaled according to x_scale parameter (see below).' \
        'If None and ‘method’ is not ‘lm’, the termination by this condition is disabled.' \
        ' If ‘method’ is ‘lm’, this tolerance must be higher than machine epsilon'
    dslmout.x_scale.attrs['description'] = \
        'Characteristic scale of each variable. Setting x_scale is equivalent to ' \
        'reformulating the problem in scaled variables xs = x / x_scale. An alternative ' \
        'view is that the size of a trust region along jth dimension is proportional ' \
        'to x_scale[j]. Improved convergence may be achieved by setting x_scale such that' \
        ' a step of a given size along any of the scaled variables has a similar effect on' \
        ' the cost function. If set to ‘jac’, the scale is iteratively updated using the ' \
        'inverse norms of the columns of the Jacobian matrix (as described in [JJMore]).'
    dslmout.bounds.attrs['description'] = \
        'Lower and upper bounds on independent variables. Defaults to no bounds. ' \
        'Each array must match the size of x0 or be a scalar, in the latter case a bound ' \
        'will be the same for all variables. ' \
        'Use np.inf with an appropriate sign to disable bounds on all or some variables.'

    return dslmout

def optimizeResults2dataset(lmout, x0, level1):
    d = dict()
    d['x_variables'] = {"dims": "x_variables", "data": np.array(['u', 'v', 'c_u', 'c_v'])}
    d['Observables'] = {"dims": "Observables", "data": np.array(['sigma0', 'RSV'])}
    # d['fun_variables'] = {"dims": "fun_variables", "data": range(8)}
    d['Antenna'] = {"dims": "Antenna", "data": level1.Antenna.data}
    d['fun_variables'] = {"dims": ("Observables", "Antenna"),
                          "data": np.array([range(0,len(d['Antenna']['data'])),
                                            range(len(d['Antenna']['data']), len(d['Antenna']['data'])*len(d['Observables']['data']))])
                          }

    # import variables with dimension
    dims_variables = {
        'x': 'x_variables',
        'active_mask': 'x_variables',
        'grad': 'x_variables',
        'fun': ('Observables', 'Antenna'),
        'jac': ('fun_variables', 'x_variables'),
    }
    for var in dims_variables.keys():
        if var in ['fun','jac']:
            d[var] = {
                "dims": dims_variables[var],
                "data": lmout[var].reshape(
                    d[dims_variables[var][0]]['data'].size,
                    d[dims_variables[var][1]]['data'].size
                )
            }
        else:
            d[var] = {"dims": (dims_variables[var]), "data": lmout[var]}

    ds = xr.Dataset.from_dict(d)

    # add x0
    ds['x0'] = (('x_variables'), x0)


    # import variables without dimension
    nodims_variables = ['cost', 'optimality', 'nfev', 'njev', 'status', 'message', 'success']
    for var in nodims_variables:
        ds[var] = lmout[var]

    # description from
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
    ds.x0.attrs['description'] = \
        'Initial guess on independent variables'
    ds.x.attrs['description'] = \
        'Solution found'
    ds.cost.attrs['description'] = \
        'Value of the cost function at the solution'
    ds.fun.attrs['description'] = \
        'Vector of residuals at the solution'
    ds.jac.attrs['description'] = \
        'Modified Jacobian matrix at the solution, in the sense that J^T J is a Gauss-' \
        'Newton approximation of the Hessian of the cost function. The type is the same ' \
        'as the one used by the algorithm'
    ds.grad.attrs['description'] = \
        'Gradient of the cost function at the solution'
    ds.optimality.attrs['description'] = \
        'First-order optimality measure. In unconstrained problems, it is always the ' \
        'uniform norm of the gradient. In constrained problems, it is the quantity ' \
        'which was compared with gtol during iterations'
    ds.active_mask.attrs['description'] = \
        'Each component shows whether a corresponding constraint is active ' \
        '(that is, whether a variable is at the bound):' \
        '0 : a constraint is not active.; -1 : a lower bound is active.; 1 : an upper bound is active.' \
        'Might be somewhat arbitrary for ‘trf’ method as it generates a sequence of ' \
        'strictly feasible iterates and active_mask is determined within a tolerance threshold. '
    ds.nfev.attrs['description'] = \
        'Number of function evaluations done. Methods ‘trf’ and ‘dogbox’ do not ' \
        'count function calls for numerical Jacobian approximation, as opposed to ‘lm’ method.'
    ds.njev.attrs['description'] = \
        'Number of Jacobian evaluations done. If numerical Jacobian approximation ' \
        'is used in ‘lm’ method, it is set to None'
    ds.status.attrs['description'] = \
        'The reason for algorithm termination: ' \
        '-1 : improper input parameters status returned from MINPACK.' \
        '0 : the maximum number of function evaluations is exceeded.' \
        '1 : gtol termination condition is satisfied.' \
        '2 : ftol termination condition is satisfied.' \
        '3 : xtol termination condition is satisfied.' \
        '4 : Both ftol and xtol termination conditions are satisfied.'
    ds.message.attrs['description'] = \
        'Verbal description of the termination reason'
    ds.success.attrs['description'] = \
        'True if one of the convergence criteria is satisfied (status > 0)'

    return ds

def find_initial_values(sol1st_x, level1_inst, gmf):
    # find_initial_values( lmout[0].x,  level1_conf, gmf) # gmf??? or default Mouche?
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
    out : list of dotdict.x0 containing [u,v,c_u,c_v]
        list of the 3 new initial values to look for ambiguities
    """

    # Intern Function initialisation TODO to update using the original matlab code below
    # TODO, instead of this function, we can calculate the wind ambiguities;
    #  calculate the WASV component for each, then current = total_motion - wasv
    WS = np.array([5,10])
    dte_coef = 0.03 # to update with matlab code below diff(WASV)/diff(WS)
    def smooth(x):
        if np.abs(x) < 0.5: #3:
            return(x/3)
        else:
            return(x)
    # dte = lambda x: smooth(x) * np.sign(x) * dte_coef * (np.abs(x) - WS[0])
    dte = lambda x: -np.sign(x) * smooth ( 0.5 + dte_coef * (np.abs(x) - WS[0]) )



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
        init['vis_wdir'] = np.mod( sol['vis_wdir'] + (ii+1)*90, 360)
        init['vis_u'], init['vis_v'] = \
            seastar.utils.tools.windSpeedDir2UV(
                init['vis_wspd'], init['vis_wdir']
            )
        init['c_u'] = meas_cur['c_u'] - dte(init['vis_u'])
        init['c_v'] = meas_cur['c_v'] - dte(init['vis_v'])
        init['u'] = init['vis_u'] + init['c_u']
        init['v'] = init['vis_v'] + init['c_v']
        init_list[ii] = \
            dotdict({
                'x0': uvcucv2x(init)
            })
        # test boundary # TODO properly from least_square/opt in future # with raise Exception
        # print(init_list[ii])
        if ((init_list[ii]['x0'] - np.array([-30,-30,-5,-5])) < 0).any():
            print('below boundary')
            print(init_list[ii], init)
        if ((init_list[ii]['x0'] - np.array([30,30,5,5])) > 0).any():
            print('above boundary')
            print(init_list[ii], init)


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

    print("To Be Done find_initial_value")

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
#             RSV=( ['across','along','Antenna'], np.full([10,10,4], 0.5) ),
#             dRSV=( ['across','along','Antenna'], np.full([10,10,4], 0.01) ),
#         )
#
#     )
#
#
#     return
