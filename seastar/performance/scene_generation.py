# -*- coding: utf-8 -*-
"""
Created on February 2023

@author: admartin, dlmccann
"""

import numpy as np
import xarray as xr
from scipy.optimize import least_squares
import seastar
from seastar.utils.tools import dotdict

# import seastar.gmfs.doppler
import pdb


def create_scene_dataset(geo, inst):
    """
    Scene generation function of geophysical parameters (direct models) and acquisition geometry

    Can be run with:
    - a single point
    - a swath of datapoints
    - a 2D field without geographical coordinate
    - a 2D field WITH geographical coordinate

    To Be Discussed: we need to link the ground truth grid with the instrument grid

    Parameters
    ----------
    geo : ``xarray.Dataset``
        Truth in term of:
        - geophysical parameters: WindSpeed, WindDirection, CurrentVel, CurrentDir, others (waves)
        - geographic coordinates: longitude, latitude (optional?)
    inst : ``xarray.Dataset``
        Instrument characteristics for: (every pixel?, or constant for a given swath, or constant everywhere)
        geometry: antenna, incidence angle, azimuth look direction
        uncertainty on: NRCS, (Doppler? or RVL?)


    Returns
    -------
    level1 : ``xarray.Dataset``
        L1 measurements dataset with data: Sigma0 (NRCS), RVL (Radial Velocity) <= truth + noise
        and coordinates: longitude, latitude, antenna, incidence angle, azimuth look direction
        and attributes: (if needed)
        L1.shape (antenna, across, along)
    noise : ``xarray.Dataset``
        same as truth but with noise values and longitude and latitude
    truth :
        same as geo but with direct model measurements and longitude, latitude
        - direct model measurements: NRCS, RVL (including attributes concerning the model used and its attributes)
    """

    gmf = dotdict({'nrcs': dotdict({'name': 'nscat4ds'})})
    gmf['doppler'] = dotdict({'name': 'mouche12'})

    print("To Be Done")


    truth_out = truth_fct(geo, inst)


    return level1, truth_out, inst_out


def truth_fct(geo, inst):
    """

    Should work for all dimension; same behavior than xarray with dimension between geo and inst

   Parameters
    ----------
    geo : ``xarray.Dataset``
        Truth in term of:
        - geophysical parameters: WindSpeed, WindDirection, CurrentVel, CurrentDir, others (waves)
        - geographic coordinates: longitude, latitude (optional?)
    inst : ``xarray.Dataset``
        Instrument characteristics for: (every pixel?, or constant for a given swath, or constant everywhere)
        geometry: antenna, incidence angle, azimuth look direction
        uncertainty on: NRCS, (Doppler? or RVL?)


    Returns
    -------
    truth : ``xarray.Dataset``
        with WindSpeed, wdir, cvel, cdir, u, v, c_u, c_v, vis_u, vis_v,
             sigma0, rvl
    """

    truth_out['Sigma0'] = ss.gmfs.nrcs.compute_nrcs(inst, geo, gmf.nrcs)

    # TODO below, to enable to compute this without a loop through the antenna;
    for aa, ant in enumerate(level1.Antenna.data):
        model_rvl_list[aa] = ss.gmfs.doppler.compute_total_surface_motion(inst.sel(Antenna=ant), geo,
                                                                          gmf=gmf.doppler.name)
    truth_out['RVL'] = xr.concat(model_rvl_list, dim='Antenna')

    # truth.msg_uv = sprintf('u: %1.0f, v: %1.0f; c_u: %2.1f, c_v: %2.1f', truth.u, truth.v, truth.c_u, truth.c_v);
    # truth.msg_vel = sprintf('wind: %2.1f, %3.0f^o, cur.: %2.1f, %3.0f^o', truth.wspd, truth.wdir, truth.cvel,
    #                         truth.cdir);


    return truth



def uncertainty_fct( truth, uncertainty):
    """
    Computed and Filled in all the uncertainty parameters

    Should work for:
    - a point
    - same dimension for truth and uncertainty 1D, 2D, xD (with antenna?)
    - dimension different for truth and uncertainty


    Parameters
    ----------
    truth:
    uncertainty:
    Returns
    -------
    uncertainty : ``xarray.Dataset``
        with all values filled


    """

    print("To Be Done")

    # from matlab_my_toolbox/wind_current_inversion/uncertainty_fct.m
    ##########################
    # uncerty = varargin{1};
    # un = uncerty; % added March 2022 to test consistency between input / output all along in the processes
    # if length(uncerty.N) == 1
    #     uncerty.N = ones(size(truth.dop)) * uncerty.N
    # end
    # if isfield(uncerty, 'NESZ')
    #     SNR = truth.sig0. / db2lin(uncerty.NESZ);
    #     SNR_min = truth.sig0x. / db2lin(uncerty.NESZ);
    #     if ~isfield(uncerty, 'Kp')
    #         uncerty.Kp = (1 + 1. / SNR_min). / sqrt(uncerty.N);
    #         uncerty.Kp_gen = (1 + 1. / SNR). / sqrt(uncerty.N);
    #     end
    #     if ~isfield(uncerty, 'dop')
    #         gamma_SNR_min = 1. / (1 + 1. / SNR_min);
    #         gamma_tot_min = gamma_SNR_min; % ideally * gamma_geometry * gamma_temporal_decorrelation(Wollstadt
    #         et al. 2017, TGRS)
    #         phase_noise_min = 1. / sqrt(2 * uncerty.N). * sqrt(1 - gamma_tot_min. ^ 2). / gamma_tot_min;
    #         uncerty.dop = phase_noise_min. / 2. / pi. / truth.comb.time_lag;
    #         gamma_SNR = 1. / (1 + 1. / SNR);
    #         gamma_tot = gamma_SNR; % ideally * gamma_geometry * gamma_temporal_decorrelation(Wollstadt
    #         et al. 2017, TGRS)
    #         phase_noise = 1. / sqrt(2 * uncerty.N). * sqrt(1 - gamma_tot. ^ 2). / gamma_tot;
    #         uncerty.dop_gen = phase_noise. / 2. / pi. / truth.comb.time_lag;
    #     end
    #     un_sig0_prctg_gen = uncerty.Kp_gen;
    #     un_dop_gen = uncerty.dop_gen;
    # end
    # un_sig0_prctg = uncerty.Kp;
    # un_dop = uncerty.dop;



    return uncertainty




def satellite_looking_geometry(input):
    """
    Computed satellite looking geometry as function of given parameters

    Should work for:
    - a point
    - a swath 1D
    - 2D field?


    Parameters
    ----------


    Returns
    -------
    geometry : ``xarray.Dataset``
        with all values filled


    """

    print("To Be Done")

    # from satellite_looking_geometry.m
    ####################################
    # function out = satellite_looking_geometry(in)
    #    % out = satellite_looking_geometry(in)
    #    %
    #    %%% MANDATORY
    #    % in.altitude = 600*10^3; % in meter
    #    % in.azi_mid  = 90; % in degree
    #    % in.azi_fore = 45; % in degree
    #    %
    #    %%% CHOICES AMONG:
    #    % in.inci_mid = 30; % in degree
    #    % or
    #    % in.inci_fore = 30; % in degree
    #    %
    #    % inci_z => inci_mid
    #    % squint_surf => azi_fore
    #
    #
    #    % R = Earth radius
    #    R = earth_radius; % in metre
    #    % h: altitude
    #    h = in.altitude; % in metre
    #
    #    azi_z =  in.azi_mid; %degree
    #    squint_surf = in.azi_fore; %squint_surf; % degree
    #
    #    % Flat Earth
    #    % inci_zd  = atand( tand(inci_sq) * cosd(squint) )
    #    % inci_sq  = atand( tand(inci_zd) / cosd(squint) )
    #
    #    % Spherical Earth, but not rotating
    #    warning('Spherical Earth, but not rotating');
    #    % Cf Raney 86; to make it rotate
    #
    #    % Solution of triangles (Wikipedia)
    #    % Spherical triangle
    #    %
    #    % SCHEMA 1
    #    %  b _ _ A        C: Nadir point below the satellite at the surface
    #    %   /   \         B: point at the surface where the measurement are done at zero-doppler (across-track)
    #    %  /     \        A: point at the surface for the squint antenna
    #    % C___   | c      c: gamma = <ACB> squint at the surface from the satellite (nadir)
    #    %     \  /        b: beta  = <CBA> angle between the track and the looking direction: 90°
    #    %   a  \/         a: alpha = <BAC> squint angle at the surface: typically +-45°, could be smaller
    #    %       B
    #
    #    beta = 180-azi_z; % degree
    #    alpha = 90-squint_surf;
    #
    #   if isfield(in,'inci_mid')
    #       inci_z = in.inci_mid; % degree % z as zero Doppler for the mid antenna, not necessary at 90°
    #
    #       % Sinus law
    #       % sin(el)/R = sin(pi-inci)/(R+h)
    #       % => sin(inci) = (1+h/R)*sin(el)
    #       % => sin(el)   = sin(inci) / (1+h/R)
    #
    #       el_z = asind( sind(inci_z) / (1+h/R) );
    #
    #       % SCHEME 2
    #       % S
    #       % |\        el_z = <OSB> : elevation angle between the local nadir and the pointing direction to the surface
    #       % | \ /     pi - inci_z = <OBS>: inci_z: incidence angle between the local zenith and the satellite
    #       % |  B      a = <SOB>: angle of the earth between the nadir point of the satellite and the observation
    #       % | /
    #       % |/        (sum angle in triangle = pi) => a = inci_z - el_z
    #       % O
    #       %
    #       %
    #
    #       a = inci_z - el_z;
    #
    #       % Sinus of law for sperical in SCHEME 1:
    #       % sin(a)/sin(alpha) = sin(b)/sin(beta) = sin(c)/sin(gamma)
    #       %
    #       b = asind( sind(a).*sind(beta)./sind(alpha) );
    #
    #       % Cf SCHEME 1, solution of triangles and Napier's analogies
    #       % cotan(gamma/2) = tan((alpha-beta)/2)*sin((a+b)/2)/sin((a-b)/2)
    #       gamma =  2*acotd( tand((alpha-beta)/2).*sind((a+b)/2)./sind((a-b)/2) );
    #       c     =  2*atand( tand((a-b)/2).*sind((alpha+beta)/2)./sind((alpha-beta)/2) );
    #
    #       % Using same as SCHEME 2 but for the point at the squint i.e. with B=A
    #       % el_s: elevation for the squint; inci_s: incidence angle for the squint
    #       %
    #       % el_s = inci_s - b
    #       %
    #       % put this in the sinus of law:
    #       % sin(inci) = (1+h/R)*sin(el)
    #       % => tan(inci) = (1+h/R)*sin(b) / [ (1+h/R)*cos(b) -1]
    #       inci_s = atand( (1+h/R)*sind(b) ./ ( (1+h/R)*cosd(b) -1) );
    #       el_s = asind( sind(inci_s) / (1+h/R) );
    #
    #      elseif isfield(in,'inci_fore')
    #       inci_s = in.inci_fore;
    #       azi_z =  in.azi_mid; %degree
    #       squint_surf = in.azi_fore; %squint_surf; % degree
    #
    #       el_s = asind( sind(inci_s) / (1+h/R) );
    #       b  =  inci_s - el_s;
    #       a  =  asind( sind(b).*sind(alpha)./sind(beta) );
    #       gamma =  2*acotd( tand((alpha-beta)/2).*sind((a+b)/2)./sind((a-b)/2) );
    #       c     =  2*atand( tand((a-b)/2).*sind((alpha+beta)/2)./sind((alpha-beta)/2) );
    #
    #       inci_z   =  atand( (1+h/R)*sind(a) ./ ( (1+h/R)*cosd(a) -1) );
    #       el_z = asind( sind(inci_z) / (1+h/R) );
    #    end
    #
    #
    #
    #    out.altitude   =  in.altitude;
    #    out.inci_mid   =  inci_z;
    #    out.el_mid     =  el_z;
    #    out.azi_mid    =  azi_z;
    #    out.inci_fore  =  inci_s;
    #    out.el_fore    =  el_s;
    #    out.azi_fore   =  squint_surf;
    #    out.nadir_fore = gamma;
    #    out.dist_mid   =  R*sind(a);
    #    out.dist_fore  =  R*sind(b);
    #    out.dist_beam  =  R*sind(c);
    #
    #
    # end



    return sat_geometry








