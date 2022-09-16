# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 16:48:48 2022

@author: admartin
"""

import numpy as np

def compute_radial_surface_velocity(ds):
    ds['RadialSurfaceVelocity'] = ds.Interferogram / \
                                  ( ds.TimeLag * ds.CentralWavenumber * ds.IncidenceAngle )
    return ds

def compute_radial_surface_current(ds, aux, gmf='mouche12'):
    """
    Radial Surface Current (RSC) = Radial Surface Velocity (RSV) - Wind-wave Artefact Surface Velocity (WASV)
    :param ds:
    :return:
    """
    if gmf == 'mouche12':
        # from JSTARSS publish
        [trash wasv trash] = my_cdop(ds.inci, aux.wind_speed, aux.relative_wind_direction, 'vv');
        level2['RadialSurfaceCurrent'] = ds.RadialSurfaceVelocity - wasv
    return level2

def compute_surface_current_vector(level2):
    """
    Compute Surface Current Vector by combining fore and aft RSC
    :param level2:
    :return:
    """
    """
    """
    # cf Matlab fore_aft2retrieved(comb(ii).fore, comb(ii).aft);

    return level2