# -*- coding: utf-8 -*-
"""
Calculates DOP and surface velocities using inputs:

   % - inc: incidence angle in degree
   % - u10: wind speed
   % - wdir: wind direction (degree)
   % - pol: hh or vv


Created on Thu Aug 11 09:55:36 2022

@author: davidmccann
"""
import numpy as np

def my_cdop(inc,u10,wdir,pol):
    
    f_c = 5.5 #GHz
    c = 299792458 #speed of light in vacuum
    n = 1.000293
    c_air = c/n # speed of light in air
    lambda_c  = c_air / f_c / 1e9
    
    dop = cdop(inc, u10, wdir, pol)
    los_vel = -dop * lambda_c / 2
    surf_vel = los_vel / sind(inc)
    
    return dop, surf_vel, los_vel


def sind(deg):
    rad = np.radians(deg)
    return np.sin(rad)