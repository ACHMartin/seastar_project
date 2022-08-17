# -*- coding: utf-8 -*-
"""
Function to calculate fore and aft surface currents from retrieved currents
Inputs:
    c_u
    c_v
    fore_azi
    aft_azi
Outputs:
    for_surf_current
    aft_surf_current

Created on Thu Aug 11 12:47:07 2022

@author: davidmccann
"""
import numpy as np

def retrieved2fore_aft(c_u,c_v,fore_azi,aft_azi):
    antenna_angle=np.mod(fore_azi - aft_azi, 360) #Angle between the two antennas
    (cvel,cdir)=u_v_to_current_vel_dir(c_u, c_v)
    tmp_dir1 = np.mod(cdir - fore_azi + 180, 360) - 180
    tmp_dir = -tmp_dir1
    ind_pos = tmp_dir > 0
    x=np.zeros(ind_pos.shape)
    y=x
    x[ind_pos,1] = cvel[ind_pos] * cosd(tmp_dir[ind_pos])
    x[~ind_pos,1] = cvel[~ind_pos] * cosd(-tmp_dir[~ind_pos])
    y[ind_pos,1] = np.sqrt((cvel[ind_pos]**2 - x[ind_pos]**2) * sind(antenna_angle[ind_pos])**2) + x[ind_pos] * cosd(antenna_angle[ind_pos])
    y[~ind_pos,1] = -np.sqrt((cvel[~ind_pos]**2 - x[~ind_pos]**2) * sind(antenna_angle[~ind_pos])**2) + x[~ind_pos] * cosd(antenna_angle[~ind_pos])
    for_surf_current = x
    aft_surf_current=y
    return for_surf_current, aft_surf_current



def u_v_to_current_vel_dir(c_u,c_v):
    tmp = c_u + 1j * c_v
    cvel = np.abs(tmp)
    cdir = np.mod(90 - np.angle(tmp) * 180/np.pi,360)
    return cvel, cdir

def cosd(deg):
    rad = np.radians(deg)
    return np.cos(rad)

def sind(deg):
    rad = np.radians(deg)
    return np.sin(rad)