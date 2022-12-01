# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 11:38:31 2022

@author: davidmccann
"""
import xarray as xr
import numpy as np
from scipy.io import loadmat
from scipy import interpolate


def colocate_xband_marine_radar_data(filename, dsl2):
    """
    Colocate X-band data from matlab to SAR lat/long.

    Parameters
    ----------
    filename : ``str``
        Filename of the X-band matlab .mat data file
    dsl2 : ``xarray.Dataset``
        Dataset containing coordinates and dimensions to colocate to

    Raises
    ------
    Exception
        Exeption raised if ``latitude`` and ``longitude`` variables are not
        present in the radar .mat file

    Returns
    -------
    ds_out : ``xarray.Dataset``
        Dataset containing colocated X-band radar data

    Notes
    -----
    This function written to be as agnostic as possible but is designed
    primarily to colocate X-band radar data as supplied for the SEASTARex
    project.

    """
    ds_out = xr.Dataset()
    data = loadmat(filename)
    data_vars = list(data.keys())
    if 'longitude' in data_vars and 'latitude' in data_vars:
        for var_name in data_vars:
            var_data = data[var_name]
            if var_name in ['__header__', '__version__', '__globals__']:
                ds_out.attrs[var_name] = var_data
            if isinstance(var_data, np.ndarray):
                if var_data.shape == data['longitude'].shape:
                    ds_out[var_name] = xr.DataArray(
                        data=interpolate.griddata(
                            points=(np.ravel(data['longitude']),
                                    np.ravel(data['latitude'])),
                            values=(np.ravel(var_data)),
                            xi=(dsl2.longitude.values,
                                dsl2.latitude.values)
                            ),
                        dims=dsl2.dims,
                        coords=dsl2.coords
                        )
                elif var_data.shape == (1, 1):
                    ds_out[var_name] = float(var_data)
    else:
        raise Exception(
            'longitude and latitude not present in Xband .mat file'
            )
    ds_out.coords['longitude'] = dsl2.longitude
    ds_out.coords['latitude'] = dsl2.latitude
    return ds_out

def compute_Xband_current_magnitude_and_direction(ds):
    current_magnitude = np.sqrt(ds.Ux ** 2 + ds.Uy ** 2)
    direction = np.degrees(np.arctan2(ds.Ux, ds.Uy))
    ind_pos = direction < 0
    current_direction = xr.where(ind_pos,
                                 180 + (180 - np.abs(direction)),
                                 direction
                                 )
    return current_magnitude, current_direction

def colocate_variable_lat_lon(data_in, latitude, longitude, ds_out):
    new_data = interpolate.griddata(
                            points=(np.ravel(longitude),
                                    np.ravel(latitude)),
                            values=(np.ravel(data_in)),
                            xi=(ds_out.longitude.values,
                                ds_out.latitude.values),
                            
                            )  
    colocated_var = xr.DataArray(
                        data=new_data,
                        dims=ds_out.dims,
                        coords=ds_out.coords
                        )
    return colocated_var