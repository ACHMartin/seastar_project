# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 07:42:39 2023

@author: davidmccann
"""

import os
import xarray as xr
import numpy as np
import seastar as ss
from seastar.utils.tools import dotdict
from multiprocessing import Pool


level1 = xr.Dataset(
    data_vars=dict(
            CentralWavenumber=([], 270),
            CentralFreq=([], 13.5 * 10**9),
            IncidenceAngleImage=(['across', 'along', 'Antenna'],
                                 np.full([5, 6, 4], 30)),
            AntennaAzimuthImage=(['across', 'along', 'Antenna'],
                                 np.stack((np.full([5, 6], 45),
                                           np.full([5, 6], 90),
                                           np.full([5, 6], 90),
                                           np.full([5, 6], 135)
                                           ), axis=-1
                                          )
                                 ),
            Polarization=(['across', 'along', 'Antenna'],
                          np.stack((np.full([5, 6], 'VV'),
                                    np.full([5, 6], 'VV'),
                                    np.full([5, 6], 'HH'),
                                    np.full([5, 6], 'VV')
                                    ), axis=-1
                                   )
                          ),
#             Sigma0=( ['across','along','Antenna'], np.full([9,11,4], 1.01) ),
#             dsig0=( ['across','along','Antenna'], np.full([9,11,4], 0.05) ),
#             RVL=( ['across','along','Antenna'], np.full([9,11,4], 0.5) ),
#             drvl=( ['across','along','Antenna'], np.full([9,11,4], 0.01) ),
        ),
    coords=dict(
            across=np.arange(0, 5),
            along=np.arange(0, 6),
            Antenna=['Fore', 'MidV', 'MidH', 'Aft'],
        ),
)
level1 = level1.set_coords([
    'CentralWavenumber',
    'CentralFreq',
    'IncidenceAngleImage',
    'AntennaAzimuthImage',
    'Polarization',
])
geo = xr.Dataset(
        data_vars=dict(
            WindSpeed=(['across', 'along'], np.full([5, 6], 10)),
            WindDirection=(['across', 'along'], np.full([5, 6], 150)),
            CurrentVelocity=(['across', 'along'], np.full([5, 6], 1)),
            CurrentDirection=(['across', 'along'], np.full([5, 6], 150)),
        ),
        coords=dict(
            across=np.arange(0, 5),
            along=np.arange(0, 6),
        ),
    )

gmf = dotdict({'nrcs': dotdict({'name': 'nscat4ds'})})
gmf['doppler'] = dotdict({'name': 'mouche12'})

level1['Sigma0'] = ss.gmfs.nrcs.compute_nrcs(level1, geo, gmf.nrcs)*1.001

model_rsv_list = [None] * level1.Antenna.size
model_wasv_list = [None] * level1.Antenna.size
for aa, ant in enumerate(level1.Antenna.data):
#     model_wasv_list[aa] = ss.gmfs.doppler.compute_wasv(level1.sel(Antenna=ant), geo, gmf=gmf.doppler.name)
    model_rsv_list[aa] = ss.gmfs.doppler.compute_total_surface_motion(
        level1.sel(Antenna=ant), geo, gmf=gmf.doppler.name)
# level1['WASV'] = xr.concat(model_wasv_list, dim='Antenna')*1.001
level1['RSV'] = xr.concat(model_rsv_list, dim='Antenna')*1.001

noise = level1.drop_vars([var for var in level1.data_vars])
# noise = level1.drop_vars(['Sigma0','dsig0','RVL','drvl'])
noise['Sigma0'] = level1.Sigma0*0.05
noise['RSV'] = level1.RSV*0.05

sL1 = level1.isel(along=slice(0, 2), across=slice(0, 2))
sN = noise.isel(along=slice(0, 2), across=slice(0, 2))  # Pass wind_current_retrieval
ambiguity = {'name': 'sort_by_cost'}

# Serialize input

list_L1s0 = list(level1.Sigma0.dims)
list_L1s0.remove('Antenna')
level1_stack = level1.stack(z=tuple(list_L1s0))
noise_stack = noise.stack(z=tuple(list_L1s0))
lmoutmap = [None] * level1_stack.z.size
input_mp = [None] * level1_stack.z.size
for ii in range(level1_stack.z.size):
    input_mp[ii] = dict({
        'level1': level1_stack.isel(z=ii),
        'noise': noise_stack.isel(z=ii),
        'gmf': gmf,
    })

# Define worker

def task(input_mp):
    level1 = input_mp['level1']
    noise = input_mp['noise']
    gmf = input_mp['gmf']
    lmout = ss.retrieval.level2.run_find_minima(level1, noise, gmf)
    sol = ss.retrieval.ambiguity_removal.solve_ambiguity(lmout, ambiguity)
    l2 = ss.retrieval.level2.sol2level2(sol)
    return l2


if __name__ == '__main__':
    # create and configure the process pool
    with Pool() as pool:
        # execute tasks in order
        l2 = pool.map(task, input_mp)

l2.to_netcdf(os.getcwd())
