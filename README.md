<p align="center">
  <img src="/docs/source/_static/images/seastar.png" width="500">
</p>
Welcome to the **SeaSTAR** Project software repository,
tailored for the Ocean Surface Current Airborne Radar demonstrator (OSCAR). 

**SeaSTAR** is a new Earth Explorer mission concept dedicated to observing fast-evolving small-scale
ocean surface dynamics in all coastal seas, shelf seas and marginal ice zones. Its science goals are:

1. To understand the role of fast-evolving small-scale ocean dynamics in mediating exchanges between
   land, the cryosphere, the atmosphere, the marine biosphere and the deep ocean.

2. To determine the ocean circulation and dominant transport pathways in the global coastal,
   shelf and marginal ice zones.

3. To improve understanding of coastal, shelf and marginal ice zones contributions to the global
   climate system.

**SeaSTAR** has been selected as an ESA Earth Explorer 11 candidate to proceed to phase 0
in competition with 3 other candidates.

OSCAR is the airborne demonstrator of SeaSTAR and has been developed by MetaSensing under the
framework of ESA contract *4000116401/16/NL/BJ*.

This software has been developed to be applicable to a wide range of **SeaSTAR** studies including the
processing of OSCAR data from Level-1p (as delivered by MetaSensing before pre-processing) to Level-2
(Wind and Current map per airborne track). The processing, in short, consists of multilooking,
calculation of the Radial Surface Velocity (RSV) from the interferogram, application of calibration
factor, retrieval of geophysical parameters (Total Surface Current Vector TSCV and potentially Ocean Surface
Vector Wind OSVW). The retrieval use either a sequential, where wind and current are calculated separately,
or a simultaneous, where wind and current are retrieved simultaneously, approach.

In this software, no sequential inversion for the wind is provided, this is taken from either an indepent
software (PenWP-OSCAR) or from external data (e.g. Numerical Weather Prediction outputs).

The different steps are as below:

1. *Pre-processing* using Matlab scripts to compute and add Incidence Angle and
   antenna Squint fields to the data files  (Level-1p to Level-1a):
```
      matlab/metasensing/add_inc_and_squint_to_netcdf_batch.m
```

2. *Multilooking*, computation of the *Radial Surface Velocity* (L1a to L1b) using functions
   in the *oscar.level1* module:
```
      seastar.oscar.level1.compute_SLC_Master_Slave()
   
      seastar.oscar.level1.compute_multilooking_Master_Slave()
   
      seastar.oscar.level1.compute_time_lag_Master_Slave()
   
      seastar.oscar.level1.compute_radial_surface_velocity()
```
3. *Residual calibration* and coarsening (averaging) to required ground resolution (L1b to L1c).

4. *Retrieval of TSCV and OSVW* using *simultaenous inversion*, or computation of TSCV using
   *sequential inversion* and ancilliary OSVW data, using functions in the *retrieval.level2*
   module (simultaneous) or the *oscar.level1* module (sequential):
```  
       seastar.retrieval.level2.wind_current_retrieval()
       
       seastar.oscar.level1.compute_radial_surface_current()
```

## 1. Installation

### 1.1 Download the **seastar_project** repository

Navigate to the latest release `(v2023.10.3)` on the RHS of the root project page and download and unzip the source code.

### 1.2 Create an environment

To run the code in the project you need to install the required Python packages in an environment. To create and activate the new environment with all the required packages using `Mamba`, you can run:
```
>>> mamba env create -f seastar_project/env/environment.yml
>>> mamba activate seastar
```
Alternatively, using `Conda` (slightly slower than `Mamba` but widely used), you can run:
```
>>> conda env create -f seastar_project/env/environment.yml
>>> conda activate seastar
```

It is also possible to install the environment using `Poetry`. This methods, which is faster than with `Mamba` and `Conda`, is explained in more details in the [documentation](https://seastar-project.readthedocs.io/en/latest/)

## 2. Documentation

A Sphinx documentation is available following this [link] (https://seastar-project.readthedocs.io/en/latest/). It provides more details on the installation and on the organisation of the project code.

## 3. License

Copyright 2023 Adrien Martin & David McCann & Eva Le Merle

Licensed under the Apache License, Version 2.0 (the "License"); you may not 
use this file except in compliance with the License. You may obtain a copy of 
the License at: http://www.apache.org/licenses/LICENSE-2.0. An additional copy
can be found in this repository (License.txt).

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.” 

