## SeaSTAR 

<p align="left">
  <img src="seastar_project/docs/source/_static/images/seastar_logo.png" width="500">
</p>
Welcome to the SeaSTAR Project software repository,
tailored for the Ocean Surface Current Airborne Radar demonstrator (OSCAR). 

SeaSTAR is a new Earth Explorer mission concept dedicated to observing fast-evolving small-scale
ocean surface dynamics in all coastal seas, shelf seas and marginal ice zones. Its science goals are:

1.  To understand the role of fast-evolving small-scale ocean dynamics in mediating exchanges between
land, the cryosphere, the atmosphere, the marine biosphere and the deep ocean.

2.  To determine the ocean circulation and dominant transport pathways in the global coastal,
shelf and marginal ice zones.

3.  To improve understanding of coastal, shelf and marginal ice zones contributions to the global
climate system.

SeaSTAR has been selected as an ESA Earth Explorer 11 candidate to proceed to phase 0
in competition with 3 other candidates.

OSCAR is the airborne demonstrator of SeaSTAR and has been developed by MetaSensing under the
framework of ESA contract 4000116401/16/NL/BJ.

This software has been developed to be applicable to a wide range of SeaSTAR studies including the
processing of OSCAR data from Level-1p (as delivered by MetaSensing before pre-processing) to Level-2
(Wind and Current map per airborne track). The processing, in short, consists of multilooking,
calculation of the Radial Surface Velocity (RSV) from the interferogram, application of calibration
factor, retrieval of geophysical parameters (Total Surface Current Vector TSCV and potentially Ocean Surface
Vector Wind OSVW). The retrieval use either a sequential, where wind and current are calculated separately,
or a simultaneous, where wind and current are retrieved simultaneously, approach.

In this software, no sequential inversion for the wind is provided, this is taken from either an indepent
software (PenWP-OSCAR) or from external data (e.g. Numerical Weather Prediction outputs).

The different steps are as below:

1. Pre-processing using Matlab scripts to compute and add Incidence Angle and
antenna Squint fields to the data files  (Level-1p to Level-1a)

2. Multilooking, computation of the Radial Surface Velocity (L1a to L1b)

3. Residual calibration and coarsening (averaging) to required ground resolution (L1b to L1c)

4. Retrieval of TSCV and OSVW using simultaenous inversion, or computation of TSCV using
sequential inversion and ancilliary OSVW data


## 1. Installation

### 1.1 Download the **seastar_project** repository

Navigate to the latest release `(v1.0)` on the RHS of the root project page and download and unzip the source code.


### 1.2 Create an environment with Anaconda

To run the code in the project you need to install the required Python packages in an environment. To do this we will use **Anaconda**, which can be downloaded [here](https://www.anaconda.com/download/).

Open the Anaconda prompt (in Mac and Linux, open a terminal window) and use the `cd` command (change directory) to the directory where you have installed the **seastar_project** repository.

Create a new environment named `seastar` with all the required packages and activate this environment by entering the following commands:

```
>>> conda env create --file env/environment.yml
>>> conda activate seastar
```

To confirm that you have successfully activated `seastar`, your terminal command line prompt should now start with `(seatar)`.

### 1.3 Optional: Add the path of the seastar_project into your conda pythonpath
To permanently include packages or folder into the `PYTHONPATH` of an Anaconda 
environment, activate the Conda environment and use `conda develop` to add the 
path permanently to the `PYTHONPATH` of the Conda environment.
```
(seastar)>>> conda develop /PATH/TO/seastar_project
```
you should get the following prints on screen:
```
added /PATH/TO/seastar_project
completed operation for: /PATH/TO/seastar_project
```

## 2. Running the code

### 2.1 Set parameters for your local environment

From the directory containing the **seastar_project** edit the file **seatarx_config.ini** and set the parameters as required e.g. set the path to the  local directories for the SAR data and for writing the results.

### 2.2 Execute the processor

In the terminal window opened in the **seastar_project** directory enter the following command:

```
>>> python master_processor.py
```

## 3. Documentation

[readthedocs](https://seastar-project.readthedocs.io/en/latest/)
