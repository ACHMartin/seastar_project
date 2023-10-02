SeaSTAR documentation
=====================
=====================

This document is the user manual for the SeaSTAR Project's specialized software,
tailored for the Ocean Surface Current Airborne Radar demonstrator (OSCAR). 
This manual is designed to provide the user with a straightforward guide to utilizing
this software effectively in the processing of OSCAR data as delivered from MetaSensing BV.

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


.. toctree::
    level1
    gmfs
    retrieval
    utils
