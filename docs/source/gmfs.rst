Geophysical Model Functions (GMFs)
==================================

This section is about the Geophysical Model Functions (GMFs) used for the data processing. 

Doppler
-------

The Doppler section concerns all the elements directly related to the surface motion as sensed by radar Doppler measurements. It concerns both the total surface current component and the Wave Doppler (WD) component.
In this package the Wave Doppler is refered as Wind-wave Artefact Surface Velocity (WASV).
Two GMFs to compute the WASV are available in this package: mouche12 and yurovsky19.

All these methods are described below:


.. automodule:: seastar.gmfs.doppler
   :members:
   
NRCS
-------
The NRCS section concerns all the elements directly related to the Normalized Radar Cross Section (NRCS).

A Ku-band GMF to compute NRCS is available in this package: nscat4ds.

The methods are described below:
.. automodule:: seastar.gmfs.nrcs
   :members: