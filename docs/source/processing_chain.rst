OSCAR data processing chain
---------------------------

OSCAR data processing follows a data-level heirarchy with the following levels:
    - L1A (Level-1A): SAR-focussed Single-Look Complex (SLC) images from one of the three OSCAR look directions (Fore, Mid, Aft)
    - L1AP (Level-1AP): L1A data pre-processed using MATLAB to inject required data fields (e.g., Squint, Incidence Angle)
    - L1B (Level-1B): Combined dataset with all three independent looks (Fore, Aft, Mid) from a single acquisition. Processed for Multilooking, computation of Radial Surface Velocity, Sigma0, etc.
    - L1C (Level-1C): Calibrated L1B dataset. Calibration takes the form of over-land or over-ocean methods to correct for incidence angle dependent biases in Sigma0 and Interferometric phase.
    - L2A (Level-2A): Retrieved Total Surface Current Vector (TSCV) and Ocean Surface Vector Wind (OSVW) from the Wind-Current Retrieval (WCR) method, including full output from all ambiguities.
    - L2B (Level-2B): Retrieved TSCV and OSVW using either the WCR or Sequenctial Current Retrieval (SCR) methods.

Examples of use
===============

For the following examples the OSCAR test dataset is used, found in ``seastar_project\test_data``
To process from L1 to L2 the function ``processing_OSCAR_L1_to_L2`` can be used in the following way:

.. code-block:: python

    import seastar as ss
    import glob
    import os
    import xarray as xr
    
    dict_L2_process = {
    'gmf':{'doppler':{'name':'mouche12'}, # Doppler GMF of Mouce (2012)
          'nrcs':{'name':'nscat4ds'}}, # NRCS GMF of NSCAT-4DS
    'L2_processor':'WCR', # Simultaneous Wind-Current retrieval method
    'RSV_Noise':0.2, # Radial Surface Velocity noise for the WCR method. Here set to 0.2m/s
    'Kp':0.2, # Radiometric resolution for the WCR method. Here set to 20%
    }
    dict_ambiguity = {'name' : 'sort_by_cost'} # Set ambiguity removal to lowest cost
    
    (ds_L2A, ds_L2B) = processing_OSCAR_L1_to_L2(ds_L1, dict_L2_process, dict_ambiguity, dict_env)
    L1_file_list = glob.glob(os.path.join('test_data','*.nc'))
    filename = L1_file_list[0] # Load first file in list of test_data files
    ds_L1C = xr.open_dataset(filename)
    ds_L2B = ss.oscar.processing_chain.processing_OSCAR_L1_to_L2(ds_L1C,
                                                                dict_L2_process=dict_L2_process,
                                                                dict_ambiguity=dict_ambiguity,
                                                                write_nc=False,
                                                                write_L2A_nc=False,
                                                               )

To process L1AP to L1B data the function `processing_OSCAR_L1AP_to_L1B` is used in the following way:

.. code-block:: python

    import seastar as ss
    
    L1B_options = {'vars_to_keep' : ['LatImage',
                                     'LonImage',
                                     'IncidenceAngleImage',
                                     'LookDirection',
                                     'SquintImage',
                                     'CentralFreq',
                                     'OrbitHeadingImage',
                                     'OrbTimeImage'],
               'vars_to_send' : ['Intensity', 'Interferogram', 'Coherence'],
               'window' : 7}
    L1AP_folder = 'path/to/L1AP_data'
    campaign = '202205_IroiseSea' # Campaign name as found in config\OSCAR_config.ini
    flight = '20220517' # Flight date as found in fields of 202205_IroiseSea_TrackNames.ini
    track = 'Track_1' # Track name as found in 202205_IroiseSea_TrackNames.ini
    ds_L1B = ss.oscar.processing_chain.processing_OSCAR_L1AP_to_L1B(
        L1AP_path,
        campaign,
        flight,
        track,
        dict_L1B_process=L1B_options,
        write_nc=False
        )

To perform the calibration step from L1B to L1C data the function `processing_OSCAR_L1B_to_L1C` is used in the following way:

.. code-block:: python

    import seastar as ss
    import os
    
    L1B_path = 'path/to/L1B_data'
    campaign = '202205_IroiseSea' # Campaign name as found in config\OSCAR_config.ini
    flight = '20220517' # Flight date as found in fields of 202205_IroiseSea_TrackNames.ini
    track = 'Track_1' # Track name as found in 202205_IroiseSea_TrackNames.ini
    # calib_dict contains the NRCS (Sigma0) and Interferogram calibration file names. These can be found along with the rest of the OSCAR
    # campaign data in the ESA data store under campaign_name/calib
    calib_dict = {'Sigma0_calib_file':os.path.join(calib_file_path, NRCS_OceanPattern_file_name),
              'Interferogram_calib_file':os.path.join(calib_file_path, InterferogramCalib_file_name)}
    ds_L1C = ss.oscar.processing_chain.processing_OSCAR_L1B_to_L1C(L1B_path,
                                                                   campaign,
                                                                   flight,
                                                                   track,
                                                                   calib_dict,
                                                                   write_nc=False)
    
If `write_nc` is set to `True`, a formatted L1B netCDF file will be written to disk in a new or existing folder tree mirroring the L1AP file structure. It is recommended that the following
file tree structure is followed:
::

    OSCAR
    ├── Campaign Name
        ├── Processing Level
            ├── Data Version         
                ├── Flight
 
All OSCAR data processors when the `write_nc` switch is set to `True` will write their outputs to a mirrored structure, e.g.:
::

    OSCAR
    ├── Campaign Name
        ├── L1AP
        ├   ├── Data Version         
        ├       ├── Flight
        ├── L1B
            ├── Data Version         
                ├── Flight

Processing chain functions
==========================
.. automodule:: seastar.oscar.processing_chain
   :members:

.. |br| raw:: html

   <br />