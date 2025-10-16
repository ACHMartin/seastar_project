import seastar
from _logger import logger
from _version import __version__
from seastar.retrieval import ambiguity_removal
from seastar.retrieval.level2 import run_find_minima, sol2level2


import xarray as xr


import os
from datetime import datetime as dt, timezone


def processing_OSCAR_L1_to_L2(ds_L1, 
                              dict_L2_process, 
                              dict_ambiguity: Optional[dict]=None, 
                              dict_env: Optional[dict]=None, 
                              write_nc: Optional[bool]=False, 
                              L1_folder: Optional[str]="." 
                              ):
    """
    Processing OSCAR from L1 to L2.
    Processing chain of the OSCAR data from L1 (L1B or L1C) to L2. This processing chain allows to calculate the wind direction and speed as well as the current speed and direction.

    Parameters
    ----------
        ds_L1 : ``xr.Dataset"
            L1B or L1C OSCAR dataset.
        dict_L2_process : ``dict``
            Dictionary containing the information needed for L2 processing:
            "gmf" : gmf dictionary
            "L2_processor" : L2 processor for wind current inversion. Can be "SCR" or "WCR", set by default on SCR,
            "RSV_Noise": RSV_noise,
            "Kp" : Kp (noise of NRCS).
            Defaults to dict().
        dict_ambiguity : ``dict``, (optional)
            Dictionary containing the information needed for ambiguity removal. Example:
            dict_ambiguity = {"name" : "closest_truth",       # Can be "sort_by_cost" or "closest_truth"
                              "method" : "wind",      # Can be "wind", "current", or "windcurrent"
                              "truth" : geo}  
            Defaults to None.
        dict_env : ``dict``, (optional)
            Dictionary containing the environnement information needed for SCR inversion. Shall contain 'u10' and 'wind_direction'.
            Defaults to None.
        write_nc : bool (optional)
            Argument to write the data in a netcdf file. Defaults to False.
        L1_folder : ``str``, (optional)
            Path to save the L2 OSCAR data. Defaults to ".".

    Returns:
    ----------
        ds_L2: ``xr.Dataset``
            Xarray dataset of the L2 OSCAR data.
    """

    # Initialisation
    gmf_dict = dict_L2_process['gmf']
    seastar.oscar.tools.is_valid_gmf_dict(gmf_dict)         # Check the format of gmf_dict

    if dict_ambiguity is None:
        dict_ambiguity = {'name':'sort_by_cost'}

    #-----------------------------------------------------------
    #               L2 PROCESSING
    #-----------------------------------------------------------
    # L2_processor, set by default on SCR (Sequential Current Retrieval) 
    L2_processor = dict_L2_process.get("L2_processor","SCR")

    if L2_processor == "SCR":
        logger.info(f"The processor chosen is: {L2_processor}")

        ds_L2 = seastar.retrieval.level2.sequential_current_retrieval(ds_L1, dict_env, gmf_dict['doppler']['name'])

    elif L2_processor == "WCR":
        logger.info(f"The processor chosen is: {L2_processor}")
        ds_L1 = ds_L1.load()

        logger.info("Compute uncertainty")
        uncertainty = xr.Dataset({"RSV":ds_L1.RadialSurfaceVelocity.copy(deep=True),
                            "Kp":ds_L1.Intensity.copy(deep=True)})

        if "RSV_Noise" not in dict_L2_process:
            logger.error("Missing 'RSV_Noise' in dict_L2_process. The code will crash.")
            raise KeyError("Missing 'RSV_Noise' in dict_L2_process")
        if "Kp" not in dict_L2_process:
            logger.error("Missing 'Kp' in dict_L2_process. The code will crash.")
            raise KeyError("Missing 'Kp' in dict_L2_process")

        uncertainty["RSV"] = dict_L2_process["RSV_Noise"]
        uncertainty["Kp"] = dict_L2_process["Kp"]

        if "Sigma0" not in ds_L1:
            if "Intensity" in ds_L1:
                logger.warning("Variable 'Sigma0' is missing from the dataset, we create it from 'Intensity'.")
                ds_L1["Sigma0"] = ds_L1.Intensity
            else:
                logger.error("Variables 'Sigma0' and 'Intensity are missing from the dataset.")
                raise ValueError("Variables 'Sigma0' and 'Intensity are missing from the dataset.")

        ds_L1["RSV"] = ds_L1.RadialSurfaceVelocity
        uncerty, noise = seastar.performance.scene_generation.uncertainty_fct(ds_L1, uncertainty)

        lmout = run_find_minima(ds_L1, noise, gmf_dict) # noise is a dataset same size as ds_L1
        sol = ambiguity_removal.solve_ambiguity(lmout, dict_ambiguity)
        ds_L2 = sol2level2(sol)

        ds_L2.attrs = ds_L1.attrs.copy()            # Copy the attrs from L1 to L2
        ds_L2.attrs['Kp'] = dict_L2_process["Kp"]
        ds_L2.attrs['RSV_Noise'] = dict_L2_process["RSV_Noise"]
        ds_L2.attrs['Sigma0GMF'] = gmf_dict['nrcs']['name']

        logger.info("Merging L2 data with sol giving L2A data")
        ds_L2A = xr.merge([ds_L2, sol])

        # Defining filename for datafile of L2A products  
        ds_L2A.attrs['ProcessingLevel'] = "L2A"
        filename_L2A = seastar.oscar.tools.formatting_filename(ds_L2A)

        # Write the data in a NetCDF file
        if ds_L1.attrs["ProcessingLevel"] in os.path.dirname(L1_folder):
            path_L2A_data = os.path.join(os.path.dirname(L1_folder).replace(ds_L1.attrs["ProcessingLevel"], "L2A"), os.path.basename(L1_folder))
        else:
            path_L2A_data = os.path.join(L1_folder, "L2A")
        if not os.path.exists(path_L2A_data):
            os.makedirs(path_L2A_data, exist_ok=True)
            logger.info(f"Created directory {path_L2A_data}")
        else: logger.info(f"Directory {path_L2A_data} already exists.")

        logger.info(f"Writing in {os.path.join(path_L2A_data, filename_L2A)}")
        ds_L2A.to_netcdf(os.path.join(path_L2A_data, filename_L2A))

    else:
        logger.error("Unknown level 2 processor, should be in {valid_L2_processor}. The code will crash.")
        raise ValueError("Unknown level 2 processor. The code will crash.")

    #Updating of the CodeVersion and ProcessingLevel in the attrs:
    ds_L2.attrs["CodeVersion"] = __version__
    ds_L2.attrs['ProcessingLevel'] = "L2B"

    # Adding of L2 attrs
    ds_L2.attrs['DopplerGMF'] = gmf_dict['doppler']['name']
    ds_L2.attrs['L2Processor'] = L2_processor

    # Updating of the history in the attrs:
    current_history = ds_L2.attrs.get("History", "")                                               # Get the current history or initialize it
    new_entry = f"{dt.now(timezone.utc).strftime(r'%d-%b-%Y %H:%M:%S')} L2 processing."             # Create a new history entry
    updated_history = f"{current_history}\n{new_entry}" if current_history else new_entry           # Append to the history
    ds_L2.attrs["History"] = updated_history                                                       # Update the dataset attributes

    # Defining filename for datafile
    filename = seastar.oscar.tools.formatting_filename(ds_L2)

    # Write the data in a NetCDF file
    if write_nc:
        if ds_L1.attrs["ProcessingLevel"] in os.path.dirname(L1_folder):
            path_L2B_data = os.path.join(os.path.dirname(L1_folder).replace(ds_L1.attrs["ProcessingLevel"], "L2B"), os.path.basename(L1_folder))
        else:
            path_L2B_data = os.path.join(L1_folder, "L2B")
        if not os.path.exists(path_L2B_data):
            os.makedirs(path_L2B_data, exist_ok=True)
            logger.info(f"Created directory {path_L2B_data}")
        else: logger.info(f"Directory {path_L2B_data} already exists.")

        logger.info(f"Writing in {os.path.join(path_L2B_data, filename)}")
        ds_L2.to_netcdf(os.path.join(path_L2B_data, filename))

    return ds_L2