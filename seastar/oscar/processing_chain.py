import numpy as np
import xarray as xr

import glob, re, os
from os.path import join
from datetime import datetime as dt, timezone

from typing import Optional

import seastar
from seastar.oscar.level1 import apply_calibration, compute_radial_surface_velocity
from seastar.retrieval import ambiguity_removal
from seastar.retrieval.level2 import run_find_minima, sol2level2, is_valid_gmf_dict

from _logger import logger
from _version import __version__

#TODO put all the os.path.join/basename as directly "join" (after rebase clean_tree)

def processing_OSCAR_L1_to_L2(ds_L1, 
                              dict_L2_process, 
                              dict_ambiguity: Optional[dict]=None, 
                              dict_env: Optional[dict]=None, 
                              write_nc: Optional[bool]=False, 
                              L1_folder: Optional[str]="." 
                              ) ->  xr.DataArray:
    """
    Processing OSCAR from L1 (NRCS, Interferogram) to L2 (Current vectors + Wind vectors for WCR).

    Processing chain of the OSCAR data from L1 (L1B or L1C) to L2. This processing chain allows 
    to calculate the wind direction and speed as well as the current speed and direction.
    It provides two outputs: 
    - L2A including full details about the inversions for the WCR inversion; 
    - L2B including only the retrieved Current (+ Wind vectors for WCR inversion)

    WCR: Wind & Current retrieval
    SCR: Sequential Current retrieval (using wind vector as input)

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
        dict_ambiguity : ``dict``, (optional)             Defaults to None.
            Dictionary containing the information needed for ambiguity removal. Example:
            dict_ambiguity = {"name" : "closest_truth",       # Can be "sort_by_cost" or "closest_truth"
                              "method" : "wind",      # Can be "wind", "current", or "windcurrent"
                              "truth" : geo}  
        dict_env : ``dict``, (optional)             Defaults to None.
            Dictionary containing the environnement information needed for SCR inversion. Shall contain 'u10' and 'wind_direction'.
        write_nc : bool (optional) Defaults to False.
            Argument to write the data in a netcdf file. 
        L1_folder : ``str``, (optional) Defaults to ".".
            Path to save the L2 OSCAR data. 

    Returns:
    ----------
        ds_L2: ``xr.Dataset``
            Xarray dataset of the L2 OSCAR data.
    """

    # Initialisation
    gmf_dict = dict_L2_process['gmf']
    is_valid_gmf_dict(gmf_dict)         # Check the format of gmf_dict

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
        sol = ambiguity_removal.solve_ambiguity(lmout, dict_ambiguity) # include full details of the retrieval
        ds_L2 = sol2level2(sol)

        ds_L2.attrs = ds_L1.attrs.copy()            # Copy the attrs from L1 to L2
        ds_L2.attrs['Kp'] = dict_L2_process["Kp"]
        ds_L2.attrs['RSV_Noise'] = dict_L2_process["RSV_Noise"]
        ds_L2.attrs['Sigma0GMF'] = gmf_dict['nrcs']['name']

        #----------------------
        #     L2A products 
        #----------------------
        logger.info("Merging L2 data with sol giving L2A data")
        ds_L2A = xr.merge([ds_L2, sol]) #merge with the full details of the retrieval from "sol"

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


def processing_OSCAR_L1AP_to_L1B(L1AP_folder, campaign, acq_date, track, dict_L1B_process=dict(), write_nc=False):
    """
    Processing chain from L1AP tp L1B.

    This function processes the OSCAR data from L1AP to L1B. It needs a triplet of L1AP file to generate a L1B file with data of the three antennas.

    Parameters
    ----------
        L1AP_folder : ``str``
            Folder of the L1AP files.
        campaign : ``str``
            Campaign name as defined in config/Campaign_name_lookup.ini file. Example of campaign name: 202205_IroiseSea, 202305_MedSea.
        acq_date : ``str``
            Date of the data acquisition with the format "YYYYMMDD".
        track : ``str``
            Track name as defined in config/XXX_TrackNames.ini file. Format should be "Track_x" for tracks over ocean and 
            "Track_Lx" for tracks over land with "x" the number of the track.
        dict_L1B_process : ``dict``
            Dictionnary containing information about the window for the rolling mean for the multilooking computation,
            the vars to keep (vars_to_keep) from L1AP to L1B file and the vars to provide (vars_to_send) after the multilooking computation.
            window default value is 3
            vars_to_keep default list is: ['LatImage', 'LonImage', 'IncidenceAngleImage',
                                           'LookDirection', 'SquintImage', 'CentralFreq', 'OrbitHeadingImage']
            vars_to_send default list is: ['Intensity', 'Interferogram', 'Coherence']
        write_nc (bool, optional): 
            Argument to write the data in a netcdf file. Defaults to False.

    Returns:
    ----------
        ds_L1B: ``xr.Dataset``
            Xarray dataset of the L1B OSCAR data.
    """

    # checking acq_date format:
    if seastar.oscar.tools.is_valid_acq_date(acq_date):
        logger.info("'acq_date' format is okay")
    else:
        logger.error("'acq_date' should be a valid date string in 'YYYYMMDD' format ")
        raise ValueError("'acq_date' should be a valid date string in 'YYYYMMDD' format ")

    # Checking campaign name:
    valid_campaign = ["202205_IroiseSea", "202305_MedSea"]
    if campaign in valid_campaign:
        logger.info("Camapaign name has valid value.")
    else:
        logger.error(f"Unexpected campaign value: {campaign}")
        raise ValueError(f"Unexpected campaign value: {campaign}. Shall be {valid_campaign}.")


    # Getting the date for every tracks in the dict.
    track_names_dict = seastar.utils.readers.read_config_OSCAR('track', {"campaign" : campaign, "flight" : acq_date})  #read_OSCAR_track_names_config(campaign, acq_date)

    # Getting the date for the track we are interested in
    if track in track_names_dict.values():
        date_of_track = [key for key, v in track_names_dict.items() if v == track][0]
        logger.info(f"Track '{track}' found, acquisition time: {date_of_track}")
    else:
        logger.warning(f"Track '{track}' not found in track_names_dict. The code will crash.")

    # Loading the file names of the files corresponding to date_of_track (triplet of file - one per antenna)
    L1AP_file_names = [os.path.basename(file) for file in sorted(glob.glob(join(L1AP_folder, f"*{date_of_track}*.nc")))]
#-----------------------------------------------------------
#               L1B PROCESSING
#-----------------------------------------------------------

    vars_to_keep = dict_L1B_process.get('vars_to_keep',['LatImage','LonImage','IncidenceAngleImage',
                                                    'LookDirection','SquintImage','CentralFreq','OrbitHeadingImage'])
    logger.info(f"Opening track: {track} on day: {acq_date}")
    ds_dict = seastar.oscar.tools.load_L1AP_OSCAR_data(L1AP_folder, L1AP_file_names)
    # Getting the antenna identificators
    antenna_list = list(ds_dict.keys())
    logger.info(f"Antenna: {antenna_list}")
    ds_ml = dict()
    for i, antenna in enumerate(antenna_list):
        logger.info(f"Begining of the processing of {L1AP_file_names[i]}")
        ds_dict[antenna] = seastar.oscar.level1.replace_dummy_values(
            ds_dict[antenna], dummy_val=float(ds_dict[antenna].Dummy.data))
        ds_ml[antenna] = seastar.oscar.level1.compute_multilooking_Master_Slave(ds_dict[antenna], dict_L1B_process['window'])
        ds_ml[antenna]['Polarization'] = seastar.oscar.level1.check_antenna_polarization(ds_dict[antenna])
        ds_ml[antenna]['AntennaAzimuthImage'] = seastar.oscar.level1.compute_antenna_azimuth_direction(ds_dict[antenna], antenna=antenna)
        ds_ml[antenna]['TimeLag'] = seastar.oscar.level1.compute_time_lag_Master_Slave(ds_dict[antenna], options='from_SAR_time')         # Time difference between Master and Slave
        #Rolling median to smooth out TimeLag errors
        if not np.isnan(ds_ml[antenna].TimeLag).all():
            ds_ml[antenna]['TimeLag'] = ds_ml[antenna].TimeLag\
                .rolling({'CrossRange': 5}).median()\
                    .rolling({'GroundRange': 5}).median()

        ds_ml[antenna][vars_to_keep] = ds_dict[antenna][vars_to_keep]
        ds_ml[antenna]['TrackTime'] = seastar.oscar.level1.track_title_to_datetime(ds_ml[antenna].StartTime)
        ds_ml[antenna]['Intensity_dB'] = seastar.utils.tools.lin2db(ds_ml[antenna].Intensity)
        #Applying phase sign convention correction from config\OSCAR_config.ini
        ds_ml[antenna]['Interferogram'] = seastar.oscar.level1.apply_phase_sign_convention(ds_ml[antenna])
        ds_ml[antenna]['RadialSurfaceVelocity'] = seastar.oscar.level1.compute_radial_surface_velocity(ds_ml[antenna])
    ds_ml = seastar.oscar.level1.fill_missing_variables(ds_ml, antenna_list)

    # Building L1 dataset
    logger.info(f"Build L1 dataset for :  {track} of day: {acq_date}")
    ds_L1B = seastar.oscar.level1.merge_beams(ds_ml, antenna_list)
    # Clean units attribute '[m]' -> 'm' in the L1B dataset
    seastar.oscar.tools.clean_units_attribute(ds_L1B)
    # Change long_name + add description for GroundRange and CrossRange
    ds_L1B['GroundRange'].attrs['description'] = 'Dimension of the image in ground range (ie across track) direction'
    ds_L1B['GroundRange'].attrs['long_name'] = 'Across track direction'
    ds_L1B['CrossRange'].attrs['description'] = 'Dimension of the image along cross-range (ie along track) direction'
    ds_L1B['CrossRange'].attrs['long_name'] = 'Along track direction'

    del ds_ml
    ds_L1B = ds_L1B.drop_vars(['LatImage', 'LonImage'], errors='ignore')
    #Updating of the CodeVersion in the attrs:
    ds_L1B.attrs["CodeVersion"] = __version__
    # Updating of the history in the attrs:
    current_history = ds_L1B.attrs.get("History", "")                                          # Get the current history or initialize it
    str_time = dt.now(timezone.utc).strftime('%d-%b-%Y %H:%M:%S')
    new_entry = f"{str_time} L1B processing."                                                  # Create a new history entry
    updated_history = f"{current_history}\n{new_entry}" if current_history else new_entry           # Append to the history
    ds_L1B.attrs["History"] = updated_history                                                       # Update the dataset attributes
    # Defining filename for datafile
    filename = seastar.oscar.tools.formatting_filename(ds_L1B)
    # Write the data in a NetCDF file
    if write_nc:
        path_new_data = os.path.join(os.path.dirname(L1AP_folder).replace("L1AP", "L1B"), os.path.basename(L1AP_folder))
        if not os.path.exists(path_new_data):
            os.makedirs(path_new_data, exist_ok=True)
            logger.info(f"Created directory {path_new_data}")
        else: logger.info(f"Directory {path_new_data} already exists.")

        logger.info(f"Writing in {os.path.join(path_new_data, filename)}")
        ds_L1B.to_netcdf(os.path.join(path_new_data, filename))

    return ds_L1B


def processing_OSCAR_L1B_to_L1C(L1B_folder, campaign, acq_date, track, calib_dict, write_nc=False):
    """
    L1B to L1C processing chain.

    Processes OSCAR L1B data to L1C by applying one or more calibration files.

    Parameters
    ----------
    L1B_folder : ``str``
        Path to folder on disk containing OSCAR L1B files to process
    campaign : ``str``
        OSCAR campaign name
    acq_date : ``str``
        Acquisition date of the data in the form YYYYMMDD
    track : ``str``
        Track name of data to process in the form of e.g., `Track_1`
    calib_dict : ``dict``
        Dict containing the following (name:content):
            {'Sigma0_calib_file': full filename including path for Sigma0 calib file,
             'Interferogram_calib_file': full filename including path for Interferogram calib file,
             }
    write_nc : ``bool``, optional
        Option to write L1C file to disk. The default is False.

    Raises
    ------
    Exception
        Raises an exception if not all required entries found in calib_dict

    Returns
    -------
    ds_L1C : ``xr.Dataset``
        Calibrated OSCAR L1C dataset

    """
    #Checking calib_dict
    valid_dict_keys = ['Sigma0_calib_file', 'Interferogram_calib_file']
    if not all([i in calib_dict.keys() for i in valid_dict_keys]):
        pattern = re.compile(r'^(' + '|'.join(map(re.escape, calib_dict.keys())) + r')$')
        missing_keys = [entry for entry in valid_dict_keys if not pattern.match(entry)]
        raise Exception(str(missing_keys) + ' missing from calib_dict')


    # checking acq_date format:
    if seastar.oscar.tools.is_valid_acq_date(acq_date):
        logger.info("'acq_date' format is okay")
    else:
        logger.error("'acq_date' should be a valid date string in 'YYYYMMDD' format ")
        raise ValueError("'acq_date' should be a valid date string in 'YYYYMMDD' format ")

    # Checking campaign name:
    valid_campaign = ["202205_IroiseSea", "202305_MedSea"]
    if campaign in valid_campaign:
        logger.info("Campaign name has valid value.")
    else:
        logger.error(f"Unexpected campaign value: {campaign}")
        raise ValueError(f"Unexpected campaign value: {campaign}. Shall be {valid_campaign}.")


    # Getting the date for every tracks in the dict.
    track_names_dict = seastar.utils.readers.read_config_OSCAR('track', {"campaign" : campaign, "flight" : acq_date})  #read_OSCAR_track_names_config(campaign, acq_date)

    # Getting the date for the track we are interested in
    if track in track_names_dict.values():
        date_of_track = [key for key, v in track_names_dict.items() if v == track][0]
        logger.info(f"Track '{track}' found, acquisition time: {date_of_track}")
    else:
        logger.warning(f"Track '{track}' not found in track_names_dict. The code will crash.")

    #-----------------------------------------------------------
    #               L1C PROCESSING
    #-----------------------------------------------------------

    # Loading calib files
    Interferogram_calib_file = calib_dict.get('Interferogram_calib_file')
    Sigma0_calib_file = calib_dict.get('Sigma0_calib_file')

    logger.info(f"Loading Interferogram calibration file: {Interferogram_calib_file}")
    Interferogram_calib_file_path, Interferogram_calib_file_name = os.path.split(Interferogram_calib_file)
    ds_Interferogram_calib = xr.open_dataset(Interferogram_calib_file)
    logger.info(f"Loading Sigma0 calibration file: {Sigma0_calib_file}")
    Sigma0_calib_file_path, Sigma0_calib_file_name = os.path.split(Sigma0_calib_file)
    ds_Sigma0_calib = xr.open_dataset(Sigma0_calib_file)
    # Loading data
    logger.info(f"Opening track: {track} on day: {acq_date}")
    L1B_file_name = seastar.oscar.tools.find_file_by_track_name(os.listdir(L1B_folder), track=track)
    ds_L1B = xr.open_dataset(os.path.join(L1B_folder, L1B_file_name))
    ds_L1C = ds_L1B.copy(deep=True)
    ds_L1C.attrs['ProcessingLevel'] = 'L1C'

    # Radiometric Calibration
    logger.info(f"Calibrating Sigma0 for :  {track} of day: {acq_date}")
    ds_L1C['Sigma0'], ds_L1C['Sigma0CalImage'] = apply_calibration(ds_L1B, ds_Sigma0_calib, 'Sigma0')
    ds_L1C['Sigma0'].attrs['calibration_file_name'] = Sigma0_calib_file_name
    ds_L1C['Sigma0'].attrs['calibration_file_short_name'] = seastar.utils.readers.short_file_name_from_md5(
        seastar.utils.readers.md5_checksum_from_file(Sigma0_calib_file)
    )
    ds_L1C['Sigma0CalImage'].attrs['calibration_file_name'] = Sigma0_calib_file_name
    ds_L1C['Sigma0CalImage'].attrs['calibration_file_short_name'] = seastar.utils.readers.short_file_name_from_md5(
        seastar.utils.readers.md5_checksum_from_file(Sigma0_calib_file)
    )

    # Interferometric Calibration
    logger.info(f"Calibrating Interferogram for :  {track} of day: {acq_date}")
    ds_L1C['Interferogram'], ds_L1C['InterferogramCalImage'] = apply_calibration(ds_L1B, ds_Interferogram_calib, 'Interferogram')
    ds_L1C['Interferogram'].attrs['calibration_file_name'] = Interferogram_calib_file_name
    ds_L1C['Interferogram'].attrs['calibration_file_short_name'] = seastar.utils.readers.short_file_name_from_md5(
        seastar.utils.readers.md5_checksum_from_file(Interferogram_calib_file)
    )
    ds_L1C['InterferogramCalImage'].attrs['calibration_file_name'] = Interferogram_calib_file_name
    ds_L1C['InterferogramCalImage'].attrs['calibration_file_short_name'] = seastar.utils.readers.short_file_name_from_md5(
        seastar.utils.readers.md5_checksum_from_file(Interferogram_calib_file)
    )

    # Updating RSV with new Interferogram

    logger.info(f"Re-computing RadialSurfaceVelocity using calibrated Interferograms for :  {track} of day: {acq_date}")
    rsv_list = [compute_radial_surface_velocity(ds_L1C.sel(Antenna=ant)) for ant in ds_L1C.Antenna]
    ds_L1C['RadialSurfaceVelocity'] = xr.concat(rsv_list, dim='Antenna', join='outer')
    ds_L1C['RadialSurfaceVelocity'].attrs['description'] = 'Radial Surface Velocity computed with calibrated Interferograms'
    #Updating of the CodeVersion in the attrs:
    ds_L1C.attrs["CodeVersion"] = __version__

    # Updating of the history in the attrs:
    current_history = ds_L1C.attrs.get("History", "")                                               # Get the current history or initialize it
    new_entry = f"{dt.now(timezone.utc).strftime(r'%d-%b-%Y %H:%M:%S')} L1C processing."             # Create a new history entry
    updated_history = f"{current_history}\n{new_entry}" if current_history else new_entry           # Append to the history
    ds_L1C.attrs["History"] = updated_history                                                       # Update the dataset attributes
    ds_L1C.attrs['OceanPatternCalibrationFileName'] = Sigma0_calib_file_name
    ds_L1C.attrs['OceanPatternCalibrationFileShortName'] = seastar.utils.readers.short_file_name_from_md5(
        seastar.utils.readers.md5_checksum_from_file(Sigma0_calib_file)
    )
    ds_L1C.attrs['LandCalibFileName'] = Interferogram_calib_file_name
    ds_L1C.attrs['LandCalibFileShortName'] = seastar.utils.readers.short_file_name_from_md5(
        seastar.utils.readers.md5_checksum_from_file(Interferogram_calib_file)
    )
    ds_L1C.attrs['NRCSGMF'] = ds_Sigma0_calib.attrs['NRCSGMF']
    ds_L1C.attrs['Calibration'] = ' '.join(['NRCS calibrated using OceanPattern calibration. OceanPattern process uses data from a star pattern of multiple acquisitions taken at different headings',
    'relative to the wind direction. Median along-track data are then grouped by incidence angle and antenna look direction to produce data variables relative to azimuth for a discrete',
    'set of incidence angles. Curves are fitted to these points. Similar curves are generated using the NRCS GMF (',
                                           ds_Sigma0_calib.attrs['NRCSGMF'],
                                           ') and the difference',
    'between the fitted curves and the GMF averaged over azimuth are taken as the NRCS calibration bias for a given incidence angle. Interferograms calibrated with LandCalib. LandCalib',
    'process uses an OSCAR acquisition over land and finds land pixes using the GSHHS global coastline dataset to generate a land mask. Applying this land mask to OSCAR L1B Interferogram imagery',
    'the along-track median is taken to generate Interferogram bias wrt the cross range (incidence angle) dimension. These data are then smoothed with a polynomial fit to generate Interferogram',
    'bias curves for the Fore and Aft antenna directions (Mid is set to zero)'])

    # Drop uncalibrated high level variables
    vars_to_drop = ['Intensity','Intensity_dB']
    ds_L1C = ds_L1C.drop_vars(vars_to_drop)

    # Filename formatting
    filename = seastar.oscar.tools.formatting_filename(ds_L1C)

    # Write the data in a NetCDF file
    if write_nc:
        path_new_data = os.path.join(os.path.dirname(L1B_folder).replace("L1B", "L1C"), os.path.basename(L1B_folder))
        if not os.path.exists(path_new_data):
            os.makedirs(path_new_data, exist_ok=True)
            logger.info(f"Created directory {path_new_data}")
        else: logger.info(f"Directory {path_new_data} already exists.")

        logger.info(f"Writing in {os.path.join(path_new_data, filename)}")
        ds_L1C.to_netcdf(os.path.join(path_new_data, filename))

    return ds_L1C