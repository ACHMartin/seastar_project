"""Functions for input/output and file reading."""
import os
import glob
import configparser
import xarray as xr
import platform
from configparser import ConfigParser, NoSectionError, MissingSectionHeaderError
import hashlib

from _logger import logger



def readNetCDFFile(netCFD_path):
    """
    Read a netCDF file and returns it as an `xarray.Dataset`.

    :param netCFD_path: path to the netCDF file
    :type netCFD_path: ``str``

    :raises: ``ValueError`` if file cannot be read as netCDF and
        returns ``None`` object

    :return: xrray read from the netCDF file
    :rtype: ``xarray``
    """
    data_xr = None
    try:
        data_xr = xr.open_dataset(netCFD_path)

    except ValueError:
        print(f'WARNING "{netCFD_path}" is not a readable netCDF file')

    return data_xr

def md5_checksum_from_file(file_name):
    """
    Create hexidecimal MD5 checksum from input file.

    Parameters
    ----------
    file_name : ``str``
        Full filename including path to create MD5 checksum from.

    Returns
    -------
    md5_checksum : ``str``
        Hexidecimal MD5 checksum based on input file.

    """
    md5_checksum = hashlib.md5(open(os.path.join(file_name),'rb').read()).hexdigest()
    
    return md5_checksum

def short_file_name_from_md5(md5_checksum):
    """
    Create short name from an MD5 checksum.

    Parameters
    ----------
    md5_checksum : ``str``
        Hexidecimal MD5 checksum.

    Returns
    -------
    file_short_name : ``str``
        4-character hexidecimal name based on first 4 characters from MD5 checksum.

    """
    file_short_name = md5_checksum[0:4]
    
    return file_short_name

def read_OSCAR_track_names_config(campaign, flight):
    """
    Read track names configuration file.
    
    Reads in an OSCAR campaign track names INI file and parses it as a dict of
    {track_time : track_name}.

    Parameters
    ----------
    campaign : ``str``
        OSCAR campaign name
    flight : ``str``
        OSCAR flight date, in the form YYYYMMDD

    Returns
    -------
    track_names_dict : ``dict``
        Dict of {Track_time : Track_name}. Track time identical to L1A / L1AP
        track time in file name.

    """
    config_file_name = campaign + '_' + 'TrackNames.ini'
    logger.info(f"Reading Track name config file {config_file_name}...")
    track_names_config = ConfigParser()
    track_names_config.optionxform = str
    track_names_config.read(os.path.join(os.path.join('config', config_file_name)))
    track_names_dict = dict(track_names_config.items(flight))
    return track_names_dict

def read_campaign_config():
    """
    Read campaign names configuration file.
    
    Reads in an OSCAR campaign names INI file and parses it as a dict of
    {date : campaign_name}.

    Returns
    -------
    campaign_names_dict : ``dict``
        Dict of {date : campaign_name}. 
    """
    config_file_name = 'Campaign_name_lookup.ini'
    logger.info(f"Reading Campaign config file {config_file_name}...")
    campaign_names_config = ConfigParser()
    campaign_names_config.optionxform = str
    campaign_names_config.read(os.path.join(os.path.join('config', config_file_name)))
    campaign_names_dict = dict(campaign_names_config.items('OSCAR_campaigns'))
    return campaign_names_dict


def read_config_OSCAR(config_type, info_dict=None):
    """
    Read configuration file related to OSCAR campaigns.
    
    Reads in an OSCAR campaign names INI file or in an OSCAR campaign track names INI file and parses it as a dict

    Parameters
    ----------
    config_type : ``str``
        Config requested. Can be either "campaign" to extract the campaign names or "track" to extract the track names.
    info_dict : ``dict``
        Dict contraining the campaign name and the date of the flight with the following format {campaign: "campaign_name",  flight: "YYYYMMDD"}. 
        Default is None.

    Returns
    -------
    config_mapping : ``dict``
        Dict of {Track_time : Track_name}. Track time identical to L1A / L1AP track time in file name.
        or 
        Dict of {date : campaign_name}. 
    """
    if not isinstance(config_type, str):
        logger.error("Argument 'config_type' must be a string.")
        raise ValueError("Argument 'config_type' must be a string.")

    config_type = config_type.lower()
    config_file_name = ""
    section = ""
    
    if "campaign" in config_type:
        config_file_name = "Campaign_name_lookup.ini"
        section = "OSCAR_campaigns"
        logger.info(f"Reading {section} in config file : {config_file_name}...")
    elif "track" in config_type:
        if info_dict is None:
            logger.error("info_dict is required when looking for track information.")
            raise ValueError("info_dict is required when looking for track information.")
        try:
            section = info_dict['flight']
            campaign = info_dict["campaign"]
        except KeyError as e:
            logger.error(f"Missing key in info_dict: {e}")
            raise KeyError(f"Missing key in info_dict: {e}")
        
        config_file_name = campaign+"_TrackNames.ini"
        logger.info(f"Reading {section} in config file : {config_file_name}...")
        
    else:
        logger.error(f"Invalid 'config_type' value: '{config_type}'. Must contain 'campaign' or 'track'.")
        raise ValueError(f"Invalid 'config_type' value: '{config_type}'. Must contain 'campaign' or 'track'.")
        
    config_path = os.path.join('config', config_file_name)
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = ConfigParser()
    config.optionxform = str 

    try:
        config.read(config_path)
        config_mapping = dict(config.items(section))
    except NoSectionError:
        logger.error(f"Section '{section}' not found in {config_file_name}")
        raise ValueError(f"Section '{section}' not found in {config_file_name}")
    except MissingSectionHeaderError:
        logger.error(f"{config_file_name} does not contain valid INI format (missing section headers)")
        raise ValueError(f"{config_file_name} does not contain valid INI format (missing section headers)")
    
    return config_mapping 
