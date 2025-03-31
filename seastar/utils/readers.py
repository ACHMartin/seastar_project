"""Functions for input/output and file reading."""
import os
import glob
import configparser
import xarray as xr
import platform
from configparser import ConfigParser
import hashlib

from _logger import logger

def _read_config(config_file='config.ini'):
    """
    Read configuration file.

    Reads configuration file local machine using the output of platform.node()
    as the device name. Device names and config parameters for each device are
    stored in ../config.ini

    Parameters
    ----------
    config_file : ``str``, optional
        File name of the config file

    Raises
    ------
    Exception
        Device name not in config.ini

    Returns
    -------
    configuration : ``dict``
        Dict of configuration parameters with their variable identifiers as
        keys.

    """
    device_name = platform.node()
    print('Device name = ' + device_name)
    print('Setting local paths...')
    config = ConfigParser()
    config.read(config_file)
    if device_name not in config.sections():
        raise Exception('Device name not in config file. Please check device'
                        'name and file paths are entered into config and try'
                        ' again.')
    configuration = dict(config.items(platform.node()))

    return configuration

def findNetCDFilepaths(directory_path, recursive=False):
    """Returns a list of netCDF files fom a given directory with
    a recursive option.

    :param directory_path: path to the directory to look in
    :type directory_path: ``str``

    :param recursive: whether to search in sub-directories
    :type recursive: ``boolean``, optional

    :return: a list of file paths with '.nc' extension that were found
    :rtype: ``list``
    """

    if not os.path.isdir(directory_path):
        print(f'WARNING: {directory_path} is not a directory')
        return []

    netCDF_filepaths = glob.glob(pathname=directory_path+'/*.nc',
                                 recursive=recursive)

    return netCDF_filepaths


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
    print('Reading Track name config...')
    config_file_name = campaign + '_' + 'TrackNames.ini'
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
    logger.info("Reading Campaign config...")
    config_file_name = 'Campaign_name_lookup.ini'
    campaign_names_config = ConfigParser()
    campaign_names_config.optionxform = str
    campaign_names_config.read(os.path.join(os.path.join('config', config_file_name)))
    campaign_names_dict = dict(campaign_names_config.items('OSCAR_campaigns'))
    return campaign_names_dict
