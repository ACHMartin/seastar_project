import os
import glob
import configparser
import xarray as xr
import platform
from configparser import ConfigParser


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
    """Reads a netCDF file and returns it as an xarray.

    :param netCFD_path: path to the netCDF file
    :type netCFD_path: ``str``

    :raises: ``ValueError`` if file cannot be read as netCDF and \
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

