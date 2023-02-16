import os
import glob
import configparser
import xarray as xr
import platform
from configparser import ConfigParser


def _set_file_paths():
    """
    Set file paths.

    Sets file paths for local machine using the output of platform.node() as
    the device name. Device names and local paths for each device are stored in
    ../config.ini

    Raises
    ------
    Exception
        Device name not in config.ini

    Returns
    -------
    local_paths : ``dict`` of ``str``
        Dict of file paths with their variable identifiers as keys and local
        paths as values in ``str`` format.

    """
    device_name = platform.node()
    print('Device name = ' + device_name)
    print('Setting local paths...')
    config = ConfigParser()
    config.read('config.ini')
    if device_name not in config.sections():
        raise Exception('Device name not in config.ini. Please check device'
                        'name and file paths are entered into config and try'
                        ' again.')
    local_paths = dict(config.items(platform.node()))

    return local_paths


def _read_DAR_config(date):
    """
    Read DAR track name config.

    Reads the seastarex_DAR_config.ini file containing names for aquisition
    as a dict with keys equalling track names and values equalling the file
    list number for loading.

    Parameters
    ----------
    date : ``str``, ``int``
        Date of aquisition in the form YYYYMMDD

    Raises
    ------
    Exception
        Raises exception if data aquisition date not present in
        seastarex_DAR_config.ini

    Returns
    -------
    DAR_tracks : ``dict``
        Dict of {track names : file number}. File numbers of type ``int``.
        Track names capitalised.

    """
    if type(date) is not str:
        date = str(date)
    config = ConfigParser()
    config_path = os.path.join('seastar', 'oscar', 'seastarex_DAR_config.ini')
    config.read(config_path)
    if date not in config.sections():
        raise Exception('Date not present in seastarex_DAR_config.ini.'
                        'Please check config file')
    DAR_tracks = dict(config.items(date))
    DAR_tracks = {str.capitalize(k): int(v) for k, v in DAR_tracks.items()}
    return DAR_tracks


def _readConfig(config_file_path):
    """Reads the configuration ini file.

    :param config_file_path: path to the configuration file
    :type config_file_path: ``str``

    :return: the configuration object read from the lconfiguration file
    :rtype: ``configparser``

    :meta private:
    """

    configuration = configparser.ConfigParser()
    configuration.read(config_file_path)

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

