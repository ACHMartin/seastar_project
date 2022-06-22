import os
import glob
import configparser
import xarray as xr


def _readConfig(config_file_path):
    """Reads the configuration file and returns it.

    :param config_file_path: path to the configuration file
    :type config_file_path: String

    :return: the configuration object read from the lconfiguration file
    :rtype: configparser.SectionProxy

    :meta private:
    """

    config_parser = configparser.RawConfigParser(comment_prefixes='%')
    with open(config_file_path) as f:
        config_file_content = '[configuration]\n' + f.read()
    config_parser.read_string(config_file_content)

    configuration = config_parser['configuration']

    return configuration


def findNetCDFilepaths(directory_path, recursive=False):
    """Returns a list of netCDF files fom a given directory and has
    a recursive option.

    :param directory_path: path to the directory to look in
    :type directory_path: String

    :param recursive: whether to search in sub-directories
    :type directory_path: Boolean, optional

    :return: a list of file paths with '.nc' extension that were found
    :rtype: List
    """

    if not os.path.isdir(directory_path):
        print(f'WARNING: {directory_path} is not a directory')
        return []

    netCDF_filepaths = glob.glob(pathname=directory_path+'/*.nc',
                                 recursive=recursive)

    return netCDF_filepaths


def readNetCDFFile(netCFD_path):
    """Reads a netCDFFile and returns it as an xarray

    :param netCFD_path: path to the netCDF file
    :type netCFD_path: String

    :raises ValueError: if file cannot be read as netCDF

    :return: xrray read from the netCDF file
    :rtype: xarray
    """

    data_xr = None
    try:
        data_xr = xr.open_dataset(netCFD_path)

    except ValueError:
        print(f'WARNING "{netCFD_path}" is not a readable netCDF file')

    return data_xr
