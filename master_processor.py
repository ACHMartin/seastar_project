#!/usr/bin/env python
import os
import sys

from utils import readers
from processing import example_functions


class SEASTARX(object):
    """SEASTARX class reads netCDF files containing SAR data and processes the data

    :param config_file: the conficuration file name
    :type config_file: String"""

    def __init__(self, config_file):
        """Constructor method
        """
        self.CONFIG_FILE_PATH = config_file


    def run(self):

        SEASTARX_CONFIG = readers.readConfig(self.CONFIG_FILE_PATH)

        DATA_DIR = SEASTARX_CONFIG['DATA DIRECTORY']

        OSCAR_DIR = os.path.join(DATA_DIR, 'OSCAR')
        netCDF_filepaths = readers.findNetCDFilepaths(OSCAR_DIR)

        if netCDF_filepaths:
            print(f'the list of netCDF files found in {OSCAR_DIR}:')

            for file_index, filepath in enumerate(netCDF_filepaths):
                print(f'netCDF file {file_index+1}:')

                oscar_xr = readers.readNetCDFFile(netCDF_filepaths[0])

                if oscar_xr:
                    example_functions.doSomething(oscar_xr)

                else:
                    print(f'WARNING {filepath} could not be opened as an xarray')

        else:
            print(f'WARNING no netCDF files found in {OSCAR_DIR}')


if __name__ == '__main__':

    # turn off console warnings
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter('ignore')

    # make an instance of the class and implement the run method
    obj = SEASTARX('seastarx_config.txt')
    obj.run()
