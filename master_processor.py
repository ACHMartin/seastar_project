#!/usr/bin/env python
import os
import sys

from utils import readers


CONFIG_FILE_PATH = 'seastarx_config.txt'


class SEASTARX(object):


    @staticmethod
    def run():

        SEASTARX_CONFIG = readers.readConfig(CONFIG_FILE_PATH)

        DATA_DIR = SEASTARX_CONFIG['DATA DIRECTORY']

        OSCAR_DIR = os.path.join(DATA_DIR, 'OSCAR')
        netCDF_filepaths = readers.findNetCDFilepaths(OSCAR_DIR)

        if netCDF_filepaths:
            print(f'the list of netCDF files found in {OSCAR_DIR}:')

            for file_index, filepath in enumerate(netCDF_filepaths):
                print(f'netCDF file {file_index+1}:')

                oscar_xr = readers.readNetCDFFile(netCDF_filepaths[0])

                attributes = oscar_xr.attrs
                for key in attributes.keys():
                    print(f'\t{key}, {attributes[key]}')

        else:
            print(f'no netCDF files found in {OSCAR_DIR}')


if __name__ == '__main__':

    # turn off console warnings
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter('ignore')

    # make an instance of the class and implement the run method
    obj = SEASTARX()
    obj.run()
