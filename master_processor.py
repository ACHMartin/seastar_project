#!/usr/bin/env python
import os
import sys

from utils import readers


CONFIG_FILE_PATH = 'c:\code\seastar\seastarx_config.txt'


class SEASTARX(object):


    @staticmethod
    def run():

        SEASTARX_CONFIG = readers.readConfig(CONFIG_FILE_PATH)

        DATA_DIR = SEASTARX_CONFIG['DATA_DIRECTORY']

        OSCAR_DIR = os.path.join(DATA_DIR, 'OSCAR')
        netCDF_filepaths = readers.findNetCDFilepaths(OSCAR_DIR)

        if netCDF_filepaths:
            print(f'the list of netCDF files found in {OSCAR_DIR}:')
            for filepath in netCDF_filepaths:

                _, filename = os.path.split(filepath)
                print(filename)
        else:
            print(f'no netCDF files found in {OSCAR_DIR}')

        oscar_xr = readers.readNetCDFFile(netCDF_filepaths[0])
        print(oscar_xr)


if __name__ == '__main__':

    # turn off console warnings
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter('ignore')

    # make an instance of the class and implement the run method
    obj = SEASTARX()
    obj.run()
