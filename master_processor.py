#!/usr/bin/env python
import os
import sys
import configparser

from processing import daves_functions, example_functions
CONFIG_FILE = 'seastarx_config.txt'

# read the local configration file
config_parser = configparser.RawConfigParser(comment_prefixes='%')
with open(CONFIG_FILE) as f:
    config_file_content = '[configuration]\n' + f.read()
config_parser.read_string(config_file_content)
SEATSARX_CONFIG = config_parser['configuration']

DATA_DIR = SEATSARX_CONFIG['DATA_DIRECTORY']


class SEASTARX(object):


    @staticmethod
    def run():

        print('calling test function #1')
        OSCAR_DIR = os.path.join(DATA_DIR, 'OSCAR')
        netCDF_filepaths = example_functions.findNetCDFilepaths(OSCAR_DIR)

        if netCDF_filepaths:
            print(f'the list of netCDF files found in {OSCAR_DIR}:')
            for filepath in netCDF_filepaths:

                _, filename = os.path.split(filepath)
                print(filename)
        else:
            print(f'no netCDF files found in {OSCAR_DIR}')
        print('call to test function #1 complete')

        print('calling test function #2')
        daves_functions.plotSimpleLine()
        print('')
        print('call to test function #2 complete')


if __name__ == '__main__':

    # turn off console warnings
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter('ignore')

    # Make an instance of the class and implement the run function
    obj = SEASTARX()
    obj.run()
