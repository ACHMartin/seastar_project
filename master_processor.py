#!/usr/bin/env python
import os
import sys

from seastar.examples import example_functions
from seastar.utils import readers


class SEASTARX(object):
    """SEASTARX class that includes a ``run()`` method to iteratively
    process each netCDF file found in the SAR data directory.

    :param config_file: the configuration file name
    :type config_file: ``str``
    """

    def __init__(self, config_file):
        """Constructor method.

        :param config_file: the configuration file name
        :type config_file: ``str``
        """

        self.CONFIG_FILE = config_file


    def example_method(self):
        """A simple example to show use of the ``self`` keyword.

        :return: True if results directory was created else False
        :rtype: ``boolean``
        """

        results_dir_path = self.configuration['DIRECTORY PATHS']['results']
        self.results_dir = results_dir_path

        if not os.path.isdir(results_dir_path):
            os.mkdir(results_dir_path)
            return True
        else:
            return False


    def run(self):
        """The run method to control all workflow.
        """

        self.configuration = readers._readConfig(self.CONFIG_FILE)
        DIRECTORY_PATHS = self.configuration['DIRECTORY PATHS']
        DATA_DIR = DIRECTORY_PATHS['data']

        CAMPAIGNS = self.configuration['CAMPAIGNS']
        OSCAR_DIR = os.path.join(DATA_DIR, CAMPAIGNS['first'])

        netCDF_filepaths = readers.findNetCDFilepaths(OSCAR_DIR)

        if netCDF_filepaths:
            print(f'the list of netCDF files found in {OSCAR_DIR}:')

            for file_index, filepath in enumerate(netCDF_filepaths):
                print(f'netCDF file {file_index+1}:')

                oscar_xr = readers.readNetCDFFile(netCDF_filepaths[0])

                if oscar_xr:
                    example_functions.printXarrayAttributeKeys(oscar_xr)

                else:
                    print(f'WARNING {filepath} could not be opened as an xarray')

        else:
            print(f'WARNING no netCDF files found in {OSCAR_DIR}')

        # this calls an example method showing how self can use used
        if self.example_method():
            print(f'created directory: {self.results_dir}')


if __name__ == '__main__':

    # turn off console warnings
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter('ignore')

    # make an instance of the class and implement the run method
    obj = SEASTARX('seastarx_config.ini')
    obj.run()
