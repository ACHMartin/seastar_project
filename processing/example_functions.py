import os
import glob

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
