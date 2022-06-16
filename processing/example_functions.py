import os
import glob


def isThisTreeSpeciesKnown(tree_species_str=''):
    """Tests whether a tree species is known and prints to console
    whether it is native or not and returns boolean.

    :param tree_species_str: a tree species to test, defaults to ''
    :type tree_species_str: String


    :return: True if the tree species parameter appears in list of 
    native or other trees else False
    :rtype: Boolean
    """

    is_known_species = True

    NATIVE_TREE_SPECIES = ['ash', 'elm', 'oak', 'beech', 'rowan']
    OTHER_TREE_SPECIES = ['acer', 'aspen']

    if tree_species_str.lower() in NATIVE_TREE_SPECIES:

        print(f'{tree_species_str} is a native tree species')

    elif tree_species_str.lower() in OTHER_TREE_SPECIES:

        print(f'{tree_species_str} is a known tree species but not native')

    else:

        print(f'{tree_species_str} is an unknown tree species')
        is_known_species = False

    return is_known_species


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
