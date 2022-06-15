def isThisTreeSpeciesKnown(tree_species_str=''):
    """This function tests whether a tree species is known and
    prints to console whether it is native or not

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

    elif tree_species_str.lower in OTHER_TREE_SPECIES:

        print(f'{tree_species_str} is a known tree species but not native')

    else:

        print(f'{tree_species_str} is unknown to this code')
        is_known_species = False

    return is_known_species
