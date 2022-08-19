
def printXarrayAttributeKeys(data_xr):
    """Dummy function that prints the xrray attribute keys
    to the console.

    :param data_xr: the data to be processed
    :type data_xr: ``xarray``
    """

    attributes = data_xr.attrs
    for key in attributes.keys():
        print(f'\t{key}, {attributes[key]}')
