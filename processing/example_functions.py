
def doSomethin(data_xr):
    """Performs some processing on the data in the xarray.

    :param data_xr: the data to be processed
    :type data_xr: Xarray
    """

    attributes = data_xr.attrs
    for key in attributes.keys():
        print(f'\t{key}, {attributes[key]}')
