
def doSomething(data_xr):
    """Performs some examples on the data in the xarray.

    :param data_xr: the data to be processed
    :type data_xr: ``xarray``
    """

    attributes = data_xr.attrs
    for key in attributes.keys():
        print(f'\t{key}, {attributes[key]}')
