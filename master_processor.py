#!/usr/bin/env python

import sys


from processing import clives_functions, adriens_fuctions


class SEASTARX(object):

    MY_VALUE = 42

    @staticmethod
    def run():

        clives_functions.doSomething()
        adriens_fuctions.speakEnglish()

        print(f'triple MY_VALUE: {clives_functions.doSomethingElse(SEASTARX.MY_VALUE)}')
        adriens_fuctions.speakFrench()


if __name__ == '__main__':

    # turn off console warnings
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter('ignore')

    # Make an instance of the class and implement the run function
    obj = SEASTARX()
    obj.run()
