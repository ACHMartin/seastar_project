#!/usr/bin/env python

import sys


from processing import example_functions, daves_functions


class SEASTARX(object):

    MY_VALUE = 42

    @staticmethod
    def run():

        print('Hello, HAL. Do you read me, HAL?')
        print('Affirmative, Dave. I read you')

        print(example_functions.isThisTreeSpeciesKnown('Ash'))
        print(example_functions.isThisTreeSpeciesKnown('Hazel'))
        print(example_functions.isThisTreeSpeciesKnown('Aspen'))

        daves_functions.plotSimpleLine()

        print('Dave, this conversation can serve no purpose anymore. Goodbye.')


if __name__ == '__main__':

    # turn off console warnings
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter('ignore')

    # Make an instance of the class and implement the run function
    obj = SEASTARX()
    obj.run()
