from setuptools import setup, find_packages
import numpy
import xarray

        
setup(
    name='seastar',
    version='1.0',
    packages=find_packages(include=['utils', 'examples', 'gmfs', 'retrieval']),
    setup_requires=['numpy'],['xrray']
)
