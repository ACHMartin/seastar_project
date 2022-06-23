from setuptools import setup, find_packages

setup(
    name='seastar',
    version='1.0',
    packages=find_packages(include=['utils', 'examples', 'gmfs', 'retrieval'])
)
