#!/usr/bin/env python
from setuptools import setup, find_packages
from distutils.core import Extension

VERSION = '1.0.5'
LONG_DESCRIPTION = """
A numerical geometric algebra module for python. BSD License.
"""
setup(
    name='clifford',
    version=VERSION,
    license='bsd',
    description='Numerical Geometric Algebra Module',
    long_description=LONG_DESCRIPTION,
    author='Robert Kern',
    author_email='alexarsenovic@gmail.com',
    url='http://clifford.readthedocs.io',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'numba==0.45.1',
        'future',
        'h5py'
    ],
    package_dir={'clifford':'clifford'},

    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',

        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    python_requires='>=3.5',
)
