#!/usr/bin/env python

#import ez_setup
#ez_setup.use_setuptools()
from setuptools import setup, find_packages
from distutils.core import Extension

VERSION = '1.0.1'
LONG_DESCRIPTION = """
A numerical geometric algebra module for python. BSD License.
"""
setup(name='clifford',
	version=VERSION,
	license='bsd',
	description='Numerical Geometric Algebra Module',
	long_description=LONG_DESCRIPTION,
	author='Robert Kern',
	author_email='alexarsenovic@gmail.com',
	url='http://clifford.readthedocs.io',
	packages=find_packages(),
	install_requires = [
		'numpy',
		'scipy',
		'numba',
		'future',
		],
	package_dir={'clifford':'clifford'},
	
	)

