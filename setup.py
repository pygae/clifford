#!/usr/bin/env python

#import ez_setup
#ez_setup.use_setuptools()
from setuptools import setup, find_packages
from distutils.core import Extension

VERSION = '0.8'
LONG_DESCRIPTION = """
This module implements geometric algebras (a.k.a. Clifford algebras). For the uninitiated, a geometric algebra is an algebra of vectors of given dimensions and signature. The familiar inner (dot) product and the outer product, a generalized relative of the three-dimensional cross product, are unified in an invertible geometric product. Scalars, vectors, and higher-grade entities can be mixed freely and consistently in the form of mixed-grade multivectors.
"""
setup(name='clifford',
	version=VERSION,
	#license='gpl',
	description='Numerical Geometric Algebra Module',
	long_description=LONG_DESCRIPTION,
	author='Robert Kern',
	author_email='alexarsenovic@gmail.com',
	url='http://clifford.readthedocs.io',
	packages=find_packages(),
	install_requires = [
		'numpy',
		],
	package_dir={'clifford':'clifford'},
	
	)

