#!/usr/bin/env python

#import ez_setup
#ez_setup.use_setuptools()
from setuptools import setup, find_packages
from distutils.core import Extension

VERSION = '0.8'
LONG_DESCRIPTION = """
	Geometric algebras (a.k.a. Clifford algebras). 
"""
setup(name='clifford',
	version=VERSION,
	#license='gpl',
	description='Geometric algebras (a.k.a. Clifford algebras)',
	long_description=LONG_DESCRIPTION,
	author='Robert Kern',
	author_email='robert.kern@gmail.com',
	#url='http://scikit-rf.org',
	packages=find_packages(),
	install_requires = [
		'numpy',
		],
	package_dir={'clifford':'clifford'},
	
	)

