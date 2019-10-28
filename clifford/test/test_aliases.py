"""
Tests of the ``clifford.g*`` modules and friends.

Only used for modules which are not yet tested elsewhere """

import importlib
import pytest
import sys


all_algebras = {
    'g2', 'g2c', 'g3', 'g3_1', 'g3c', 'g4', 'gac', 'pga', 'sta'
}

# algrebras which are tested elsewhere
tested_algebras = {'g3c'}


@pytest.mark.parametrize('alg_name', all_algebras - tested_algebras)
def test_import(alg_name):
    """ Test that this file is importable, assuming nothing else is testing it """
    assert 'clifford.{}'.format(alg_name) not in sys.modules, "already tested elsewhere"

    importlib.import_module('clifford.{}'.format(alg_name))


@pytest.mark.parametrize('alg_name', tested_algebras)
def test_already_imported(alg_name):
    """ Test that some other test has already imported this module to test it """
    assert 'clifford.{}'.format(alg_name) in sys.modules
