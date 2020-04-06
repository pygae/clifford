""" Tests of clifford._bit_helpers """
import pytest
import numpy as np

from clifford._bit_helpers import count_set_bits, set_bit_indices, BitPermuter


def test_count_bits():
    assert count_set_bits(0b0) == 0
    assert count_set_bits(0b1) == 1
    assert count_set_bits(0b1001) == 2
    assert count_set_bits(0b1111) == 4


def test_bit_indices():
    assert list(set_bit_indices(0b0)) == []
    assert list(set_bit_indices(0b1)) == [0]
    assert list(set_bit_indices(0b101)) == [0, 2]
    assert list(set_bit_indices(0b101010)) == [1, 3, 5]


class TestPermuter:
    @pytest.fixture(scope='class')
    def permuter(self):
        r"""
        Performs the transformation::

            src 0  1  2  x  x  x
                |  |  |
                |  \-----------\
                |     |        |
                \--------\     |
                      |  |     |
                   /--/  |     |
                   V     V     V
            dst x  1  x  3  x  5
        """
        return BitPermuter({0: 3, 1: 5, 2: 1})

    @pytest.fixture(scope='class')
    def null_permuter(self):
        return BitPermuter({})

    def test_call(self, permuter):
        assert permuter(0b000) == 0b000000

        assert permuter(0b001) == 0b001000
        assert permuter(0b010) == 0b100000
        assert permuter(0b100) == 0b000010

        # out of range bits are ignored
        assert permuter(0b100111) == 0b101010

    def test_attributes(self, permuter):
        assert permuter.domain == 0b111
        assert permuter.range == 0b101010
        assert permuter(permuter.domain) == permuter.range

    def test_attributes_null(self, null_permuter):
        assert null_permuter.domain == 0
        assert null_permuter.range == 0

    def test_inverse(self, permuter):
        # test the inverse
        assert permuter.inverse(permuter(0b001)) == 0b001
        assert permuter.inverse(permuter(0b010)) == 0b010
        assert permuter.inverse(permuter(0b100)) == 0b100

        assert permuter(permuter.inverse(0b001000)) == 0b001000
        assert permuter(permuter.inverse(0b100000)) == 0b100000
        assert permuter(permuter.inverse(0b000010)) == 0b000010

        assert permuter.inverse(permuter.range) == permuter.domain

    def test_arrays(self, permuter):
        np.testing.assert_equal(
            permuter(np.array([0b001, 0b010, 0b100])),
            np.array([0b001000, 0b100000, 0b000010]),
        )
        np.testing.assert_equal(
            permuter.inverse(np.array([0b001000, 0b100000, 0b000010])),
            np.array([0b001, 0b010, 0b100]),
        )

    def test_arrays_null(self, null_permuter):
        np.testing.assert_equal(
            null_permuter(np.array([0b001, 0b010, 0b100])),
            np.array([0, 0, 0]),
        )
        np.testing.assert_equal(
            null_permuter.inverse(np.array([0b001000, 0b100000, 0b000010])),
            np.array([0, 0, 0]),
        )
