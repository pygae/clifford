""" Tests of clifford._bit_helpers """
from clifford._bit_helpers import count_set_bits, set_bit_indices


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
