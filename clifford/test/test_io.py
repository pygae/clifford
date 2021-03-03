import numpy as np

from clifford import MVArray
from clifford.g3c import layout
from clifford.tools.g3c import random_point_pair, ConformalMVArray
from clifford.io import (
    read_ga_file, write_ga_file,
    read_json_file, write_json_file
)

from . import rng  # noqa: F401


class TestParseMultivector:

    def test_parse_multivector(self, rng):  # noqa: F811
        A = layout.randomMV(rng=rng)
        B = layout.parse_multivector(str(A))
        np.testing.assert_almost_equal(A.value, B.value, 3)


class TestHDF5BasicIO:

    def test_write_and_read(self, tmp_path, rng):  # noqa: F811
        file_name = str(tmp_path / "test.ga")

        basis_names = np.array(layout.basis_names, dtype=str)

        mv_array = ConformalMVArray([random_point_pair(rng=rng) for i in range(1000)]).value
        write_ga_file(file_name, mv_array, layout.metric, basis_names, compression=True,
                      transpose=False, sparse=False, support=False)

        data_array, metric_2, basis_names_2, support = read_ga_file(file_name)

        np.testing.assert_equal(data_array, mv_array)
        np.testing.assert_equal(layout.metric, metric_2)
        np.testing.assert_equal(basis_names, basis_names_2)

    def test_write_and_read_array(self, tmp_path, rng):  # noqa: F811
        file_name = str(tmp_path / "test.ga")

        mv_array = MVArray([random_point_pair(rng=rng) for i in range(1000)])
        mv_array.save(file_name, compression=True, transpose=False, sparse=False, support=False)

        loaded_array = layout.load_ga_file(file_name)

        np.testing.assert_equal(loaded_array.value, mv_array.value)


class TestJSONBasicIO:

    def test_write_and_read(self, tmp_path, rng):  # noqa: F811
        file_name = str(tmp_path / "test.ga.json")

        basis_names = np.array(layout.basis_names, dtype=str)

        mv_array = ConformalMVArray([random_point_pair(rng=rng) for i in range(1000)]).value
        write_json_file(file_name, mv_array, layout.metric, basis_names, compression=True,
                        transpose=False, sparse=False, support=False)

        data_array, metric_2, basis_names_2, support = read_json_file(file_name)

        np.testing.assert_equal(data_array, mv_array)
        np.testing.assert_equal(layout.metric, metric_2)
        np.testing.assert_equal(basis_names, basis_names_2)
