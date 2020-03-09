import pytest

from clifford import Layout, BasisBladeOrder, BasisVectorIds


class TestLayout:
    def test_defaults(self):
        l = Layout([1, 1], names='v')
        assert l.basis_names == ['v1', 'v2']

    def test_deprecated_ctor(self):
        with pytest.warns(DeprecationWarning, match=r'constructor.*instead'):
            Layout(sig=[1, 1], bladeTupList=[(), (1,), (2,), (1, 2)], firstIdx=1, names="test")

        # should work for positional arguments too
        with pytest.warns(DeprecationWarning, match=r'constructor.*instead'):
            Layout([1, 1], [(), (1,), (2,), (1, 2)])

    def test_bad_ctor(self):
        with pytest.raises(TypeError, match='__init__'):
            Layout(sigtypo=[1, 1])

        with pytest.raises(ValueError, match='length'):
            Layout(sig=[1, 1], ids=BasisVectorIds(['not enough']))

    def test_deprecated_firstIdx(self):
        from clifford.g3 import layout
        with pytest.warns(DeprecationWarning):
            layout.firstIdx

    def test_custom_blade_ids(self):
        l = Layout(
            [1, 1],
            ids=BasisVectorIds(['y', 'x']),
            order=BasisBladeOrder([0b00, 0b10, 0b01, 0b11])
        )

        blades = l.blades

        # should match constructor order
        assert l.basis_names == ['ey', 'ex']
        assert l.basis_vectors_lst == [blades['ey'], blades['ex']]

        # should match storage order
        assert l.blades_of_grade(1) == [blades['ex'], blades['ey']]

        with pytest.raises(AttributeError):
            l.firstIdx

    def test_swapped(self):
        # swapping the order and id arguments should give an error
        with pytest.raises(TypeError, match='order must'):
            Layout(
                [1, 1],
                order=BasisVectorIds(['y', 'x'])
            )
        with pytest.raises(TypeError, match='ids must'):
            Layout(
                [1, 1],
                ids=BasisBladeOrder([0b00, 0b10, 0b01, 0b11])
            )
