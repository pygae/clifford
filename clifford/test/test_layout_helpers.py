import pickle

from numpy.testing import assert_equal
import pytest

from clifford._layout_helpers import BasisBladeOrder, BasisVectorIds


class TestBasisBladeOrder:

    def test_basic(self):
        order = BasisBladeOrder([0b01, 0b11, 0b10, 0b00])
        assert_equal(order.index_to_bitmap, [0b01, 0b11, 0b10, 0b00])
        assert_equal(order.bitmap_to_index, [3, 0, 2, 1])
        assert_equal(order.grades, [1, 2, 1, 0])

        with pytest.raises(ValueError):
            BasisBladeOrder([0, 0])

    def test_sparse(self):
        # -1 is inserted into the reverse mapping for missing blades
        order = BasisBladeOrder([0b01, 0b10])
        assert_equal(order.index_to_bitmap, [0b01, 0b10])
        assert_equal(order.bitmap_to_index, [-1, 0, 1, -1])
        assert_equal(order.grades, [1, 1])

    def test_eq(self):
        a = BasisBladeOrder([0b01, 0b11, 0b10, 0b00])
        b = BasisBladeOrder([0b01, 0b11, 0b10, 0b00])
        c = BasisBladeOrder([0b00, 0b01, 0b11, 0b10])
        assert a == a
        assert a == b
        assert a != c
        assert a != 0  # comparing against other objects is safe

    def test_hash(self):
        a = BasisBladeOrder([0b01, 0b11, 0b10, 0b00])
        d = {a: True, None: False}
        assert d[a] is True

    def test_pickle(self):
        a = BasisBladeOrder([0b01, 0b11, 0b10, 0b00])
        assert pickle.loads(pickle.dumps(a)) == a

    def test_repr(self):
        two = BasisBladeOrder([0b00, 0b01, 0b11, 0b10])
        assert repr(two) == "BasisBladeOrder([0b00, 0b01, 0b11, 0b10])"
        one = BasisBladeOrder([0b0, 0b1])
        assert repr(one) == "BasisBladeOrder([0b0, 0b1])"
        zero = BasisBladeOrder([])
        assert repr(zero) == "BasisBladeOrder([])"

    def test_shortlex(self):
        a = BasisBladeOrder([
            0b000,
            0b001, 0b010, 0b100,
            0b011, 0b101, 0b110,
            0b111
        ])

        b = BasisBladeOrder.shortlex(3)
        assert_equal(a.index_to_bitmap, b.index_to_bitmap)
        assert_equal(a.bitmap_to_index, b.bitmap_to_index)
        assert_equal(a.grades, b.grades)

        assert b == b
        assert a == b
        assert b == a

        assert repr(b) == "BasisBladeOrder.shortlex(3)"

        # shortlex is supposed to be smaller to save
        assert len(pickle.dumps(b)) < len(pickle.dumps(a))

        assert pickle.loads(pickle.dumps(b)) == b


class TestBasisVectorIds:

    def test_basic(self):
        # no requirement that they be integers
        a, b, c = (object() for i in range(3))
        ids = BasisVectorIds([a, b, c])

        assert ids.id_as_bitmap(a) == 0b001
        assert ids.id_as_bitmap(c) == 0b100
        with pytest.raises(ValueError):
            ids.id_as_bitmap(object())

        assert ids.bitmap_as_tuple(0b101) == (a, c)
        assert ids.bitmap_as_tuple(0b011) == (a, b)
        assert ids.bitmap_as_tuple(0b000) == ()

        assert ids.tuple_as_sign_and_bitmap((a, c)) == (1, 0b101)
        assert ids.tuple_as_sign_and_bitmap((c, a)) == (-1, 0b101)
        assert ids.tuple_as_sign_and_bitmap(()) == (1, 0b000)
        with pytest.raises(ValueError):
            ids.tuple_as_sign_and_bitmap((a, a))

        with pytest.raises(ValueError):
            BasisVectorIds([a, a])

    def test_repr(self):
        ids = BasisVectorIds([1, 2, 3])
        assert repr(ids) == "BasisVectorIds([1, 2, 3])"

    def test_ordered(self):
        ids = BasisVectorIds.ordered_integers(3)
        assert repr(ids) == "BasisVectorIds.ordered_integers(3)"
        assert list(ids.values) == [1, 2, 3]

        ids = BasisVectorIds.ordered_integers(4, first_index=0)
        assert repr(ids) == "BasisVectorIds.ordered_integers(4, first_index=0)"
        assert list(ids.values) == [0, 1, 2, 3]

    def test_pickle(self):
        ids = BasisVectorIds([1, 2, 3])
        assert pickle.loads(pickle.dumps(ids)).values == ids.values

        ids = BasisVectorIds.ordered_integers(3)
        assert pickle.loads(pickle.dumps(ids)).values == ids.values

        ids = BasisVectorIds.ordered_integers(3, first_index=0)
        assert pickle.loads(pickle.dumps(ids)).values == ids.values

    def test_augment(self):
        ids = BasisVectorIds([3, 2, 1])
        assert ids.augmented_with(3).values == [3, 2, 1, 4, 5, 6]

        ids = BasisVectorIds.ordered_integers(3)
        assert ids.augmented_with(3).values == range(1, 7)

    def test_order_from_tuples(self):
        ids = BasisVectorIds([1, 2])
        order = ids.order_from_tuples([(1, 2), (2,), (1,)])
        assert order == BasisBladeOrder([0b11, 0b10, 0b01])

        with pytest.raises(NotImplementedError):
            # sign flip not allowed
            ids.order_from_tuples([(2, 1)])
