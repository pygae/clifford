from clifford import BasisVectorMap, Layout, BasisVectorIds


class TestBasisVectorMap:
	def test_same_layout(self):
		g3 = Layout([1, 1, 1], ids=BasisVectorIds(['x', 'y', 'z']))
		ex, ey, ez = g3.basis_vectors_lst
		x = ex + 2*ey + 4*ex*ez

		m = BasisVectorMap(g3, g3)
		assert m(x) == x

		m_swap = BasisVectorMap(g3, g3, ['x', 'y', 'z'], ['y', 'z', 'x'])
		assert m_swap(x) == ey + 2*ez + 4*ey*ex
