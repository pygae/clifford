"""
.. currentmodule:: clifford.transformations

=================================================
transformations (:mod:`clifford.transformations`)
=================================================

.. versionadded:: 1.3.0


This module may in future become the home for optimized rotor transformations,
or non-linear transformations.

See the :ref:`/tutorials/linear-transformations.ipynb` tutorial for an
introduction to how to use parts of this module.

Base classes
------------

.. autodata:: Transformation

.. autoclass:: Linear

.. autoclass:: FixedLayout

Matrix-backed implementations
-----------------------------

.. autoclass:: LinearMatrix
    :members:

.. autoclass:: OutermorphismMatrix

Helper functions
----------------

.. autofunction:: between_basis_vectors

"""
from typing import Dict, Any, Callable
from abc import abstractmethod

import numpy as np

from ._bit_helpers import set_bit_indices
from ._layout_helpers import BasisBladeOrder
from ._multivector import MultiVector
from ._layout import Layout


# keep this in sync with the docstring!
__all__ = [
    'Transformation',
    'Linear',
    'FixedLayout',

    'LinearMatrix',
    'OutermorphismMatrix',

    'between_basis_vectors',
]


def _make_outermorphism(
    t_vector: np.ndarray,
    src_order: BasisBladeOrder,
    dst_order: BasisBladeOrder,
    dst_omt_func
) -> np.ndarray:
    """ Extend the matrix transformation `t_vector` to operate on multivectors

    The extension is done via an outermorphism, using the provided outer product
    function.

    The resulting matrix has its first dimension indexed by blades in
    `dst_order`, and its second dimension indexed by blades in `src_order`.
    This means that the resulting matrix can be used as
    ``to_value = M @ from_value``.

    This is an internal function used to implement the higher-level APIs.
    """
    n_dst, n_src = t_vector.shape
    full_n_dst = len(dst_order.index_to_bitmap)
    full_n_src = len(src_order.index_to_bitmap)

    t = np.zeros((full_n_dst, full_n_src), dtype=t_vector.dtype)

    # shorthand so this fits in our indexing below
    t_ind = dst_order.bitmap_to_index
    f_ind = src_order.bitmap_to_index

    # fill the vectors
    for v_dst in range(n_dst):
        for v_src in range(n_src):
            t[t_ind[1 << v_dst], f_ind[1 << v_src]] = t_vector[v_dst, v_src]

    # Fill in the rest. Note this algorithm is inefficient, as it computes
    # f(e1)^f(e2) and f(e1)^f(e2)^f(e3) without reusing intermediate results.
    for i_src in range(full_n_src):
        # vectors were already filled
        if src_order.grades[i_src] == 1:
            continue

        out = t[:, i_src]

        # f(e1^...^en) == f(e1)^...^f(en)
        out[t_ind[0]] = 1
        for v_src in set_bit_indices(src_order.index_to_bitmap[i_src]):
            out[:] = dst_omt_func(out[:], t[:, f_ind[1 << v_src]])

    return t


Transformation = Callable[[MultiVector], MultiVector]
""" A callable mapping one MultiVector to another. """


class Linear(Transformation):
    """ A transformation which is linear, such that for scalar :math:`a_i`, :math:`f(a_1 x_1 + a_2 x_2) = a_1 f(x_1) + a_2 f(x_2)`. """
    @property
    @abstractmethod
    def adjoint(self) -> "LinearMatrix":
        r""" The adjoint function :math:`\bar f`

        This is such that :math:`f(a) * b = a * \bar f(b)`, where :math:`*` is the scalar product.

        See :cite:`ga4cs` section 4.3.2.
        """
        raise NotImplementedError


class FixedLayout(Transformation):
    """
    A transformation with a fixed source and destination layout.

    Parameters
    ----------
    layout_src : :class:`~clifford.Layout` of S dimensions
        The layout from which this transformation takes multivectors as input
    layout_dst :  :class:`~clifford.Layout` of D dimensions
        The layout in which this transformation produces multivectors as output.
        Defaults to the same as the input.
    """
    def __init__(self, layout_src: Layout, layout_dst: Layout = None):
        if layout_dst is None:
            layout_dst = layout_src
        self.layout_src = layout_src
        self.layout_dst = layout_dst

    @abstractmethod
    def __call__(self, mv: MultiVector) -> MultiVector:
        """ Apply this transformation """
        if mv.layout != self.layout_src:
            raise ValueError("Object is from the wrong layout")
        # superclass should produce the result

    def _repr_pretty_layout_args_helper(self, p):
        """ helper function to be called by _repr_pretty_ in subclasses to print the layout arguments """
        if self.layout_src is not self.layout_dst:
            # only show both if they're different
            prefix = 'layout_src='
            with p.group(len(prefix), prefix, ''):
                p.pretty(self.layout_src)
                p.text(',')
            p.breakable()
            prefix = 'layout_dst='
            with p.group(len(prefix), prefix, ''):
                p.pretty(self.layout_dst)
        else:
            p.pretty(self.layout_src)


class LinearMatrix(FixedLayout, Linear):
    """
    Linear transformation implemented by a matrix

    Transformations need not be grade preserving.

    Parameters
    ----------
    matrix : (2**D, 2**S) array_like
        A matrix that transforms multivectors from `layout_src` with
        :math:`2^S` elements to multivectors in `layout_dst` with :math:`2^D`
        elements, by left-multiplication.
    layout_src : ~clifford.Layout of S dimensions
        Passed on to :class:`FixedLayout`.
    layout_dst : ~clifford.Layout of D dimensions
        Passed on to :class:`FixedLayout`.

    See Also
    --------
    clifford.BladeMap : A faster but less general approach that works on basis blades

    """
    def __init__(self, matrix, layout_src: Layout, layout_dst: Layout = None):
        super().__init__(layout_src, layout_dst)
        expected_shape = (self.layout_dst.gaDims, self.layout_src.gaDims)
        if matrix.shape != expected_shape:
            raise ValueError(
                "Matrix should have shape {} to transform multivectors from "
                "layout_src to layout_dst, but has shape {}"
                .format(expected_shape, matrix.shape)
            )

        self._matrix = matrix

    def __call__(self, mv: MultiVector) -> MultiVector:
        super().__call__(mv)
        return self.layout_dst.MultiVector(value=self._matrix @ mv.value)

    @property
    def adjoint(self) -> "LinearMatrix":
        """ The adjoint transformation """
        return LinearMatrix(self._matrix.T, self.layout_dst, self.layout_src)

    @classmethod
    def from_function(cls, func: Callable[[MultiVector], MultiVector], layout_src: Layout, layout_dst: Layout = None) -> 'LinearMatrix':
        """ Build a linear transformation from the result of a function applied to each basis blade.

        Parameters
        ----------
        func :
            A function taking basis blades from `layout_src` that produces
            multivectors in `layout_dst`.
        layout_src : ~clifford.Layout of S dimensions
            The layout to pass into the generating function
        layout_dst : ~clifford.Layout of D dimensions
            The layout the generating function is expected to produce. If not
            passed, this is inferred.

        Example
        -------

        >>> from clifford import transformations, Layout
        >>> l = Layout([1, 1])
        >>> e1, e2 = l.basis_vectors_lst
        >>> rot_90 = transformations.LinearMatrix.from_function(lambda x: (1 + e1*e2)*x*(1 - e1*e2)/2, l)
        >>> rot_90(e1)
        (1.0^e2)
        >>> rot_90(e2)
        -(1.0^e1)
        >>> rot_90(e1*e2)
        (1.0^e12)

        See also
        --------
        LinearMatrix.from_rotor : a shorter way to spell the above example
        clifford.linear_operator_as_matrix : a lower-level function for working with a subset of basis blades
        """
        blades_dst = [
            func(b_src) for b_src in layout_src.blades_list
        ]

        # check the layouts of the resulting blades match
        for d in blades_dst:
            if layout_dst is None:
                layout_dst = d.layout
            elif d.layout != layout_dst:
                raise ValueError(
                    "result of func() is from the wrong layout, expected: {}, "
                    "got {}.".format(layout_dst, d.layout))

        matrix = np.array([b_dst.value for b_dst in blades_dst])
        return cls(matrix, layout_src, layout_dst)

    @classmethod
    def from_rotor(cls, rotor: MultiVector) -> 'LinearMatrix':
        """ Build a linear transformation from the result of applying a rotor sandwich.

        The resulting transformation operates within the algebra of the provided rotor.

        Parameters
        ----------
        rotor :
            The rotor to apply

        Example
        -------

        >>> from clifford import transformations, Layout
        >>> l = Layout([1, 1])
        >>> e1, e2 = l.basis_vectors_lst
        >>> rot_90 = transformations.LinearMatrix.from_rotor(1 + e1*e2)
        >>> rot_90(e1)
        (1.0^e2)
        >>> rot_90(e2)
        -(1.0^e1)
        >>> rot_90(e1*e2)
        (1.0^e12)
        """
        # precompute for speed
        rotor_rev = ~rotor
        rotor_mag2 = rotor.mag2()
        return cls.from_function(
            lambda x: (rotor*x*rotor_rev)/rotor_mag2,
            rotor.layout, rotor.layout
        )

    def _repr_pretty_(self, p, cycle):
        if cycle:
            raise RuntimeError("Should not be cyclic")

        prefix = '{}('.format(type(self).__name__)

        with p.group(len(prefix), prefix, ')'):
            p.pretty(self._matrix)
            p.text(',')
            p.breakable()
            self._repr_pretty_layout_args_helper(p)


class OutermorphismMatrix(LinearMatrix):
    r"""
    A generalization of a linear transformation to vectors via the outer product.

    Namely, given a linear transformation :math:`F(u) \to v`, this generalizes
    to the blades by outermorphism, :math:`F(u_1 \wedge u_2) \to F(u_1) \wedge F(u_2)`, and to
    the multivectors by distributivity.

    Such a transformation is grade preserving.

    See :cite:`ga4cs` Chapter 4 for more information

    Arguments
    ---------
    matrix : (D, S) array_like
        A matrix that transforms vectors from `layout_src` of size S to vectors
        in `layout_dst` of size D by left-multiplication.
    layout_src : ~clifford.Layout of S dimensions
        Passed on to :class:`FixedLayout`.
    layout_dst : ~clifford.Layout of D dimensions
        Passed on to :class:`FixedLayout`.

    Example
    -------
    We can construct a simple transformation that permutes and non-uniformly
    scales the basis vectors::

        >>> from clifford import transformations, Layout
        >>> layout = Layout([1, 1, 1])
        >>> e1, e2, e3 = layout.basis_vectors_lst
        >>> layout_new = Layout([1, 1, 1], names='f')
        >>> m = np.array([[0, 1, 0],
        ...               [0, 0, 2],
        ...               [3, 0, 0]])
        >>> lt = transformations.OutermorphismMatrix(m, layout, layout_new)

    Applying it to some multivectors::

        >>> # the transformation we specified
        >>> lt(e1), lt(e2), lt(e3)
        ((3^f3), (1^f1), (2^f2))

        >>> # the one deduced by outermorphism
        >>> lt(e1^e2), lt(e2^e3), lt(e1^e3)
        (-(3^f13), (2^f12), -(6^f23))

        >>> # and by distributivity
        >>> lt(1 + (e1^e2^e3))
        1 + (6^f123)

    """
    def __init__(self, matrix, layout_src: Layout, layout_dst: Layout = None):
        if layout_dst is None:
            layout_dst = layout_src
        expected_shape = (layout_dst.dims, layout_src.dims)
        if matrix.shape != expected_shape:
            raise ValueError(
                "Matrix should have shape {} to transform vectors from "
                "layout_src to layout_dst, but has shape {}"
                .format(expected_shape, matrix.shape)
            )

        full_matrix = _make_outermorphism(
            matrix,
            src_order=layout_src._basis_blade_order,
            dst_order=layout_dst._basis_blade_order,
            dst_omt_func=layout_dst.omt_func,
        )

        super().__init__(full_matrix, layout_src, layout_dst)
        self._vector_matrix = matrix

    def _repr_pretty_(self, p, cycle):
        if cycle:
            raise RuntimeError("Should not be cyclic")

        prefix = '{}('.format(type(self).__name__)

        with p.group(len(prefix), prefix, ')'):
            p.pretty(self._vector_matrix)
            p.text(',')
            p.breakable()
            self._repr_pretty_layout_args_helper(p)


def between_basis_vectors(layout_src: Layout, layout_dst: Layout, mapping: Dict[Any, Any]=None) -> OutermorphismMatrix:
    """
    Construct an outermorphism that maps basis vectors from one layout to basis vectors in another.

    Parameters
    ----------
    layout_src : ~clifford.Layout
        Passed on to :class:`FixedLayout`.
    layout_dst : ~clifford.Layout
        Passed on to :class:`FixedLayout`.
    mapping :
        If provided, a dictionary mapping the ids of source basis vectors
        to the ids of destination basis vectors. For example,
        ``{1: 2, 2: 3, 3: 1}`` would permute the basis vectors of
        :mod:`clifford.g3`.

    Example
    -------
    See the tutorial on :ref:`/tutorials/apollonius-cga-augmented.ipynb` for a
    motivating example.
    """
    # TODO: we know that the matrix has at most one 1 or -1 in every row or
    # column, so it would be faster to compute this via a reorder and a sign
    # multiplication.

    src_ids = layout_src._basis_vector_ids.values
    dst_ids = layout_dst._basis_vector_ids.values

    if mapping is None:
        # map by name
        common_vector_ids = set(src_ids) & set(dst_ids)
        mapping = {v_id: v_id for v_id in common_vector_ids}

    try:
        # convert to indexes
        index_pairs = [
            (src_ids.index(src_id), dst_ids.index(dst_id))
            for src_id, dst_id in mapping.items()
        ]
    except ValueError:
        # produce a better error
        unknown_src = set(mapping.keys()) - set(src_ids)
        unknown_dst = set(mapping.keys()) - set(dst_ids)
        raise ValueError(
            "Unknown basis vectors in src: {}, dst: {}"
            .format(unknown_src, unknown_dst)
        ) from None
    del mapping

    # build the matrix of ones
    matrix = np.zeros((layout_dst.dims, layout_src.dims), dtype=int)
    for src_index, dst_index in index_pairs:
        matrix[dst_index, src_index] = 1

    return OutermorphismMatrix(matrix, layout_src, layout_dst)
