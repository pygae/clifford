import numpy as np

from ._layout import Layout
from ._multivector import MultiVector


class ConformalLayout(Layout):
    r"""
    A layout for a conformal algebra, which adds extra constants and helpers.

    Typically these should be constructed via :func:`clifford.conformalize`.

    .. versionadded:: 1.2.0

    Attributes
    ----------
    ep : MultiVector
        The first added basis element, :math:`e_{+}`, usually with :math:`e_{+}^2 = +1`
    en : MultiVector
        The first added basis element, :math:`e_{-}`, usually with :math:`e_{-}^2 = -1`
    eo : MultiVector
        The null basis vector at the origin, :math:`e_o = 0.5(e_{-} - e_{+})`
    einf : MultiVector
        The null vector at infinity, :math:`e_\infty = e_{-} + e_{+}`
    E0 : MultiVector
        The minkowski subspace bivector, :math:`e_\infty \wedge e_o`
    I_base : MultiVector
        The pseudoscalar of the base ga, in cga layout
    """
    def __init__(self, *args, layout=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._base_layout = layout

        ep, en = self.basis_vectors_lst[-2:]

        # setup  null basis, and minkowski subspace bivector
        eo = .5 ^ (en - ep)
        einf = en + ep
        E0 = einf ^ eo
        I_base = self.pseudoScalar*E0

        # helper properties
        self.ep = ep
        self.en = en
        self.eo = eo
        self.einf = einf
        self.E0 = E0
        self.I_base = I_base

    @classmethod
    def _from_base_layout(cls, layout, added_sig=[1, -1], **kwargs) -> 'ConformalLayout':
        """ helper to implement :func:`clifford.conformalize` """
        sig_c = list(layout.sig) + added_sig
        return cls(
            sig_c,
            ids=layout._basis_vector_ids.augmented_with(len(added_sig)),
            layout=layout, **kwargs)

    # some convenience functions
    def up(self, x: MultiVector) -> MultiVector:
        """ up-project a vector from GA to CGA """
        try:
            if x.layout == self._base_layout:
                # vector is in original space, map it into conformal space
                old_val = x.value
                new_val = np.zeros(self.gaDims)
                new_val[:len(old_val)] = old_val
                x = self.MultiVector(value=new_val)
        except(AttributeError):
            # if x is a scalar it doesnt have layout but following
            # will still work
            pass

        # then up-project into a null vector
        return x + (.5 ^ ((x**2)*self.einf)) + self.eo

    def homo(self, x: MultiVector) -> MultiVector:
        """ homogenize a CGA vector """
        return x/(-x | self.einf)[()]

    def down(self, x: MultiVector) -> MultiVector:
        """ down-project a vector from CGA to GA """
        x_down = (self.homo(x) ^ self.E0)*self.E0
        # new_val = x_down.value[:self.base_layout.gaDims]
        # create vector in self.base_layout (not cga)
        # x_down = self.base_layout.MultiVector(value=new_val)
        return x_down
