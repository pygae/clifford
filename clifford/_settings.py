"""
Global settings used through clifford.
"""

# private names for internal use within clifford

_eps = 1e-12            # float epsilon for float comparisons
_pretty = True          # pretty-print global
_print_precision = 5    # pretty printing precision on floats


# public accessor functions to get and set the settings


def pretty(precision=None):
    """Makes ``repr(MultiVector)`` default to pretty-print.

    `precision` arg can be used to set the printed precision.

    Parameters
    -----------
    precision : int
        number of sig figs to print past decimal

    Examples
    ----------
    >>> pretty(5)

    """

    global _pretty
    _pretty = True

    if precision is not None:
        print_precision(precision)


def ugly():
    """ Makes ``repr(MultiVector)`` default to eval-able representation. """
    global _pretty
    _pretty = False


def eps(newEps=None):
    """ Get/Set the epsilon for float comparisons. """

    global _eps
    if newEps is not None:
        _eps = newEps
    return _eps


def print_precision(newVal):
    """Set the epsilon for float comparisons.

    Parameters
    -----------
    newVal : int
        number of sig figs to print (see builtin `round`)

    Examples
    ----------
    >>> print_precision(5)
    """

    global _print_precision
    _print_precision = newVal
