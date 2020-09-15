""" Functions for string parsing """

from typing import Tuple
import re
import warnings

from ._multivector import MultiVector
from ._layout import Layout

# from the python docs, https://docs.python.org/3/library/re.html#simulating-scanf
_unsigned_float_pattern = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _match_line_offset(m: 're.Match') -> Tuple[int, int, str]:
    """ Convert a match position into a lineno, offset, and line """
    pos = m.span()[0]
    lines = m.string.split('\n')
    for line_i, line in enumerate(lines, 1):
        new_pos = pos - len(line) - 1
        if new_pos < 0:
            return line_i, pos + 1, line
        pos = new_pos
    assert False  # pragma: no cover


def _parse_error(m: 're.Match', msg: str) -> SyntaxError:
    """ Produce a syntax error indicating the given match was invalid """
    return SyntaxError(msg, (None, *_match_line_offset(m)))


def _tokenize(layout: Layout, mv_string: str):
    # Get the names of the canonical blades
    blade_name_index_map = {name: index for index, name in enumerate(layout.names)}

    tokenizer = re.Scanner([(
        r'\s+',
        lambda s, t: ('space', s.match, None)
    ), (
        r'\(',
        lambda s, t: ('(', s.match, None)
    ),  (
        r'\)',
        lambda s, t: (')', s.match, None)
    ), (
        r'[+-]',
        lambda s, t: ('sign', s.match, 1 if t == '+' else -1)
    ), (
        _unsigned_float_pattern,
        lambda s, t: ('coeff', s.match, float(t))
    ), (
        r'\^',
        lambda s, t: ('wedge', s.match, None)
    ), (
        r'\b(?:{})\b'.format('|'.join(
            re.escape(name)
            for name in layout.names
            if name
        )),
        lambda s, t: ('blade', s.match, blade_name_index_map[t])
    ), (
        r'.',
        lambda s, t: ('unrecognized', s.match, None)
    )])

    tokens, rest = tokenizer.scan(mv_string)
    assert not rest  # our unrecognized match should handle this
    return tokens + [
        ('end', re.compile(r'$').match(mv_string, len(mv_string)), None)
    ]


def parse_multivector(layout: Layout, mv_string: str) -> MultiVector:
    # Create a multivector
    mv_out = MultiVector(layout)

    # parser state
    sign = None
    coeff = None
    last_t = None

    # TODO: this could do with a recursive parser to handle parens
    for t, m, data in _tokenize(layout, mv_string):
        if t == 'space':
            continue  # don't update the tokenizer
        elif t in '()':
            # Not implemented, old behavior was to just strip these - do the same
            warnings.warn(
                "Parentheses are not parsed, behavior may be surprising",
                stacklevel=3)
            continue
        elif t == 'sign' and last_t is None:
            sign = data
        elif t == 'sign' and last_t == 'blade':
            sign = data
        elif t == 'sign' and last_t == 'sign':
            sign *= data
        elif t == 'sign' and last_t == 'coeff':
            # last coeff was a scalar
            mv_out[()] += coeff
            sign = data

        elif t == 'coeff' and last_t == 'sign':
            coeff = sign * data
        elif t == 'coeff' and last_t is None:
            coeff = data

        elif t == 'blade' and last_t == 'wedge':
            mv_out.value[data] += coeff
        elif t == 'blade' and last_t == 'sign':
            mv_out.value[data] += sign
        elif t == 'blade' and last_t is None:
            mv_out.value[data] += 1

        elif t == 'wedge' and last_t == 'coeff':
            pass

        elif t == 'end' and last_t == 'coeff':
            mv_out[()] += coeff
        elif t == 'end' and last_t == 'blade':
            pass

        else:
            raise _parse_error(m, "Unexpected {} token after {}".format(t, last_t))
        last_t = t

    return mv_out
