import pytest

from clifford import Layout


@pytest.fixture(scope='module')
def g3():
    return Layout([1, 1, 1])


class TestParser:
    def test_valid(self, g3):
        e1, e2, e3 = g3.basis_vectors_lst
        assert g3.parse_multivector("e1") == e1
        assert g3.parse_multivector("-e1") == -e1
        assert g3.parse_multivector("--e1") == e1
        assert g3.parse_multivector("1.0e2 ^ e1") == 1.0e2 * e1
        assert g3.parse_multivector("1.0E2 ^ e1") == 1.0E2 * e1
        assert g3.parse_multivector("-1 ^ e1") == -e1
        assert g3.parse_multivector("1") == 1
        assert g3.parse_multivector("1 - e12") == 1 - (e1^e2)
        assert g3.parse_multivector("e12 + 1") == (e1^e2) + 1
        assert g3.parse_multivector("1 +\n 2") == 3

    def test_parentheses(self, g3):
        """ parsing ignores these for now """
        e1, e2, e3 = g3.basis_vectors_lst
        with pytest.warns(UserWarning):
            assert g3.parse_multivector("(1^e1) + (1^e2)") == e1 + e2

        with pytest.warns(UserWarning):
            assert g3.parse_multivector("1^(e1 + 1)^e2") == e1 + e2

    @pytest.mark.parametrize(["s", "err_pos"], [(
        "+",
        " ^",
    ), (
        "",
        "^",
    ), (
        "e2 + e1^",
        "       ^",
    ), (
        "e1+\n+",
        "   \n ^",
    ), (
        "1 e1",
        "  ^",
    ), (
        "1 + e1 e2",
        "       ^",
    )])
    def test_invalid(self, g3, s, err_pos):
        """ Test that when invalid strings are passed, the invalid character is shown """
        with pytest.raises(SyntaxError) as exc_info:
            g3.parse_multivector(s)
        exc = exc_info.value

        # remove trailing spaces on the expected marker
        expected_pos = "\n".join(l.rstrip() for l in err_pos.split("\n"))
        actual_pos = (exc.lineno - 1) * "\n" + (exc.offset - 1) * " " + '^'
        assert expected_pos == actual_pos

        assert '\n' not in exc.text

    def test_unusual_names(self):
        g = Layout([1, 1], names=['', 'x', 'y', 'I'])
        x, y = g.basis_vectors_lst
        I = x ^ y
        assert g.parse_multivector('2 + I') == 2 + I
        assert g.parse_multivector('2^x + 3^y') == 2*x + 3*y
