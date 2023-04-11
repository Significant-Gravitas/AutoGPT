import pytest

from numpy.f2py.symbolic import (
    Expr,
    Op,
    ArithOp,
    Language,
    as_symbol,
    as_number,
    as_string,
    as_array,
    as_complex,
    as_terms,
    as_factors,
    eliminate_quotes,
    insert_quotes,
    fromstring,
    as_expr,
    as_apply,
    as_numer_denom,
    as_ternary,
    as_ref,
    as_deref,
    normalize,
    as_eq,
    as_ne,
    as_lt,
    as_gt,
    as_le,
    as_ge,
)
from . import util


class TestSymbolic(util.F2PyTest):
    def test_eliminate_quotes(self):
        def worker(s):
            r, d = eliminate_quotes(s)
            s1 = insert_quotes(r, d)
            assert s1 == s

        for kind in ["", "mykind_"]:
            worker(kind + '"1234" // "ABCD"')
            worker(kind + '"1234" // ' + kind + '"ABCD"')
            worker(kind + "\"1234\" // 'ABCD'")
            worker(kind + '"1234" // ' + kind + "'ABCD'")
            worker(kind + '"1\\"2\'AB\'34"')
            worker("a = " + kind + "'1\\'2\"AB\"34'")

    def test_sanity(self):
        x = as_symbol("x")
        y = as_symbol("y")
        z = as_symbol("z")

        assert x.op == Op.SYMBOL
        assert repr(x) == "Expr(Op.SYMBOL, 'x')"
        assert x == x
        assert x != y
        assert hash(x) is not None

        n = as_number(123)
        m = as_number(456)
        assert n.op == Op.INTEGER
        assert repr(n) == "Expr(Op.INTEGER, (123, 4))"
        assert n == n
        assert n != m
        assert hash(n) is not None

        fn = as_number(12.3)
        fm = as_number(45.6)
        assert fn.op == Op.REAL
        assert repr(fn) == "Expr(Op.REAL, (12.3, 4))"
        assert fn == fn
        assert fn != fm
        assert hash(fn) is not None

        c = as_complex(1, 2)
        c2 = as_complex(3, 4)
        assert c.op == Op.COMPLEX
        assert repr(c) == ("Expr(Op.COMPLEX, (Expr(Op.INTEGER, (1, 4)),"
                           " Expr(Op.INTEGER, (2, 4))))")
        assert c == c
        assert c != c2
        assert hash(c) is not None

        s = as_string("'123'")
        s2 = as_string('"ABC"')
        assert s.op == Op.STRING
        assert repr(s) == "Expr(Op.STRING, (\"'123'\", 1))", repr(s)
        assert s == s
        assert s != s2

        a = as_array((n, m))
        b = as_array((n, ))
        assert a.op == Op.ARRAY
        assert repr(a) == ("Expr(Op.ARRAY, (Expr(Op.INTEGER, (123, 4)),"
                           " Expr(Op.INTEGER, (456, 4))))")
        assert a == a
        assert a != b

        t = as_terms(x)
        u = as_terms(y)
        assert t.op == Op.TERMS
        assert repr(t) == "Expr(Op.TERMS, {Expr(Op.SYMBOL, 'x'): 1})"
        assert t == t
        assert t != u
        assert hash(t) is not None

        v = as_factors(x)
        w = as_factors(y)
        assert v.op == Op.FACTORS
        assert repr(v) == "Expr(Op.FACTORS, {Expr(Op.SYMBOL, 'x'): 1})"
        assert v == v
        assert w != v
        assert hash(v) is not None

        t = as_ternary(x, y, z)
        u = as_ternary(x, z, y)
        assert t.op == Op.TERNARY
        assert t == t
        assert t != u
        assert hash(t) is not None

        e = as_eq(x, y)
        f = as_lt(x, y)
        assert e.op == Op.RELATIONAL
        assert e == e
        assert e != f
        assert hash(e) is not None

    def test_tostring_fortran(self):
        x = as_symbol("x")
        y = as_symbol("y")
        z = as_symbol("z")
        n = as_number(123)
        m = as_number(456)
        a = as_array((n, m))
        c = as_complex(n, m)

        assert str(x) == "x"
        assert str(n) == "123"
        assert str(a) == "[123, 456]"
        assert str(c) == "(123, 456)"

        assert str(Expr(Op.TERMS, {x: 1})) == "x"
        assert str(Expr(Op.TERMS, {x: 2})) == "2 * x"
        assert str(Expr(Op.TERMS, {x: -1})) == "-x"
        assert str(Expr(Op.TERMS, {x: -2})) == "-2 * x"
        assert str(Expr(Op.TERMS, {x: 1, y: 1})) == "x + y"
        assert str(Expr(Op.TERMS, {x: -1, y: -1})) == "-x - y"
        assert str(Expr(Op.TERMS, {x: 2, y: 3})) == "2 * x + 3 * y"
        assert str(Expr(Op.TERMS, {x: -2, y: 3})) == "-2 * x + 3 * y"
        assert str(Expr(Op.TERMS, {x: 2, y: -3})) == "2 * x - 3 * y"

        assert str(Expr(Op.FACTORS, {x: 1})) == "x"
        assert str(Expr(Op.FACTORS, {x: 2})) == "x ** 2"
        assert str(Expr(Op.FACTORS, {x: -1})) == "x ** -1"
        assert str(Expr(Op.FACTORS, {x: -2})) == "x ** -2"
        assert str(Expr(Op.FACTORS, {x: 1, y: 1})) == "x * y"
        assert str(Expr(Op.FACTORS, {x: 2, y: 3})) == "x ** 2 * y ** 3"

        v = Expr(Op.FACTORS, {x: 2, Expr(Op.TERMS, {x: 1, y: 1}): 3})
        assert str(v) == "x ** 2 * (x + y) ** 3", str(v)
        v = Expr(Op.FACTORS, {x: 2, Expr(Op.FACTORS, {x: 1, y: 1}): 3})
        assert str(v) == "x ** 2 * (x * y) ** 3", str(v)

        assert str(Expr(Op.APPLY, ("f", (), {}))) == "f()"
        assert str(Expr(Op.APPLY, ("f", (x, ), {}))) == "f(x)"
        assert str(Expr(Op.APPLY, ("f", (x, y), {}))) == "f(x, y)"
        assert str(Expr(Op.INDEXING, ("f", x))) == "f[x]"

        assert str(as_ternary(x, y, z)) == "merge(y, z, x)"
        assert str(as_eq(x, y)) == "x .eq. y"
        assert str(as_ne(x, y)) == "x .ne. y"
        assert str(as_lt(x, y)) == "x .lt. y"
        assert str(as_le(x, y)) == "x .le. y"
        assert str(as_gt(x, y)) == "x .gt. y"
        assert str(as_ge(x, y)) == "x .ge. y"

    def test_tostring_c(self):
        language = Language.C
        x = as_symbol("x")
        y = as_symbol("y")
        z = as_symbol("z")
        n = as_number(123)

        assert Expr(Op.FACTORS, {x: 2}).tostring(language=language) == "x * x"
        assert (Expr(Op.FACTORS, {
            x + y: 2
        }).tostring(language=language) == "(x + y) * (x + y)")
        assert Expr(Op.FACTORS, {
            x: 12
        }).tostring(language=language) == "pow(x, 12)"

        assert as_apply(ArithOp.DIV, x,
                        y).tostring(language=language) == "x / y"
        assert (as_apply(ArithOp.DIV, x,
                         x + y).tostring(language=language) == "x / (x + y)")
        assert (as_apply(ArithOp.DIV, x - y, x +
                         y).tostring(language=language) == "(x - y) / (x + y)")
        assert (x + (x - y) / (x + y) +
                n).tostring(language=language) == "123 + x + (x - y) / (x + y)"

        assert as_ternary(x, y, z).tostring(language=language) == "(x?y:z)"
        assert as_eq(x, y).tostring(language=language) == "x == y"
        assert as_ne(x, y).tostring(language=language) == "x != y"
        assert as_lt(x, y).tostring(language=language) == "x < y"
        assert as_le(x, y).tostring(language=language) == "x <= y"
        assert as_gt(x, y).tostring(language=language) == "x > y"
        assert as_ge(x, y).tostring(language=language) == "x >= y"

    def test_operations(self):
        x = as_symbol("x")
        y = as_symbol("y")
        z = as_symbol("z")

        assert x + x == Expr(Op.TERMS, {x: 2})
        assert x - x == Expr(Op.INTEGER, (0, 4))
        assert x + y == Expr(Op.TERMS, {x: 1, y: 1})
        assert x - y == Expr(Op.TERMS, {x: 1, y: -1})
        assert x * x == Expr(Op.FACTORS, {x: 2})
        assert x * y == Expr(Op.FACTORS, {x: 1, y: 1})

        assert +x == x
        assert -x == Expr(Op.TERMS, {x: -1}), repr(-x)
        assert 2 * x == Expr(Op.TERMS, {x: 2})
        assert 2 + x == Expr(Op.TERMS, {x: 1, as_number(1): 2})
        assert 2 * x + 3 * y == Expr(Op.TERMS, {x: 2, y: 3})
        assert (x + y) * 2 == Expr(Op.TERMS, {x: 2, y: 2})

        assert x**2 == Expr(Op.FACTORS, {x: 2})
        assert (x + y)**2 == Expr(
            Op.TERMS,
            {
                Expr(Op.FACTORS, {x: 2}): 1,
                Expr(Op.FACTORS, {y: 2}): 1,
                Expr(Op.FACTORS, {
                    x: 1,
                    y: 1
                }): 2,
            },
        )
        assert (x + y) * x == x**2 + x * y
        assert (x + y)**2 == x**2 + 2 * x * y + y**2
        assert (x + y)**2 + (x - y)**2 == 2 * x**2 + 2 * y**2
        assert (x + y) * z == x * z + y * z
        assert z * (x + y) == x * z + y * z

        assert (x / 2) == as_apply(ArithOp.DIV, x, as_number(2))
        assert (2 * x / 2) == x
        assert (3 * x / 2) == as_apply(ArithOp.DIV, 3 * x, as_number(2))
        assert (4 * x / 2) == 2 * x
        assert (5 * x / 2) == as_apply(ArithOp.DIV, 5 * x, as_number(2))
        assert (6 * x / 2) == 3 * x
        assert ((3 * 5) * x / 6) == as_apply(ArithOp.DIV, 5 * x, as_number(2))
        assert (30 * x**2 * y**4 / (24 * x**3 * y**3)) == as_apply(
            ArithOp.DIV, 5 * y, 4 * x)
        assert ((15 * x / 6) / 5) == as_apply(ArithOp.DIV, x,
                                              as_number(2)), (15 * x / 6) / 5
        assert (x / (5 / x)) == as_apply(ArithOp.DIV, x**2, as_number(5))

        assert (x / 2.0) == Expr(Op.TERMS, {x: 0.5})

        s = as_string('"ABC"')
        t = as_string('"123"')

        assert s // t == Expr(Op.STRING, ('"ABC123"', 1))
        assert s // x == Expr(Op.CONCAT, (s, x))
        assert x // s == Expr(Op.CONCAT, (x, s))

        c = as_complex(1.0, 2.0)
        assert -c == as_complex(-1.0, -2.0)
        assert c + c == as_expr((1 + 2j) * 2)
        assert c * c == as_expr((1 + 2j)**2)

    def test_substitute(self):
        x = as_symbol("x")
        y = as_symbol("y")
        z = as_symbol("z")
        a = as_array((x, y))

        assert x.substitute({x: y}) == y
        assert (x + y).substitute({x: z}) == y + z
        assert (x * y).substitute({x: z}) == y * z
        assert (x**4).substitute({x: z}) == z**4
        assert (x / y).substitute({x: z}) == z / y
        assert x.substitute({x: y + z}) == y + z
        assert a.substitute({x: y + z}) == as_array((y + z, y))

        assert as_ternary(x, y,
                          z).substitute({x: y + z}) == as_ternary(y + z, y, z)
        assert as_eq(x, y).substitute({x: y + z}) == as_eq(y + z, y)

    def test_fromstring(self):

        x = as_symbol("x")
        y = as_symbol("y")
        z = as_symbol("z")
        f = as_symbol("f")
        s = as_string('"ABC"')
        t = as_string('"123"')
        a = as_array((x, y))

        assert fromstring("x") == x
        assert fromstring("+ x") == x
        assert fromstring("-  x") == -x
        assert fromstring("x + y") == x + y
        assert fromstring("x + 1") == x + 1
        assert fromstring("x * y") == x * y
        assert fromstring("x * 2") == x * 2
        assert fromstring("x / y") == x / y
        assert fromstring("x ** 2", language=Language.Python) == x**2
        assert fromstring("x ** 2 ** 3", language=Language.Python) == x**2**3
        assert fromstring("(x + y) * z") == (x + y) * z

        assert fromstring("f(x)") == f(x)
        assert fromstring("f(x,y)") == f(x, y)
        assert fromstring("f[x]") == f[x]
        assert fromstring("f[x][y]") == f[x][y]

        assert fromstring('"ABC"') == s
        assert (normalize(
            fromstring('"ABC" // "123" ',
                       language=Language.Fortran)) == s // t)
        assert fromstring('f("ABC")') == f(s)
        assert fromstring('MYSTRKIND_"ABC"') == as_string('"ABC"', "MYSTRKIND")

        assert fromstring("(/x, y/)") == a, fromstring("(/x, y/)")
        assert fromstring("f((/x, y/))") == f(a)
        assert fromstring("(/(x+y)*z/)") == as_array(((x + y) * z, ))

        assert fromstring("123") == as_number(123)
        assert fromstring("123_2") == as_number(123, 2)
        assert fromstring("123_myintkind") == as_number(123, "myintkind")

        assert fromstring("123.0") == as_number(123.0, 4)
        assert fromstring("123.0_4") == as_number(123.0, 4)
        assert fromstring("123.0_8") == as_number(123.0, 8)
        assert fromstring("123.0e0") == as_number(123.0, 4)
        assert fromstring("123.0d0") == as_number(123.0, 8)
        assert fromstring("123d0") == as_number(123.0, 8)
        assert fromstring("123e-0") == as_number(123.0, 4)
        assert fromstring("123d+0") == as_number(123.0, 8)
        assert fromstring("123.0_myrealkind") == as_number(123.0, "myrealkind")
        assert fromstring("3E4") == as_number(30000.0, 4)

        assert fromstring("(1, 2)") == as_complex(1, 2)
        assert fromstring("(1e2, PI)") == as_complex(as_number(100.0),
                                                     as_symbol("PI"))

        assert fromstring("[1, 2]") == as_array((as_number(1), as_number(2)))

        assert fromstring("POINT(x, y=1)") == as_apply(as_symbol("POINT"),
                                                       x,
                                                       y=as_number(1))
        assert fromstring(
            'PERSON(name="John", age=50, shape=(/34, 23/))') == as_apply(
                as_symbol("PERSON"),
                name=as_string('"John"'),
                age=as_number(50),
                shape=as_array((as_number(34), as_number(23))),
            )

        assert fromstring("x?y:z") == as_ternary(x, y, z)

        assert fromstring("*x") == as_deref(x)
        assert fromstring("**x") == as_deref(as_deref(x))
        assert fromstring("&x") == as_ref(x)
        assert fromstring("(*x) * (*y)") == as_deref(x) * as_deref(y)
        assert fromstring("(*x) * *y") == as_deref(x) * as_deref(y)
        assert fromstring("*x * *y") == as_deref(x) * as_deref(y)
        assert fromstring("*x**y") == as_deref(x) * as_deref(y)

        assert fromstring("x == y") == as_eq(x, y)
        assert fromstring("x != y") == as_ne(x, y)
        assert fromstring("x < y") == as_lt(x, y)
        assert fromstring("x > y") == as_gt(x, y)
        assert fromstring("x <= y") == as_le(x, y)
        assert fromstring("x >= y") == as_ge(x, y)

        assert fromstring("x .eq. y", language=Language.Fortran) == as_eq(x, y)
        assert fromstring("x .ne. y", language=Language.Fortran) == as_ne(x, y)
        assert fromstring("x .lt. y", language=Language.Fortran) == as_lt(x, y)
        assert fromstring("x .gt. y", language=Language.Fortran) == as_gt(x, y)
        assert fromstring("x .le. y", language=Language.Fortran) == as_le(x, y)
        assert fromstring("x .ge. y", language=Language.Fortran) == as_ge(x, y)

    def test_traverse(self):
        x = as_symbol("x")
        y = as_symbol("y")
        z = as_symbol("z")
        f = as_symbol("f")

        # Use traverse to substitute a symbol
        def replace_visit(s, r=z):
            if s == x:
                return r

        assert x.traverse(replace_visit) == z
        assert y.traverse(replace_visit) == y
        assert z.traverse(replace_visit) == z
        assert (f(y)).traverse(replace_visit) == f(y)
        assert (f(x)).traverse(replace_visit) == f(z)
        assert (f[y]).traverse(replace_visit) == f[y]
        assert (f[z]).traverse(replace_visit) == f[z]
        assert (x + y + z).traverse(replace_visit) == (2 * z + y)
        assert (x +
                f(y, x - z)).traverse(replace_visit) == (z +
                                                         f(y, as_number(0)))
        assert as_eq(x, y).traverse(replace_visit) == as_eq(z, y)

        # Use traverse to collect symbols, method 1
        function_symbols = set()
        symbols = set()

        def collect_symbols(s):
            if s.op is Op.APPLY:
                oper = s.data[0]
                function_symbols.add(oper)
                if oper in symbols:
                    symbols.remove(oper)
            elif s.op is Op.SYMBOL and s not in function_symbols:
                symbols.add(s)

        (x + f(y, x - z)).traverse(collect_symbols)
        assert function_symbols == {f}
        assert symbols == {x, y, z}

        # Use traverse to collect symbols, method 2
        def collect_symbols2(expr, symbols):
            if expr.op is Op.SYMBOL:
                symbols.add(expr)

        symbols = set()
        (x + f(y, x - z)).traverse(collect_symbols2, symbols)
        assert symbols == {x, y, z, f}

        # Use traverse to partially collect symbols
        def collect_symbols3(expr, symbols):
            if expr.op is Op.APPLY:
                # skip traversing function calls
                return expr
            if expr.op is Op.SYMBOL:
                symbols.add(expr)

        symbols = set()
        (x + f(y, x - z)).traverse(collect_symbols3, symbols)
        assert symbols == {x}

    def test_linear_solve(self):
        x = as_symbol("x")
        y = as_symbol("y")
        z = as_symbol("z")

        assert x.linear_solve(x) == (as_number(1), as_number(0))
        assert (x + 1).linear_solve(x) == (as_number(1), as_number(1))
        assert (2 * x).linear_solve(x) == (as_number(2), as_number(0))
        assert (2 * x + 3).linear_solve(x) == (as_number(2), as_number(3))
        assert as_number(3).linear_solve(x) == (as_number(0), as_number(3))
        assert y.linear_solve(x) == (as_number(0), y)
        assert (y * z).linear_solve(x) == (as_number(0), y * z)

        assert (x + y).linear_solve(x) == (as_number(1), y)
        assert (z * x + y).linear_solve(x) == (z, y)
        assert ((z + y) * x + y).linear_solve(x) == (z + y, y)
        assert (z * y * x + y).linear_solve(x) == (z * y, y)

        pytest.raises(RuntimeError, lambda: (x * x).linear_solve(x))

    def test_as_numer_denom(self):
        x = as_symbol("x")
        y = as_symbol("y")
        n = as_number(123)

        assert as_numer_denom(x) == (x, as_number(1))
        assert as_numer_denom(x / n) == (x, n)
        assert as_numer_denom(n / x) == (n, x)
        assert as_numer_denom(x / y) == (x, y)
        assert as_numer_denom(x * y) == (x * y, as_number(1))
        assert as_numer_denom(n + x / y) == (x + n * y, y)
        assert as_numer_denom(n + x / (y - x / n)) == (y * n**2, y * n - x)

    def test_polynomial_atoms(self):
        x = as_symbol("x")
        y = as_symbol("y")
        n = as_number(123)

        assert x.polynomial_atoms() == {x}
        assert n.polynomial_atoms() == set()
        assert (y[x]).polynomial_atoms() == {y[x]}
        assert (y(x)).polynomial_atoms() == {y(x)}
        assert (y(x) + x).polynomial_atoms() == {y(x), x}
        assert (y(x) * x[y]).polynomial_atoms() == {y(x), x[y]}
        assert (y(x)**x).polynomial_atoms() == {y(x)}
