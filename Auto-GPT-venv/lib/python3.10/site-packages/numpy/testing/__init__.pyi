from numpy._pytesttester import PytestTester

from unittest import (
    TestCase as TestCase,
)

from numpy.testing._private.utils import (
    assert_equal as assert_equal,
    assert_almost_equal as assert_almost_equal,
    assert_approx_equal as assert_approx_equal,
    assert_array_equal as assert_array_equal,
    assert_array_less as assert_array_less,
    assert_string_equal as assert_string_equal,
    assert_array_almost_equal as assert_array_almost_equal,
    assert_raises as assert_raises,
    build_err_msg as build_err_msg,
    decorate_methods as decorate_methods,
    jiffies as jiffies,
    memusage as memusage,
    print_assert_equal as print_assert_equal,
    raises as raises,
    rundocs as rundocs,
    runstring as runstring,
    verbose as verbose,
    measure as measure,
    assert_ as assert_,
    assert_array_almost_equal_nulp as assert_array_almost_equal_nulp,
    assert_raises_regex as assert_raises_regex,
    assert_array_max_ulp as assert_array_max_ulp,
    assert_warns as assert_warns,
    assert_no_warnings as assert_no_warnings,
    assert_allclose as assert_allclose,
    IgnoreException as IgnoreException,
    clear_and_catch_warnings as clear_and_catch_warnings,
    SkipTest as SkipTest,
    KnownFailureException as KnownFailureException,
    temppath as temppath,
    tempdir as tempdir,
    IS_PYPY as IS_PYPY,
    IS_PYSTON as IS_PYSTON,
    HAS_REFCOUNT as HAS_REFCOUNT,
    suppress_warnings as suppress_warnings,
    assert_array_compare as assert_array_compare,
    assert_no_gc_cycles as assert_no_gc_cycles,
    break_cycles as break_cycles,
    HAS_LAPACK64 as HAS_LAPACK64,
)

__all__: list[str]
__path__: list[str]
test: PytestTester

def run_module_suite(
    file_to_run: None | str = ...,
    argv: None | list[str] = ...,
) -> None: ...
