"""
Tests for the private NumPy argument parsing functionality.
They mainly exists to ensure good test coverage without having to try the
weirder cases on actual numpy functions but test them in one place.

The test function is defined in C to be equivalent to (errors may not always
match exactly, and could be adjusted):

    def func(arg1, /, arg2, *, arg3):
        i = integer(arg1)  # reproducing the 'i' parsing in Python.
        return None
"""

import pytest

import numpy as np
from numpy.core._multiarray_tests import argparse_example_function as func


def test_invalid_integers():
    with pytest.raises(TypeError,
            match="integer argument expected, got float"):
        func(1.)
    with pytest.raises(OverflowError):
        func(2**100)


def test_missing_arguments():
    with pytest.raises(TypeError,
            match="missing required positional argument 0"):
        func()
    with pytest.raises(TypeError,
            match="missing required positional argument 0"):
        func(arg2=1, arg3=4)
    with pytest.raises(TypeError,
            match=r"missing required argument \'arg2\' \(pos 1\)"):
        func(1, arg3=5)


def test_too_many_positional():
    # the second argument is positional but can be passed as keyword.
    with pytest.raises(TypeError,
            match="takes from 2 to 3 positional arguments but 4 were given"):
        func(1, 2, 3, 4)


def test_multiple_values():
    with pytest.raises(TypeError,
            match=r"given by name \('arg2'\) and position \(position 1\)"):
        func(1, 2, arg2=3)


def test_string_fallbacks():
    # We can (currently?) use numpy strings to test the "slow" fallbacks
    # that should normally not be taken due to string interning.
    arg2 = np.unicode_("arg2")
    missing_arg = np.unicode_("missing_arg")
    func(1, **{arg2: 3})
    with pytest.raises(TypeError,
            match="got an unexpected keyword argument 'missing_arg'"):
        func(2, **{missing_arg: 3})

