"""
This module helps make data libraries like `numpy` and `pandas` optional dependencies.

The libraries add up to 130MB+, which makes it challenging to deploy applications
using this library in environments with code size constraints, like AWS Lambda.

This module serves as an import proxy and provides a few utilities for dealing with the optionality.

Since the primary use case of this library (talking to the OpenAI API) doesn’t generally require data libraries,
it’s safe to make them optional. The rare case when data libraries are needed in the client is handled through
assertions with instructive error messages.

See also `setup.py`.

"""
try:
    import numpy
except ImportError:
    numpy = None

try:
    import pandas
except ImportError:
    pandas = None

HAS_NUMPY = bool(numpy)
HAS_PANDAS = bool(pandas)

INSTRUCTIONS = """

OpenAI error: 

    missing `{library}` 

This feature requires additional dependencies:

    $ pip install openai[datalib]

"""

NUMPY_INSTRUCTIONS = INSTRUCTIONS.format(library="numpy")
PANDAS_INSTRUCTIONS = INSTRUCTIONS.format(library="pandas")


class MissingDependencyError(Exception):
    pass


def assert_has_numpy():
    if not HAS_NUMPY:
        raise MissingDependencyError(NUMPY_INSTRUCTIONS)


def assert_has_pandas():
    if not HAS_PANDAS:
        raise MissingDependencyError(PANDAS_INSTRUCTIONS)
