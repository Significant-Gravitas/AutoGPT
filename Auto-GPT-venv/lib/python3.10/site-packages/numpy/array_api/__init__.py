"""
A NumPy sub-namespace that conforms to the Python array API standard.

This submodule accompanies NEP 47, which proposes its inclusion in NumPy. It
is still considered experimental, and will issue a warning when imported.

This is a proof-of-concept namespace that wraps the corresponding NumPy
functions to give a conforming implementation of the Python array API standard
(https://data-apis.github.io/array-api/latest/). The standard is currently in
an RFC phase and comments on it are both welcome and encouraged. Comments
should be made either at https://github.com/data-apis/array-api or at
https://github.com/data-apis/consortium-feedback/discussions.

NumPy already follows the proposed spec for the most part, so this module
serves mostly as a thin wrapper around it. However, NumPy also implements a
lot of behavior that is not included in the spec, so this serves as a
restricted subset of the API. Only those functions that are part of the spec
are included in this namespace, and all functions are given with the exact
signature given in the spec, including the use of position-only arguments, and
omitting any extra keyword arguments implemented by NumPy but not part of the
spec. The behavior of some functions is also modified from the NumPy behavior
to conform to the standard. Note that the underlying array object itself is
wrapped in a wrapper Array() class, but is otherwise unchanged. This submodule
is implemented in pure Python with no C extensions.

The array API spec is designed as a "minimal API subset" and explicitly allows
libraries to include behaviors not specified by it. But users of this module
that intend to write portable code should be aware that only those behaviors
that are listed in the spec are guaranteed to be implemented across libraries.
Consequently, the NumPy implementation was chosen to be both conforming and
minimal, so that users can use this implementation of the array API namespace
and be sure that behaviors that it defines will be available in conforming
namespaces from other libraries.

A few notes about the current state of this submodule:

- There is a test suite that tests modules against the array API standard at
  https://github.com/data-apis/array-api-tests. The test suite is still a work
  in progress, but the existing tests pass on this module, with a few
  exceptions:

  - DLPack support (see https://github.com/data-apis/array-api/pull/106) is
    not included here, as it requires a full implementation in NumPy proper
    first.

  The test suite is not yet complete, and even the tests that exist are not
  guaranteed to give a comprehensive coverage of the spec. Therefore, when
  reviewing and using this submodule, you should refer to the standard
  documents themselves. There are some tests in numpy.array_api.tests, but
  they primarily focus on things that are not tested by the official array API
  test suite.

- There is a custom array object, numpy.array_api.Array, which is returned by
  all functions in this module. All functions in the array API namespace
  implicitly assume that they will only receive this object as input. The only
  way to create instances of this object is to use one of the array creation
  functions. It does not have a public constructor on the object itself. The
  object is a small wrapper class around numpy.ndarray. The main purpose of it
  is to restrict the namespace of the array object to only those dtypes and
  only those methods that are required by the spec, as well as to limit/change
  certain behavior that differs in the spec. In particular:

  - The array API namespace does not have scalar objects, only 0-D arrays.
    Operations on Array that would create a scalar in NumPy create a 0-D
    array.

  - Indexing: Only a subset of indices supported by NumPy are required by the
    spec. The Array object restricts indexing to only allow those types of
    indices that are required by the spec. See the docstring of the
    numpy.array_api.Array._validate_indices helper function for more
    information.

  - Type promotion: Some type promotion rules are different in the spec. In
    particular, the spec does not have any value-based casting. The spec also
    does not require cross-kind casting, like integer -> floating-point. Only
    those promotions that are explicitly required by the array API
    specification are allowed in this module. See NEP 47 for more info.

  - Functions do not automatically call asarray() on their input, and will not
    work if the input type is not Array. The exception is array creation
    functions, and Python operators on the Array object, which accept Python
    scalars of the same type as the array dtype.

- All functions include type annotations, corresponding to those given in the
  spec (see _typing.py for definitions of some custom types). These do not
  currently fully pass mypy due to some limitations in mypy.

- Dtype objects are just the NumPy dtype objects, e.g., float64 =
  np.dtype('float64'). The spec does not require any behavior on these dtype
  objects other than that they be accessible by name and be comparable by
  equality, but it was considered too much extra complexity to create custom
  objects to represent dtypes.

- All places where the implementations in this submodule are known to deviate
  from their corresponding functions in NumPy are marked with "# Note:"
  comments.

Still TODO in this module are:

- DLPack support for numpy.ndarray is still in progress. See
  https://github.com/numpy/numpy/pull/19083.

- The copy=False keyword argument to asarray() is not yet implemented. This
  requires support in numpy.asarray() first.

- Some functions are not yet fully tested in the array API test suite, and may
  require updates that are not yet known until the tests are written.

- The spec is still in an RFC phase and may still have minor updates, which
  will need to be reflected here.

- Complex number support in array API spec is planned but not yet finalized,
  as are the fft extension and certain linear algebra functions such as eig
  that require complex dtypes.

"""

import warnings

warnings.warn(
    "The numpy.array_api submodule is still experimental. See NEP 47.", stacklevel=2
)

__array_api_version__ = "2021.12"

__all__ = ["__array_api_version__"]

from ._constants import e, inf, nan, pi

__all__ += ["e", "inf", "nan", "pi"]

from ._creation_functions import (
    asarray,
    arange,
    empty,
    empty_like,
    eye,
    from_dlpack,
    full,
    full_like,
    linspace,
    meshgrid,
    ones,
    ones_like,
    tril,
    triu,
    zeros,
    zeros_like,
)

__all__ += [
    "asarray",
    "arange",
    "empty",
    "empty_like",
    "eye",
    "from_dlpack",
    "full",
    "full_like",
    "linspace",
    "meshgrid",
    "ones",
    "ones_like",
    "tril",
    "triu",
    "zeros",
    "zeros_like",
]

from ._data_type_functions import (
    astype,
    broadcast_arrays,
    broadcast_to,
    can_cast,
    finfo,
    iinfo,
    result_type,
)

__all__ += [
    "astype",
    "broadcast_arrays",
    "broadcast_to",
    "can_cast",
    "finfo",
    "iinfo",
    "result_type",
]

from ._dtypes import (
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float32,
    float64,
    bool,
)

__all__ += [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
    "bool",
]

from ._elementwise_functions import (
    abs,
    acos,
    acosh,
    add,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    bitwise_and,
    bitwise_left_shift,
    bitwise_invert,
    bitwise_or,
    bitwise_right_shift,
    bitwise_xor,
    ceil,
    cos,
    cosh,
    divide,
    equal,
    exp,
    expm1,
    floor,
    floor_divide,
    greater,
    greater_equal,
    isfinite,
    isinf,
    isnan,
    less,
    less_equal,
    log,
    log1p,
    log2,
    log10,
    logaddexp,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    multiply,
    negative,
    not_equal,
    positive,
    pow,
    remainder,
    round,
    sign,
    sin,
    sinh,
    square,
    sqrt,
    subtract,
    tan,
    tanh,
    trunc,
)

__all__ += [
    "abs",
    "acos",
    "acosh",
    "add",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_invert",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "ceil",
    "cos",
    "cosh",
    "divide",
    "equal",
    "exp",
    "expm1",
    "floor",
    "floor_divide",
    "greater",
    "greater_equal",
    "isfinite",
    "isinf",
    "isnan",
    "less",
    "less_equal",
    "log",
    "log1p",
    "log2",
    "log10",
    "logaddexp",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "multiply",
    "negative",
    "not_equal",
    "positive",
    "pow",
    "remainder",
    "round",
    "sign",
    "sin",
    "sinh",
    "square",
    "sqrt",
    "subtract",
    "tan",
    "tanh",
    "trunc",
]

# linalg is an extension in the array API spec, which is a sub-namespace. Only
# a subset of functions in it are imported into the top-level namespace.
from . import linalg

__all__ += ["linalg"]

from .linalg import matmul, tensordot, matrix_transpose, vecdot

__all__ += ["matmul", "tensordot", "matrix_transpose", "vecdot"]

from ._manipulation_functions import (
    concat,
    expand_dims,
    flip,
    permute_dims,
    reshape,
    roll,
    squeeze,
    stack,
)

__all__ += ["concat", "expand_dims", "flip", "permute_dims", "reshape", "roll", "squeeze", "stack"]

from ._searching_functions import argmax, argmin, nonzero, where

__all__ += ["argmax", "argmin", "nonzero", "where"]

from ._set_functions import unique_all, unique_counts, unique_inverse, unique_values

__all__ += ["unique_all", "unique_counts", "unique_inverse", "unique_values"]

from ._sorting_functions import argsort, sort

__all__ += ["argsort", "sort"]

from ._statistical_functions import max, mean, min, prod, std, sum, var

__all__ += ["max", "mean", "min", "prod", "std", "sum", "var"]

from ._utility_functions import all, any

__all__ += ["all", "any"]
