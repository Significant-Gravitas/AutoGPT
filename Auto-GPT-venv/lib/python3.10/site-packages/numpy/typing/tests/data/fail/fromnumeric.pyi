"""Tests for :mod:`numpy.core.fromnumeric`."""

import numpy as np
import numpy.typing as npt

A = np.array(True, ndmin=2, dtype=bool)
A.setflags(write=False)
AR_U: npt.NDArray[np.str_]

a = np.bool_(True)

np.take(a, None)  # E: No overload variant
np.take(a, axis=1.0)  # E: No overload variant
np.take(A, out=1)  # E: No overload variant
np.take(A, mode="bob")  # E: No overload variant

np.reshape(a, None)  # E: No overload variant
np.reshape(A, 1, order="bob")  # E: No overload variant

np.choose(a, None)  # E: No overload variant
np.choose(a, out=1.0)  # E: No overload variant
np.choose(A, mode="bob")  # E: No overload variant

np.repeat(a, None)  # E: No overload variant
np.repeat(A, 1, axis=1.0)  # E: No overload variant

np.swapaxes(A, None, 1)  # E: No overload variant
np.swapaxes(A, 1, [0])  # E: No overload variant

np.transpose(A, axes=1.0)  # E: No overload variant

np.partition(a, None)  # E: No overload variant
np.partition(  # E: No overload variant
    a, 0, axis="bob"
)
np.partition(  # E: No overload variant
    A, 0, kind="bob"
)
np.partition(
    A, 0, order=range(5)  # E: Argument "order" to "partition" has incompatible type
)

np.argpartition(
    a, None  # E: incompatible type
)
np.argpartition(
    a, 0, axis="bob"  # E: incompatible type
)
np.argpartition(
    A, 0, kind="bob"  # E: incompatible type
)
np.argpartition(
    A, 0, order=range(5)  # E: Argument "order" to "argpartition" has incompatible type
)

np.sort(A, axis="bob")  # E: No overload variant
np.sort(A, kind="bob")  # E: No overload variant
np.sort(A, order=range(5))  # E: Argument "order" to "sort" has incompatible type

np.argsort(A, axis="bob")  # E: Argument "axis" to "argsort" has incompatible type
np.argsort(A, kind="bob")  # E: Argument "kind" to "argsort" has incompatible type
np.argsort(A, order=range(5))  # E: Argument "order" to "argsort" has incompatible type

np.argmax(A, axis="bob")  # E: No overload variant of "argmax" matches argument type
np.argmax(A, kind="bob")  # E: No overload variant of "argmax" matches argument type

np.argmin(A, axis="bob")  # E: No overload variant of "argmin" matches argument type
np.argmin(A, kind="bob")  # E: No overload variant of "argmin" matches argument type

np.searchsorted(  # E: No overload variant of "searchsorted" matches argument type
    A[0], 0, side="bob"
)
np.searchsorted(  # E: No overload variant of "searchsorted" matches argument type
    A[0], 0, sorter=1.0
)

np.resize(A, 1.0)  # E: No overload variant

np.squeeze(A, 1.0)  # E: No overload variant of "squeeze" matches argument type

np.diagonal(A, offset=None)  # E: No overload variant
np.diagonal(A, axis1="bob")  # E: No overload variant
np.diagonal(A, axis2=[])  # E: No overload variant

np.trace(A, offset=None)  # E: No overload variant
np.trace(A, axis1="bob")  # E: No overload variant
np.trace(A, axis2=[])  # E: No overload variant

np.ravel(a, order="bob")  # E: No overload variant

np.compress(  # E: No overload variant
    [True], A, axis=1.0
)

np.clip(a, 1, 2, out=1)  # E: No overload variant of "clip" matches argument type

np.sum(a, axis=1.0)  # E: No overload variant
np.sum(a, keepdims=1.0)  # E: No overload variant
np.sum(a, initial=[1])  # E: No overload variant

np.all(a, axis=1.0)  # E: No overload variant
np.all(a, keepdims=1.0)  # E: No overload variant
np.all(a, out=1.0)  # E: No overload variant

np.any(a, axis=1.0)  # E: No overload variant
np.any(a, keepdims=1.0)  # E: No overload variant
np.any(a, out=1.0)  # E: No overload variant

np.cumsum(a, axis=1.0)  # E: No overload variant
np.cumsum(a, dtype=1.0)  # E: No overload variant
np.cumsum(a, out=1.0)  # E: No overload variant

np.ptp(a, axis=1.0)  # E: No overload variant
np.ptp(a, keepdims=1.0)  # E: No overload variant
np.ptp(a, out=1.0)  # E: No overload variant

np.amax(a, axis=1.0)  # E: No overload variant
np.amax(a, keepdims=1.0)  # E: No overload variant
np.amax(a, out=1.0)  # E: No overload variant
np.amax(a, initial=[1.0])  # E: No overload variant
np.amax(a, where=[1.0])  # E: incompatible type

np.amin(a, axis=1.0)  # E: No overload variant
np.amin(a, keepdims=1.0)  # E: No overload variant
np.amin(a, out=1.0)  # E: No overload variant
np.amin(a, initial=[1.0])  # E: No overload variant
np.amin(a, where=[1.0])  # E: incompatible type

np.prod(a, axis=1.0)  # E: No overload variant
np.prod(a, out=False)  # E: No overload variant
np.prod(a, keepdims=1.0)  # E: No overload variant
np.prod(a, initial=int)  # E: No overload variant
np.prod(a, where=1.0)  # E: No overload variant
np.prod(AR_U)  # E: incompatible type

np.cumprod(a, axis=1.0)  # E: No overload variant
np.cumprod(a, out=False)  # E: No overload variant
np.cumprod(AR_U)  # E: incompatible type

np.size(a, axis=1.0)  # E: Argument "axis" to "size" has incompatible type

np.around(a, decimals=1.0)  # E: No overload variant
np.around(a, out=type)  # E: No overload variant
np.around(AR_U)  # E: incompatible type

np.mean(a, axis=1.0)  # E: No overload variant
np.mean(a, out=False)  # E: No overload variant
np.mean(a, keepdims=1.0)  # E: No overload variant
np.mean(AR_U)  # E: incompatible type

np.std(a, axis=1.0)  # E: No overload variant
np.std(a, out=False)  # E: No overload variant
np.std(a, ddof='test')  # E: No overload variant
np.std(a, keepdims=1.0)  # E: No overload variant
np.std(AR_U)  # E: incompatible type

np.var(a, axis=1.0)  # E: No overload variant
np.var(a, out=False)  # E: No overload variant
np.var(a, ddof='test')  # E: No overload variant
np.var(a, keepdims=1.0)  # E: No overload variant
np.var(AR_U)  # E: incompatible type
