import numpy as np

nditer_obj: np.nditer

reveal_type(np.nditer([0, 1], flags=["c_index"]))  # E: nditer
reveal_type(np.nditer([0, 1], op_flags=[["readonly", "readonly"]]))  # E: nditer
reveal_type(np.nditer([0, 1], op_dtypes=np.int_))  # E: nditer
reveal_type(np.nditer([0, 1], order="C", casting="no"))  # E: nditer

reveal_type(nditer_obj.dtypes)  # E: tuple[dtype[Any], ...]
reveal_type(nditer_obj.finished)  # E: bool
reveal_type(nditer_obj.has_delayed_bufalloc)  # E: bool
reveal_type(nditer_obj.has_index)  # E: bool
reveal_type(nditer_obj.has_multi_index)  # E: bool
reveal_type(nditer_obj.index)  # E: int
reveal_type(nditer_obj.iterationneedsapi)  # E: bool
reveal_type(nditer_obj.iterindex)  # E: int
reveal_type(nditer_obj.iterrange)  # E: tuple[builtins.int, ...]
reveal_type(nditer_obj.itersize)  # E: int
reveal_type(nditer_obj.itviews)  # E: tuple[ndarray[Any, dtype[Any]], ...]
reveal_type(nditer_obj.multi_index)  # E: tuple[builtins.int, ...]
reveal_type(nditer_obj.ndim)  # E: int
reveal_type(nditer_obj.nop)  # E: int
reveal_type(nditer_obj.operands)  # E: tuple[ndarray[Any, dtype[Any]], ...]
reveal_type(nditer_obj.shape)  # E: tuple[builtins.int, ...]
reveal_type(nditer_obj.value)  # E: tuple[ndarray[Any, dtype[Any]], ...]

reveal_type(nditer_obj.close())  # E: None
reveal_type(nditer_obj.copy())  # E: nditer
reveal_type(nditer_obj.debug_print())  # E: None
reveal_type(nditer_obj.enable_external_loop())  # E: None
reveal_type(nditer_obj.iternext())  # E: bool
reveal_type(nditer_obj.remove_axis(0))  # E: None
reveal_type(nditer_obj.remove_multi_index())  # E: None
reveal_type(nditer_obj.reset())  # E: None

reveal_type(len(nditer_obj))  # E: int
reveal_type(iter(nditer_obj))  # E: nditer
reveal_type(next(nditer_obj))  # E: tuple[ndarray[Any, dtype[Any]], ...]
reveal_type(nditer_obj.__copy__())  # E: nditer
with nditer_obj as f:
    reveal_type(f)  # E: nditer
reveal_type(nditer_obj[0])  # E: ndarray[Any, dtype[Any]]
reveal_type(nditer_obj[:])  # E: tuple[ndarray[Any, dtype[Any]], ...]
nditer_obj[0] = 0
nditer_obj[:] = [0, 1]
