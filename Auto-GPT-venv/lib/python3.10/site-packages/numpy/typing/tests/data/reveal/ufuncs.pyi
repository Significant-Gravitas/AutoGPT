import numpy as np
import numpy.typing as npt

f8: np.float64
AR_f8: npt.NDArray[np.float64]
AR_i8: npt.NDArray[np.int64]

reveal_type(np.absolute.__doc__)  # E: str
reveal_type(np.absolute.types)  # E: builtins.list[builtins.str]

reveal_type(np.absolute.__name__)  # E: Literal['absolute']
reveal_type(np.absolute.ntypes)  # E: Literal[20]
reveal_type(np.absolute.identity)  # E: None
reveal_type(np.absolute.nin)  # E: Literal[1]
reveal_type(np.absolute.nin)  # E: Literal[1]
reveal_type(np.absolute.nout)  # E: Literal[1]
reveal_type(np.absolute.nargs)  # E: Literal[2]
reveal_type(np.absolute.signature)  # E: None
reveal_type(np.absolute(f8))  # E: Any
reveal_type(np.absolute(AR_f8))  # E: ndarray
reveal_type(np.absolute.at(AR_f8, AR_i8))  # E: None

reveal_type(np.add.__name__)  # E: Literal['add']
reveal_type(np.add.ntypes)  # E: Literal[22]
reveal_type(np.add.identity)  # E: Literal[0]
reveal_type(np.add.nin)  # E: Literal[2]
reveal_type(np.add.nout)  # E: Literal[1]
reveal_type(np.add.nargs)  # E: Literal[3]
reveal_type(np.add.signature)  # E: None
reveal_type(np.add(f8, f8))  # E: Any
reveal_type(np.add(AR_f8, f8))  # E: ndarray
reveal_type(np.add.at(AR_f8, AR_i8, f8))  # E: None
reveal_type(np.add.reduce(AR_f8, axis=0))  # E: Any
reveal_type(np.add.accumulate(AR_f8))  # E: ndarray
reveal_type(np.add.reduceat(AR_f8, AR_i8))  # E: ndarray
reveal_type(np.add.outer(f8, f8))  # E: Any
reveal_type(np.add.outer(AR_f8, f8))  # E: ndarray

reveal_type(np.frexp.__name__)  # E: Literal['frexp']
reveal_type(np.frexp.ntypes)  # E: Literal[4]
reveal_type(np.frexp.identity)  # E: None
reveal_type(np.frexp.nin)  # E: Literal[1]
reveal_type(np.frexp.nout)  # E: Literal[2]
reveal_type(np.frexp.nargs)  # E: Literal[3]
reveal_type(np.frexp.signature)  # E: None
reveal_type(np.frexp(f8))  # E: Tuple[Any, Any]
reveal_type(np.frexp(AR_f8))  # E: Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]

reveal_type(np.divmod.__name__)  # E: Literal['divmod']
reveal_type(np.divmod.ntypes)  # E: Literal[15]
reveal_type(np.divmod.identity)  # E: None
reveal_type(np.divmod.nin)  # E: Literal[2]
reveal_type(np.divmod.nout)  # E: Literal[2]
reveal_type(np.divmod.nargs)  # E: Literal[4]
reveal_type(np.divmod.signature)  # E: None
reveal_type(np.divmod(f8, f8))  # E: Tuple[Any, Any]
reveal_type(np.divmod(AR_f8, f8))  # E: Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]

reveal_type(np.matmul.__name__)  # E: Literal['matmul']
reveal_type(np.matmul.ntypes)  # E: Literal[19]
reveal_type(np.matmul.identity)  # E: None
reveal_type(np.matmul.nin)  # E: Literal[2]
reveal_type(np.matmul.nout)  # E: Literal[1]
reveal_type(np.matmul.nargs)  # E: Literal[3]
reveal_type(np.matmul.signature)  # E: Literal['(n?,k),(k,m?)->(n?,m?)']
reveal_type(np.matmul.identity)  # E: None
reveal_type(np.matmul(AR_f8, AR_f8))  # E: Any
reveal_type(np.matmul(AR_f8, AR_f8, axes=[(0, 1), (0, 1), (0, 1)]))  # E: Any
