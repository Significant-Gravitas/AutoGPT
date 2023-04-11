import numpy as np

reveal_type(np.maximum_sctype(np.float64))  # E: Type[{float64}]
reveal_type(np.maximum_sctype("f8"))  # E: Type[Any]

reveal_type(np.issctype(np.float64))  # E: bool
reveal_type(np.issctype("foo"))  # E: Literal[False]

reveal_type(np.obj2sctype(np.float64))  # E: Union[None, Type[{float64}]]
reveal_type(np.obj2sctype(np.float64, default=False))  # E: Union[builtins.bool, Type[{float64}]]
reveal_type(np.obj2sctype("S8"))  # E: Union[None, Type[Any]]
reveal_type(np.obj2sctype("S8", default=None))  # E: Union[None, Type[Any]]
reveal_type(np.obj2sctype("foo", default=False))  # E: Union[builtins.bool, Type[Any]]
reveal_type(np.obj2sctype(1))  # E: None
reveal_type(np.obj2sctype(1, default=False))  # E: bool

reveal_type(np.issubclass_(np.float64, float))  # E: bool
reveal_type(np.issubclass_(np.float64, (int, float)))  # E: bool
reveal_type(np.issubclass_(1, 1))  # E: Literal[False]

reveal_type(np.sctype2char("S8"))  # E: str
reveal_type(np.sctype2char(list))  # E: str

reveal_type(np.find_common_type([np.int64], [np.int64]))  # E: dtype[Any]

reveal_type(np.cast[int])  # E: _CastFunc
reveal_type(np.cast["i8"])  # E: _CastFunc
reveal_type(np.cast[np.int64])  # E: _CastFunc

reveal_type(np.nbytes[int])  # E: int
reveal_type(np.nbytes["i8"])  # E: int
reveal_type(np.nbytes[np.int64])  # E: int

reveal_type(np.ScalarType)  # E: Tuple
reveal_type(np.ScalarType[0])  # E: Type[builtins.int]
reveal_type(np.ScalarType[3])  # E: Type[builtins.bool]
reveal_type(np.ScalarType[8])  # E: Type[{csingle}]
reveal_type(np.ScalarType[10])  # E: Type[{clongdouble}]

reveal_type(np.typecodes["Character"])  # E: Literal['c']
reveal_type(np.typecodes["Complex"])  # E: Literal['FDG']
reveal_type(np.typecodes["All"])  # E: Literal['?bhilqpBHILQPefdgFDGSUVOMm']
