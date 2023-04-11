import numpy as np
import numpy.typing as npt

AR_b: npt.NDArray[np.bool_]
AR_u4: npt.NDArray[np.uint32]
AR_i8: npt.NDArray[np.int64]
AR_f8: npt.NDArray[np.float64]
AR_c16: npt.NDArray[np.complex128]
AR_O: npt.NDArray[np.object_]

poly_obj: np.poly1d

reveal_type(poly_obj.variable)  # E: str
reveal_type(poly_obj.order)  # E: int
reveal_type(poly_obj.o)  # E: int
reveal_type(poly_obj.roots)  # E: ndarray[Any, dtype[Any]]
reveal_type(poly_obj.r)  # E: ndarray[Any, dtype[Any]]
reveal_type(poly_obj.coeffs)  # E: ndarray[Any, dtype[Any]]
reveal_type(poly_obj.c)  # E: ndarray[Any, dtype[Any]]
reveal_type(poly_obj.coef)  # E: ndarray[Any, dtype[Any]]
reveal_type(poly_obj.coefficients)  # E: ndarray[Any, dtype[Any]]
reveal_type(poly_obj.__hash__)  # E: None

reveal_type(poly_obj(1))  # E: Any
reveal_type(poly_obj([1]))  # E: ndarray[Any, dtype[Any]]
reveal_type(poly_obj(poly_obj))  # E: poly1d

reveal_type(len(poly_obj))  # E: int
reveal_type(-poly_obj)  # E: poly1d
reveal_type(+poly_obj)  # E: poly1d

reveal_type(poly_obj * 5)  # E: poly1d
reveal_type(5 * poly_obj)  # E: poly1d
reveal_type(poly_obj + 5)  # E: poly1d
reveal_type(5 + poly_obj)  # E: poly1d
reveal_type(poly_obj - 5)  # E: poly1d
reveal_type(5 - poly_obj)  # E: poly1d
reveal_type(poly_obj**1)  # E: poly1d
reveal_type(poly_obj**1.0)  # E: poly1d
reveal_type(poly_obj / 5)  # E: poly1d
reveal_type(5 / poly_obj)  # E: poly1d

reveal_type(poly_obj[0])  # E: Any
poly_obj[0] = 5
reveal_type(iter(poly_obj))  # E: Iterator[Any]
reveal_type(poly_obj.deriv())  # E: poly1d
reveal_type(poly_obj.integ())  # E: poly1d

reveal_type(np.poly(poly_obj))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.poly(AR_f8))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.poly(AR_c16))  # E: ndarray[Any, dtype[floating[Any]]]

reveal_type(np.polyint(poly_obj))  # E: poly1d
reveal_type(np.polyint(AR_f8))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.polyint(AR_f8, k=AR_c16))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(np.polyint(AR_O, m=2))  # E: ndarray[Any, dtype[object_]]

reveal_type(np.polyder(poly_obj))  # E: poly1d
reveal_type(np.polyder(AR_f8))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.polyder(AR_c16))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(np.polyder(AR_O, m=2))  # E: ndarray[Any, dtype[object_]]

reveal_type(np.polyfit(AR_f8, AR_f8, 2))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.polyfit(AR_f8, AR_i8, 1, full=True))  # E: Tuple[ndarray[Any, dtype[{float64}]], ndarray[Any, dtype[{float64}]], ndarray[Any, dtype[signedinteger[typing._32Bit]]], ndarray[Any, dtype[{float64}]], ndarray[Any, dtype[{float64}]]]
reveal_type(np.polyfit(AR_u4, AR_f8, 1.0, cov="unscaled"))  # E: Tuple[ndarray[Any, dtype[{float64}]], ndarray[Any, dtype[{float64}]]]
reveal_type(np.polyfit(AR_c16, AR_f8, 2))  # E: ndarray[Any, dtype[{complex128}]]
reveal_type(np.polyfit(AR_f8, AR_c16, 1, full=True))  # E: Tuple[ndarray[Any, dtype[{complex128}]], ndarray[Any, dtype[{float64}]], ndarray[Any, dtype[signedinteger[typing._32Bit]]], ndarray[Any, dtype[{float64}]], ndarray[Any, dtype[{float64}]]]
reveal_type(np.polyfit(AR_u4, AR_c16, 1.0, cov=True))  # E: Tuple[ndarray[Any, dtype[{complex128}]], ndarray[Any, dtype[{complex128}]]]

reveal_type(np.polyval(AR_b, AR_b))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.polyval(AR_u4, AR_b))  # E: ndarray[Any, dtype[unsignedinteger[Any]]]
reveal_type(np.polyval(AR_i8, AR_i8))  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(np.polyval(AR_f8, AR_i8))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.polyval(AR_i8, AR_c16))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(np.polyval(AR_O, AR_O))  # E: ndarray[Any, dtype[object_]]

reveal_type(np.polyadd(poly_obj, AR_i8))  # E: poly1d
reveal_type(np.polyadd(AR_f8, poly_obj))  # E: poly1d
reveal_type(np.polyadd(AR_b, AR_b))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.polyadd(AR_u4, AR_b))  # E: ndarray[Any, dtype[unsignedinteger[Any]]]
reveal_type(np.polyadd(AR_i8, AR_i8))  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(np.polyadd(AR_f8, AR_i8))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.polyadd(AR_i8, AR_c16))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(np.polyadd(AR_O, AR_O))  # E: ndarray[Any, dtype[object_]]

reveal_type(np.polysub(poly_obj, AR_i8))  # E: poly1d
reveal_type(np.polysub(AR_f8, poly_obj))  # E: poly1d
reveal_type(np.polysub(AR_b, AR_b))  # E: <nothing>
reveal_type(np.polysub(AR_u4, AR_b))  # E: ndarray[Any, dtype[unsignedinteger[Any]]]
reveal_type(np.polysub(AR_i8, AR_i8))  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(np.polysub(AR_f8, AR_i8))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.polysub(AR_i8, AR_c16))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(np.polysub(AR_O, AR_O))  # E: ndarray[Any, dtype[object_]]

reveal_type(np.polymul(poly_obj, AR_i8))  # E: poly1d
reveal_type(np.polymul(AR_f8, poly_obj))  # E: poly1d
reveal_type(np.polymul(AR_b, AR_b))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.polymul(AR_u4, AR_b))  # E: ndarray[Any, dtype[unsignedinteger[Any]]]
reveal_type(np.polymul(AR_i8, AR_i8))  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(np.polymul(AR_f8, AR_i8))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.polymul(AR_i8, AR_c16))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(np.polymul(AR_O, AR_O))  # E: ndarray[Any, dtype[object_]]

reveal_type(np.polydiv(poly_obj, AR_i8))  # E: poly1d
reveal_type(np.polydiv(AR_f8, poly_obj))  # E: poly1d
reveal_type(np.polydiv(AR_b, AR_b))  # E: Tuple[ndarray[Any, dtype[floating[Any]]], ndarray[Any, dtype[floating[Any]]]]
reveal_type(np.polydiv(AR_u4, AR_b))  # E: Tuple[ndarray[Any, dtype[floating[Any]]], ndarray[Any, dtype[floating[Any]]]]
reveal_type(np.polydiv(AR_i8, AR_i8))  # E: Tuple[ndarray[Any, dtype[floating[Any]]], ndarray[Any, dtype[floating[Any]]]]
reveal_type(np.polydiv(AR_f8, AR_i8))  # E: Tuple[ndarray[Any, dtype[floating[Any]]], ndarray[Any, dtype[floating[Any]]]]
reveal_type(np.polydiv(AR_i8, AR_c16))  # E: Tuple[ndarray[Any, dtype[complexfloating[Any, Any]]], ndarray[Any, dtype[complexfloating[Any, Any]]]]
reveal_type(np.polydiv(AR_O, AR_O))  # E: Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]
