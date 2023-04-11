from typing import Literal

_BoolCodes = Literal["?", "=?", "<?", ">?", "bool", "bool_", "bool8"]

_UInt8Codes = Literal["uint8", "u1", "=u1", "<u1", ">u1"]
_UInt16Codes = Literal["uint16", "u2", "=u2", "<u2", ">u2"]
_UInt32Codes = Literal["uint32", "u4", "=u4", "<u4", ">u4"]
_UInt64Codes = Literal["uint64", "u8", "=u8", "<u8", ">u8"]

_Int8Codes = Literal["int8", "i1", "=i1", "<i1", ">i1"]
_Int16Codes = Literal["int16", "i2", "=i2", "<i2", ">i2"]
_Int32Codes = Literal["int32", "i4", "=i4", "<i4", ">i4"]
_Int64Codes = Literal["int64", "i8", "=i8", "<i8", ">i8"]

_Float16Codes = Literal["float16", "f2", "=f2", "<f2", ">f2"]
_Float32Codes = Literal["float32", "f4", "=f4", "<f4", ">f4"]
_Float64Codes = Literal["float64", "f8", "=f8", "<f8", ">f8"]

_Complex64Codes = Literal["complex64", "c8", "=c8", "<c8", ">c8"]
_Complex128Codes = Literal["complex128", "c16", "=c16", "<c16", ">c16"]

_ByteCodes = Literal["byte", "b", "=b", "<b", ">b"]
_ShortCodes = Literal["short", "h", "=h", "<h", ">h"]
_IntCCodes = Literal["intc", "i", "=i", "<i", ">i"]
_IntPCodes = Literal["intp", "int0", "p", "=p", "<p", ">p"]
_IntCodes = Literal["long", "int", "int_", "l", "=l", "<l", ">l"]
_LongLongCodes = Literal["longlong", "q", "=q", "<q", ">q"]

_UByteCodes = Literal["ubyte", "B", "=B", "<B", ">B"]
_UShortCodes = Literal["ushort", "H", "=H", "<H", ">H"]
_UIntCCodes = Literal["uintc", "I", "=I", "<I", ">I"]
_UIntPCodes = Literal["uintp", "uint0", "P", "=P", "<P", ">P"]
_UIntCodes = Literal["ulong", "uint", "L", "=L", "<L", ">L"]
_ULongLongCodes = Literal["ulonglong", "Q", "=Q", "<Q", ">Q"]

_HalfCodes = Literal["half", "e", "=e", "<e", ">e"]
_SingleCodes = Literal["single", "f", "=f", "<f", ">f"]
_DoubleCodes = Literal["double", "float", "float_", "d", "=d", "<d", ">d"]
_LongDoubleCodes = Literal["longdouble", "longfloat", "g", "=g", "<g", ">g"]

_CSingleCodes = Literal["csingle", "singlecomplex", "F", "=F", "<F", ">F"]
_CDoubleCodes = Literal["cdouble", "complex", "complex_", "cfloat", "D", "=D", "<D", ">D"]
_CLongDoubleCodes = Literal["clongdouble", "clongfloat", "longcomplex", "G", "=G", "<G", ">G"]

_StrCodes = Literal["str", "str_", "str0", "unicode", "unicode_", "U", "=U", "<U", ">U"]
_BytesCodes = Literal["bytes", "bytes_", "bytes0", "S", "=S", "<S", ">S"]
_VoidCodes = Literal["void", "void0", "V", "=V", "<V", ">V"]
_ObjectCodes = Literal["object", "object_", "O", "=O", "<O", ">O"]

_DT64Codes = Literal[
    "datetime64", "=datetime64", "<datetime64", ">datetime64",
    "datetime64[Y]", "=datetime64[Y]", "<datetime64[Y]", ">datetime64[Y]",
    "datetime64[M]", "=datetime64[M]", "<datetime64[M]", ">datetime64[M]",
    "datetime64[W]", "=datetime64[W]", "<datetime64[W]", ">datetime64[W]",
    "datetime64[D]", "=datetime64[D]", "<datetime64[D]", ">datetime64[D]",
    "datetime64[h]", "=datetime64[h]", "<datetime64[h]", ">datetime64[h]",
    "datetime64[m]", "=datetime64[m]", "<datetime64[m]", ">datetime64[m]",
    "datetime64[s]", "=datetime64[s]", "<datetime64[s]", ">datetime64[s]",
    "datetime64[ms]", "=datetime64[ms]", "<datetime64[ms]", ">datetime64[ms]",
    "datetime64[us]", "=datetime64[us]", "<datetime64[us]", ">datetime64[us]",
    "datetime64[ns]", "=datetime64[ns]", "<datetime64[ns]", ">datetime64[ns]",
    "datetime64[ps]", "=datetime64[ps]", "<datetime64[ps]", ">datetime64[ps]",
    "datetime64[fs]", "=datetime64[fs]", "<datetime64[fs]", ">datetime64[fs]",
    "datetime64[as]", "=datetime64[as]", "<datetime64[as]", ">datetime64[as]",
    "M", "=M", "<M", ">M",
    "M8", "=M8", "<M8", ">M8",
    "M8[Y]", "=M8[Y]", "<M8[Y]", ">M8[Y]",
    "M8[M]", "=M8[M]", "<M8[M]", ">M8[M]",
    "M8[W]", "=M8[W]", "<M8[W]", ">M8[W]",
    "M8[D]", "=M8[D]", "<M8[D]", ">M8[D]",
    "M8[h]", "=M8[h]", "<M8[h]", ">M8[h]",
    "M8[m]", "=M8[m]", "<M8[m]", ">M8[m]",
    "M8[s]", "=M8[s]", "<M8[s]", ">M8[s]",
    "M8[ms]", "=M8[ms]", "<M8[ms]", ">M8[ms]",
    "M8[us]", "=M8[us]", "<M8[us]", ">M8[us]",
    "M8[ns]", "=M8[ns]", "<M8[ns]", ">M8[ns]",
    "M8[ps]", "=M8[ps]", "<M8[ps]", ">M8[ps]",
    "M8[fs]", "=M8[fs]", "<M8[fs]", ">M8[fs]",
    "M8[as]", "=M8[as]", "<M8[as]", ">M8[as]",
]
_TD64Codes = Literal[
    "timedelta64", "=timedelta64", "<timedelta64", ">timedelta64",
    "timedelta64[Y]", "=timedelta64[Y]", "<timedelta64[Y]", ">timedelta64[Y]",
    "timedelta64[M]", "=timedelta64[M]", "<timedelta64[M]", ">timedelta64[M]",
    "timedelta64[W]", "=timedelta64[W]", "<timedelta64[W]", ">timedelta64[W]",
    "timedelta64[D]", "=timedelta64[D]", "<timedelta64[D]", ">timedelta64[D]",
    "timedelta64[h]", "=timedelta64[h]", "<timedelta64[h]", ">timedelta64[h]",
    "timedelta64[m]", "=timedelta64[m]", "<timedelta64[m]", ">timedelta64[m]",
    "timedelta64[s]", "=timedelta64[s]", "<timedelta64[s]", ">timedelta64[s]",
    "timedelta64[ms]", "=timedelta64[ms]", "<timedelta64[ms]", ">timedelta64[ms]",
    "timedelta64[us]", "=timedelta64[us]", "<timedelta64[us]", ">timedelta64[us]",
    "timedelta64[ns]", "=timedelta64[ns]", "<timedelta64[ns]", ">timedelta64[ns]",
    "timedelta64[ps]", "=timedelta64[ps]", "<timedelta64[ps]", ">timedelta64[ps]",
    "timedelta64[fs]", "=timedelta64[fs]", "<timedelta64[fs]", ">timedelta64[fs]",
    "timedelta64[as]", "=timedelta64[as]", "<timedelta64[as]", ">timedelta64[as]",
    "m", "=m", "<m", ">m",
    "m8", "=m8", "<m8", ">m8",
    "m8[Y]", "=m8[Y]", "<m8[Y]", ">m8[Y]",
    "m8[M]", "=m8[M]", "<m8[M]", ">m8[M]",
    "m8[W]", "=m8[W]", "<m8[W]", ">m8[W]",
    "m8[D]", "=m8[D]", "<m8[D]", ">m8[D]",
    "m8[h]", "=m8[h]", "<m8[h]", ">m8[h]",
    "m8[m]", "=m8[m]", "<m8[m]", ">m8[m]",
    "m8[s]", "=m8[s]", "<m8[s]", ">m8[s]",
    "m8[ms]", "=m8[ms]", "<m8[ms]", ">m8[ms]",
    "m8[us]", "=m8[us]", "<m8[us]", ">m8[us]",
    "m8[ns]", "=m8[ns]", "<m8[ns]", ">m8[ns]",
    "m8[ps]", "=m8[ps]", "<m8[ps]", ">m8[ps]",
    "m8[fs]", "=m8[fs]", "<m8[fs]", ">m8[fs]",
    "m8[as]", "=m8[as]", "<m8[as]", ">m8[as]",
]
