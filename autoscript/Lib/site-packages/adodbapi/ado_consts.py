# ADO enumerated constants documented on MSDN:
# http://msdn.microsoft.com/en-us/library/ms678353(VS.85).aspx

# IsolationLevelEnum
adXactUnspecified = -1
adXactBrowse = 0x100
adXactChaos = 0x10
adXactCursorStability = 0x1000
adXactIsolated = 0x100000
adXactReadCommitted = 0x1000
adXactReadUncommitted = 0x100
adXactRepeatableRead = 0x10000
adXactSerializable = 0x100000

# CursorLocationEnum
adUseClient = 3
adUseServer = 2

# CursorTypeEnum
adOpenDynamic = 2
adOpenForwardOnly = 0
adOpenKeyset = 1
adOpenStatic = 3
adOpenUnspecified = -1

# CommandTypeEnum
adCmdText = 1
adCmdStoredProc = 4
adSchemaTables = 20

# ParameterDirectionEnum
adParamInput = 1
adParamInputOutput = 3
adParamOutput = 2
adParamReturnValue = 4
adParamUnknown = 0
directions = {
    0: "Unknown",
    1: "Input",
    2: "Output",
    3: "InputOutput",
    4: "Return",
}


def ado_direction_name(ado_dir):
    try:
        return "adParam" + directions[ado_dir]
    except:
        return "unknown direction (" + str(ado_dir) + ")"


# ObjectStateEnum
adStateClosed = 0
adStateOpen = 1
adStateConnecting = 2
adStateExecuting = 4
adStateFetching = 8

# FieldAttributeEnum
adFldMayBeNull = 0x40

# ConnectModeEnum
adModeUnknown = 0
adModeRead = 1
adModeWrite = 2
adModeReadWrite = 3
adModeShareDenyRead = 4
adModeShareDenyWrite = 8
adModeShareExclusive = 12
adModeShareDenyNone = 16
adModeRecursive = 0x400000

# XactAttributeEnum
adXactCommitRetaining = 131072
adXactAbortRetaining = 262144

ado_error_TIMEOUT = -2147217871

# DataTypeEnum - ADO Data types documented at:
# http://msdn2.microsoft.com/en-us/library/ms675318.aspx
adArray = 0x2000
adEmpty = 0x0
adBSTR = 0x8
adBigInt = 0x14
adBinary = 0x80
adBoolean = 0xB
adChapter = 0x88
adChar = 0x81
adCurrency = 0x6
adDBDate = 0x85
adDBTime = 0x86
adDBTimeStamp = 0x87
adDate = 0x7
adDecimal = 0xE
adDouble = 0x5
adError = 0xA
adFileTime = 0x40
adGUID = 0x48
adIDispatch = 0x9
adIUnknown = 0xD
adInteger = 0x3
adLongVarBinary = 0xCD
adLongVarChar = 0xC9
adLongVarWChar = 0xCB
adNumeric = 0x83
adPropVariant = 0x8A
adSingle = 0x4
adSmallInt = 0x2
adTinyInt = 0x10
adUnsignedBigInt = 0x15
adUnsignedInt = 0x13
adUnsignedSmallInt = 0x12
adUnsignedTinyInt = 0x11
adUserDefined = 0x84
adVarBinary = 0xCC
adVarChar = 0xC8
adVarNumeric = 0x8B
adVarWChar = 0xCA
adVariant = 0xC
adWChar = 0x82
# Additional constants used by introspection but not ADO itself
AUTO_FIELD_MARKER = -1000

adTypeNames = {
    adBSTR: "adBSTR",
    adBigInt: "adBigInt",
    adBinary: "adBinary",
    adBoolean: "adBoolean",
    adChapter: "adChapter",
    adChar: "adChar",
    adCurrency: "adCurrency",
    adDBDate: "adDBDate",
    adDBTime: "adDBTime",
    adDBTimeStamp: "adDBTimeStamp",
    adDate: "adDate",
    adDecimal: "adDecimal",
    adDouble: "adDouble",
    adEmpty: "adEmpty",
    adError: "adError",
    adFileTime: "adFileTime",
    adGUID: "adGUID",
    adIDispatch: "adIDispatch",
    adIUnknown: "adIUnknown",
    adInteger: "adInteger",
    adLongVarBinary: "adLongVarBinary",
    adLongVarChar: "adLongVarChar",
    adLongVarWChar: "adLongVarWChar",
    adNumeric: "adNumeric",
    adPropVariant: "adPropVariant",
    adSingle: "adSingle",
    adSmallInt: "adSmallInt",
    adTinyInt: "adTinyInt",
    adUnsignedBigInt: "adUnsignedBigInt",
    adUnsignedInt: "adUnsignedInt",
    adUnsignedSmallInt: "adUnsignedSmallInt",
    adUnsignedTinyInt: "adUnsignedTinyInt",
    adUserDefined: "adUserDefined",
    adVarBinary: "adVarBinary",
    adVarChar: "adVarChar",
    adVarNumeric: "adVarNumeric",
    adVarWChar: "adVarWChar",
    adVariant: "adVariant",
    adWChar: "adWChar",
}


def ado_type_name(ado_type):
    return adTypeNames.get(ado_type, "unknown type (" + str(ado_type) + ")")


# here in decimal, sorted by value
# adEmpty 0 Specifies no value (DBTYPE_EMPTY).
# adSmallInt 2 Indicates a two-byte signed integer (DBTYPE_I2).
# adInteger 3 Indicates a four-byte signed integer (DBTYPE_I4).
# adSingle 4 Indicates a single-precision floating-point value (DBTYPE_R4).
# adDouble 5 Indicates a double-precision floating-point value (DBTYPE_R8).
# adCurrency 6 Indicates a currency value (DBTYPE_CY). Currency is a fixed-point number
#   with four digits to the right of the decimal point. It is stored in an eight-byte signed integer scaled by 10,000.
# adDate 7 Indicates a date value (DBTYPE_DATE). A date is stored as a double, the whole part of which is
#   the number of days since December 30, 1899, and the fractional part of which is the fraction of a day.
# adBSTR 8 Indicates a null-terminated character string (Unicode) (DBTYPE_BSTR).
# adIDispatch 9 Indicates a pointer to an IDispatch interface on a COM object (DBTYPE_IDISPATCH).
# adError 10 Indicates a 32-bit error code (DBTYPE_ERROR).
# adBoolean 11 Indicates a boolean value (DBTYPE_BOOL).
# adVariant 12 Indicates an Automation Variant (DBTYPE_VARIANT).
# adIUnknown 13 Indicates a pointer to an IUnknown interface on a COM object (DBTYPE_IUNKNOWN).
# adDecimal 14 Indicates an exact numeric value with a fixed precision and scale (DBTYPE_DECIMAL).
# adTinyInt 16 Indicates a one-byte signed integer (DBTYPE_I1).
# adUnsignedTinyInt 17 Indicates a one-byte unsigned integer (DBTYPE_UI1).
# adUnsignedSmallInt 18 Indicates a two-byte unsigned integer (DBTYPE_UI2).
# adUnsignedInt 19 Indicates a four-byte unsigned integer (DBTYPE_UI4).
# adBigInt 20 Indicates an eight-byte signed integer (DBTYPE_I8).
# adUnsignedBigInt 21 Indicates an eight-byte unsigned integer (DBTYPE_UI8).
# adFileTime 64 Indicates a 64-bit value representing the number of 100-nanosecond intervals since
#    January 1, 1601 (DBTYPE_FILETIME).
# adGUID 72 Indicates a globally unique identifier (GUID) (DBTYPE_GUID).
# adBinary 128 Indicates a binary value (DBTYPE_BYTES).
# adChar 129 Indicates a string value (DBTYPE_STR).
# adWChar 130 Indicates a null-terminated Unicode character string (DBTYPE_WSTR).
# adNumeric 131 Indicates an exact numeric value with a fixed precision and scale (DBTYPE_NUMERIC).
#   adUserDefined 132 Indicates a user-defined variable (DBTYPE_UDT).
# adUserDefined 132 Indicates a user-defined variable (DBTYPE_UDT).
# adDBDate 133 Indicates a date value (yyyymmdd) (DBTYPE_DBDATE).
# adDBTime 134 Indicates a time value (hhmmss) (DBTYPE_DBTIME).
# adDBTimeStamp 135 Indicates a date/time stamp (yyyymmddhhmmss plus a fraction in billionths) (DBTYPE_DBTIMESTAMP).
# adChapter 136 Indicates a four-byte chapter value that identifies rows in a child rowset (DBTYPE_HCHAPTER).
# adPropVariant 138 Indicates an Automation PROPVARIANT (DBTYPE_PROP_VARIANT).
# adVarNumeric 139 Indicates a numeric value (Parameter object only).
# adVarChar 200 Indicates a string value (Parameter object only).
# adLongVarChar 201 Indicates a long string value (Parameter object only).
# adVarWChar 202 Indicates a null-terminated Unicode character string (Parameter object only).
# adLongVarWChar 203 Indicates a long null-terminated Unicode string value (Parameter object only).
# adVarBinary 204 Indicates a binary value (Parameter object only).
# adLongVarBinary 205 Indicates a long binary value (Parameter object only).
# adArray (Does not apply to ADOX.) 0x2000 A flag value, always combined with another data type constant,
#   that indicates an array of that other data type.

# Error codes to names
adoErrors = {
    0xE7B: "adErrBoundToCommand",
    0xE94: "adErrCannotComplete",
    0xEA4: "adErrCantChangeConnection",
    0xC94: "adErrCantChangeProvider",
    0xE8C: "adErrCantConvertvalue",
    0xE8D: "adErrCantCreate",
    0xEA3: "adErrCatalogNotSet",
    0xE8E: "adErrColumnNotOnThisRow",
    0xD5D: "adErrDataConversion",
    0xE89: "adErrDataOverflow",
    0xE9A: "adErrDelResOutOfScope",
    0xEA6: "adErrDenyNotSupported",
    0xEA7: "adErrDenyTypeNotSupported",
    0xCB3: "adErrFeatureNotAvailable",
    0xEA5: "adErrFieldsUpdateFailed",
    0xC93: "adErrIllegalOperation",
    0xCAE: "adErrInTransaction",
    0xE87: "adErrIntegrityViolation",
    0xBB9: "adErrInvalidArgument",
    0xE7D: "adErrInvalidConnection",
    0xE7C: "adErrInvalidParamInfo",
    0xE82: "adErrInvalidTransaction",
    0xE91: "adErrInvalidURL",
    0xCC1: "adErrItemNotFound",
    0xBCD: "adErrNoCurrentRecord",
    0xE83: "adErrNotExecuting",
    0xE7E: "adErrNotReentrant",
    0xE78: "adErrObjectClosed",
    0xD27: "adErrObjectInCollection",
    0xD5C: "adErrObjectNotSet",
    0xE79: "adErrObjectOpen",
    0xBBA: "adErrOpeningFile",
    0xE80: "adErrOperationCancelled",
    0xE96: "adErrOutOfSpace",
    0xE88: "adErrPermissionDenied",
    0xE9E: "adErrPropConflicting",
    0xE9B: "adErrPropInvalidColumn",
    0xE9C: "adErrPropInvalidOption",
    0xE9D: "adErrPropInvalidValue",
    0xE9F: "adErrPropNotAllSettable",
    0xEA0: "adErrPropNotSet",
    0xEA1: "adErrPropNotSettable",
    0xEA2: "adErrPropNotSupported",
    0xBB8: "adErrProviderFailed",
    0xE7A: "adErrProviderNotFound",
    0xBBB: "adErrReadFile",
    0xE93: "adErrResourceExists",
    0xE92: "adErrResourceLocked",
    0xE97: "adErrResourceOutOfScope",
    0xE8A: "adErrSchemaViolation",
    0xE8B: "adErrSignMismatch",
    0xE81: "adErrStillConnecting",
    0xE7F: "adErrStillExecuting",
    0xE90: "adErrTreePermissionDenied",
    0xE8F: "adErrURLDoesNotExist",
    0xE99: "adErrURLNamedRowDoesNotExist",
    0xE98: "adErrUnavailable",
    0xE84: "adErrUnsafeOperation",
    0xE95: "adErrVolumeNotFound",
    0xBBC: "adErrWriteFile",
}
