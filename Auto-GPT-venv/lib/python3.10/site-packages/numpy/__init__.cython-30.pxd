# NumPy static imports for Cython >= 3.0
#
# If any of the PyArray_* functions are called, import_array must be
# called first.  This is done automatically by Cython 3.0+ if a call
# is not detected inside of the module.
#
# Author: Dag Sverre Seljebotn
#

from cpython.ref cimport Py_INCREF
from cpython.object cimport PyObject, PyTypeObject, PyObject_TypeCheck
cimport libc.stdio as stdio


cdef extern from *:
    # Leave a marker that the NumPy declarations came from NumPy itself and not from Cython.
    # See https://github.com/cython/cython/issues/3573
    """
    /* Using NumPy API declarations from "numpy/__init__.cython-30.pxd" */
    """


cdef extern from "Python.h":
    ctypedef Py_ssize_t Py_intptr_t

cdef extern from "numpy/arrayobject.h":
    ctypedef Py_intptr_t npy_intp
    ctypedef size_t npy_uintp

    cdef enum NPY_TYPES:
        NPY_BOOL
        NPY_BYTE
        NPY_UBYTE
        NPY_SHORT
        NPY_USHORT
        NPY_INT
        NPY_UINT
        NPY_LONG
        NPY_ULONG
        NPY_LONGLONG
        NPY_ULONGLONG
        NPY_FLOAT
        NPY_DOUBLE
        NPY_LONGDOUBLE
        NPY_CFLOAT
        NPY_CDOUBLE
        NPY_CLONGDOUBLE
        NPY_OBJECT
        NPY_STRING
        NPY_UNICODE
        NPY_VOID
        NPY_DATETIME
        NPY_TIMEDELTA
        NPY_NTYPES
        NPY_NOTYPE

        NPY_INT8
        NPY_INT16
        NPY_INT32
        NPY_INT64
        NPY_INT128
        NPY_INT256
        NPY_UINT8
        NPY_UINT16
        NPY_UINT32
        NPY_UINT64
        NPY_UINT128
        NPY_UINT256
        NPY_FLOAT16
        NPY_FLOAT32
        NPY_FLOAT64
        NPY_FLOAT80
        NPY_FLOAT96
        NPY_FLOAT128
        NPY_FLOAT256
        NPY_COMPLEX32
        NPY_COMPLEX64
        NPY_COMPLEX128
        NPY_COMPLEX160
        NPY_COMPLEX192
        NPY_COMPLEX256
        NPY_COMPLEX512

        NPY_INTP

    ctypedef enum NPY_ORDER:
        NPY_ANYORDER
        NPY_CORDER
        NPY_FORTRANORDER
        NPY_KEEPORDER

    ctypedef enum NPY_CASTING:
        NPY_NO_CASTING
        NPY_EQUIV_CASTING
        NPY_SAFE_CASTING
        NPY_SAME_KIND_CASTING
        NPY_UNSAFE_CASTING

    ctypedef enum NPY_CLIPMODE:
        NPY_CLIP
        NPY_WRAP
        NPY_RAISE

    ctypedef enum NPY_SCALARKIND:
        NPY_NOSCALAR,
        NPY_BOOL_SCALAR,
        NPY_INTPOS_SCALAR,
        NPY_INTNEG_SCALAR,
        NPY_FLOAT_SCALAR,
        NPY_COMPLEX_SCALAR,
        NPY_OBJECT_SCALAR

    ctypedef enum NPY_SORTKIND:
        NPY_QUICKSORT
        NPY_HEAPSORT
        NPY_MERGESORT

    ctypedef enum NPY_SEARCHSIDE:
        NPY_SEARCHLEFT
        NPY_SEARCHRIGHT

    enum:
        # DEPRECATED since NumPy 1.7 ! Do not use in new code!
        NPY_C_CONTIGUOUS
        NPY_F_CONTIGUOUS
        NPY_CONTIGUOUS
        NPY_FORTRAN
        NPY_OWNDATA
        NPY_FORCECAST
        NPY_ENSURECOPY
        NPY_ENSUREARRAY
        NPY_ELEMENTSTRIDES
        NPY_ALIGNED
        NPY_NOTSWAPPED
        NPY_WRITEABLE
        NPY_ARR_HAS_DESCR

        NPY_BEHAVED
        NPY_BEHAVED_NS
        NPY_CARRAY
        NPY_CARRAY_RO
        NPY_FARRAY
        NPY_FARRAY_RO
        NPY_DEFAULT

        NPY_IN_ARRAY
        NPY_OUT_ARRAY
        NPY_INOUT_ARRAY
        NPY_IN_FARRAY
        NPY_OUT_FARRAY
        NPY_INOUT_FARRAY

        NPY_UPDATE_ALL

    enum:
        # Added in NumPy 1.7 to replace the deprecated enums above.
        NPY_ARRAY_C_CONTIGUOUS
        NPY_ARRAY_F_CONTIGUOUS
        NPY_ARRAY_OWNDATA
        NPY_ARRAY_FORCECAST
        NPY_ARRAY_ENSURECOPY
        NPY_ARRAY_ENSUREARRAY
        NPY_ARRAY_ELEMENTSTRIDES
        NPY_ARRAY_ALIGNED
        NPY_ARRAY_NOTSWAPPED
        NPY_ARRAY_WRITEABLE
        NPY_ARRAY_WRITEBACKIFCOPY

        NPY_ARRAY_BEHAVED
        NPY_ARRAY_BEHAVED_NS
        NPY_ARRAY_CARRAY
        NPY_ARRAY_CARRAY_RO
        NPY_ARRAY_FARRAY
        NPY_ARRAY_FARRAY_RO
        NPY_ARRAY_DEFAULT

        NPY_ARRAY_IN_ARRAY
        NPY_ARRAY_OUT_ARRAY
        NPY_ARRAY_INOUT_ARRAY
        NPY_ARRAY_IN_FARRAY
        NPY_ARRAY_OUT_FARRAY
        NPY_ARRAY_INOUT_FARRAY

        NPY_ARRAY_UPDATE_ALL

    cdef enum:
        NPY_MAXDIMS

    npy_intp NPY_MAX_ELSIZE

    ctypedef void (*PyArray_VectorUnaryFunc)(void *, void *, npy_intp, void *,  void *)

    ctypedef struct PyArray_ArrayDescr:
        # shape is a tuple, but Cython doesn't support "tuple shape"
        # inside a non-PyObject declaration, so we have to declare it
        # as just a PyObject*.
        PyObject* shape

    ctypedef struct PyArray_Descr:
        pass

    ctypedef class numpy.dtype [object PyArray_Descr, check_size ignore]:
        # Use PyDataType_* macros when possible, however there are no macros
        # for accessing some of the fields, so some are defined.
        cdef PyTypeObject* typeobj
        cdef char kind
        cdef char type
        # Numpy sometimes mutates this without warning (e.g. it'll
        # sometimes change "|" to "<" in shared dtype objects on
        # little-endian machines). If this matters to you, use
        # PyArray_IsNativeByteOrder(dtype.byteorder) instead of
        # directly accessing this field.
        cdef char byteorder
        cdef char flags
        cdef int type_num
        cdef int itemsize "elsize"
        cdef int alignment
        cdef object fields
        cdef tuple names
        # Use PyDataType_HASSUBARRAY to test whether this field is
        # valid (the pointer can be NULL). Most users should access
        # this field via the inline helper method PyDataType_SHAPE.
        cdef PyArray_ArrayDescr* subarray

    ctypedef class numpy.flatiter [object PyArrayIterObject, check_size ignore]:
        # Use through macros
        pass

    ctypedef class numpy.broadcast [object PyArrayMultiIterObject, check_size ignore]:
        # Use through macros
        pass

    ctypedef struct PyArrayObject:
        # For use in situations where ndarray can't replace PyArrayObject*,
        # like PyArrayObject**.
        pass

    ctypedef class numpy.ndarray [object PyArrayObject, check_size ignore]:
        cdef __cythonbufferdefaults__ = {"mode": "strided"}

        # NOTE: no field declarations since direct access is deprecated since NumPy 1.7
        # Instead, we use properties that map to the corresponding C-API functions.

        @property
        cdef inline PyObject* base(self) nogil:
            """Returns a borrowed reference to the object owning the data/memory.
            """
            return PyArray_BASE(self)

        @property
        cdef inline dtype descr(self):
            """Returns an owned reference to the dtype of the array.
            """
            return <dtype>PyArray_DESCR(self)

        @property
        cdef inline int ndim(self) nogil:
            """Returns the number of dimensions in the array.
            """
            return PyArray_NDIM(self)

        @property
        cdef inline npy_intp *shape(self) nogil:
            """Returns a pointer to the dimensions/shape of the array.
            The number of elements matches the number of dimensions of the array (ndim).
            Can return NULL for 0-dimensional arrays.
            """
            return PyArray_DIMS(self)

        @property
        cdef inline npy_intp *strides(self) nogil:
            """Returns a pointer to the strides of the array.
            The number of elements matches the number of dimensions of the array (ndim).
            """
            return PyArray_STRIDES(self)

        @property
        cdef inline npy_intp size(self) nogil:
            """Returns the total size (in number of elements) of the array.
            """
            return PyArray_SIZE(self)

        @property
        cdef inline char* data(self) nogil:
            """The pointer to the data buffer as a char*.
            This is provided for legacy reasons to avoid direct struct field access.
            For new code that needs this access, you probably want to cast the result
            of `PyArray_DATA()` instead, which returns a 'void*'.
            """
            return PyArray_BYTES(self)

    ctypedef unsigned char      npy_bool

    ctypedef signed char      npy_byte
    ctypedef signed short     npy_short
    ctypedef signed int       npy_int
    ctypedef signed long      npy_long
    ctypedef signed long long npy_longlong

    ctypedef unsigned char      npy_ubyte
    ctypedef unsigned short     npy_ushort
    ctypedef unsigned int       npy_uint
    ctypedef unsigned long      npy_ulong
    ctypedef unsigned long long npy_ulonglong

    ctypedef float        npy_float
    ctypedef double       npy_double
    ctypedef long double  npy_longdouble

    ctypedef signed char        npy_int8
    ctypedef signed short       npy_int16
    ctypedef signed int         npy_int32
    ctypedef signed long long   npy_int64
    ctypedef signed long long   npy_int96
    ctypedef signed long long   npy_int128

    ctypedef unsigned char      npy_uint8
    ctypedef unsigned short     npy_uint16
    ctypedef unsigned int       npy_uint32
    ctypedef unsigned long long npy_uint64
    ctypedef unsigned long long npy_uint96
    ctypedef unsigned long long npy_uint128

    ctypedef float        npy_float32
    ctypedef double       npy_float64
    ctypedef long double  npy_float80
    ctypedef long double  npy_float96
    ctypedef long double  npy_float128

    ctypedef struct npy_cfloat:
        float real
        float imag

    ctypedef struct npy_cdouble:
        double real
        double imag

    ctypedef struct npy_clongdouble:
        long double real
        long double imag

    ctypedef struct npy_complex64:
        float real
        float imag

    ctypedef struct npy_complex128:
        double real
        double imag

    ctypedef struct npy_complex160:
        long double real
        long double imag

    ctypedef struct npy_complex192:
        long double real
        long double imag

    ctypedef struct npy_complex256:
        long double real
        long double imag

    ctypedef struct PyArray_Dims:
        npy_intp *ptr
        int len

    int _import_array() except -1
    # A second definition so _import_array isn't marked as used when we use it here.
    # Do not use - subject to change any time.
    int __pyx_import_array "_import_array"() except -1

    #
    # Macros from ndarrayobject.h
    #
    bint PyArray_CHKFLAGS(ndarray m, int flags) nogil
    bint PyArray_IS_C_CONTIGUOUS(ndarray arr) nogil
    bint PyArray_IS_F_CONTIGUOUS(ndarray arr) nogil
    bint PyArray_ISCONTIGUOUS(ndarray m) nogil
    bint PyArray_ISWRITEABLE(ndarray m) nogil
    bint PyArray_ISALIGNED(ndarray m) nogil

    int PyArray_NDIM(ndarray) nogil
    bint PyArray_ISONESEGMENT(ndarray) nogil
    bint PyArray_ISFORTRAN(ndarray) nogil
    int PyArray_FORTRANIF(ndarray) nogil

    void* PyArray_DATA(ndarray) nogil
    char* PyArray_BYTES(ndarray) nogil

    npy_intp* PyArray_DIMS(ndarray) nogil
    npy_intp* PyArray_STRIDES(ndarray) nogil
    npy_intp PyArray_DIM(ndarray, size_t) nogil
    npy_intp PyArray_STRIDE(ndarray, size_t) nogil

    PyObject *PyArray_BASE(ndarray) nogil  # returns borrowed reference!
    PyArray_Descr *PyArray_DESCR(ndarray) nogil  # returns borrowed reference to dtype!
    PyArray_Descr *PyArray_DTYPE(ndarray) nogil  # returns borrowed reference to dtype! NP 1.7+ alias for descr.
    int PyArray_FLAGS(ndarray) nogil
    void PyArray_CLEARFLAGS(ndarray, int flags) nogil  # Added in NumPy 1.7
    void PyArray_ENABLEFLAGS(ndarray, int flags) nogil  # Added in NumPy 1.7
    npy_intp PyArray_ITEMSIZE(ndarray) nogil
    int PyArray_TYPE(ndarray arr) nogil

    object PyArray_GETITEM(ndarray arr, void *itemptr)
    int PyArray_SETITEM(ndarray arr, void *itemptr, object obj)

    bint PyTypeNum_ISBOOL(int) nogil
    bint PyTypeNum_ISUNSIGNED(int) nogil
    bint PyTypeNum_ISSIGNED(int) nogil
    bint PyTypeNum_ISINTEGER(int) nogil
    bint PyTypeNum_ISFLOAT(int) nogil
    bint PyTypeNum_ISNUMBER(int) nogil
    bint PyTypeNum_ISSTRING(int) nogil
    bint PyTypeNum_ISCOMPLEX(int) nogil
    bint PyTypeNum_ISPYTHON(int) nogil
    bint PyTypeNum_ISFLEXIBLE(int) nogil
    bint PyTypeNum_ISUSERDEF(int) nogil
    bint PyTypeNum_ISEXTENDED(int) nogil
    bint PyTypeNum_ISOBJECT(int) nogil

    bint PyDataType_ISBOOL(dtype) nogil
    bint PyDataType_ISUNSIGNED(dtype) nogil
    bint PyDataType_ISSIGNED(dtype) nogil
    bint PyDataType_ISINTEGER(dtype) nogil
    bint PyDataType_ISFLOAT(dtype) nogil
    bint PyDataType_ISNUMBER(dtype) nogil
    bint PyDataType_ISSTRING(dtype) nogil
    bint PyDataType_ISCOMPLEX(dtype) nogil
    bint PyDataType_ISPYTHON(dtype) nogil
    bint PyDataType_ISFLEXIBLE(dtype) nogil
    bint PyDataType_ISUSERDEF(dtype) nogil
    bint PyDataType_ISEXTENDED(dtype) nogil
    bint PyDataType_ISOBJECT(dtype) nogil
    bint PyDataType_HASFIELDS(dtype) nogil
    bint PyDataType_HASSUBARRAY(dtype) nogil

    bint PyArray_ISBOOL(ndarray) nogil
    bint PyArray_ISUNSIGNED(ndarray) nogil
    bint PyArray_ISSIGNED(ndarray) nogil
    bint PyArray_ISINTEGER(ndarray) nogil
    bint PyArray_ISFLOAT(ndarray) nogil
    bint PyArray_ISNUMBER(ndarray) nogil
    bint PyArray_ISSTRING(ndarray) nogil
    bint PyArray_ISCOMPLEX(ndarray) nogil
    bint PyArray_ISPYTHON(ndarray) nogil
    bint PyArray_ISFLEXIBLE(ndarray) nogil
    bint PyArray_ISUSERDEF(ndarray) nogil
    bint PyArray_ISEXTENDED(ndarray) nogil
    bint PyArray_ISOBJECT(ndarray) nogil
    bint PyArray_HASFIELDS(ndarray) nogil

    bint PyArray_ISVARIABLE(ndarray) nogil

    bint PyArray_SAFEALIGNEDCOPY(ndarray) nogil
    bint PyArray_ISNBO(char) nogil              # works on ndarray.byteorder
    bint PyArray_IsNativeByteOrder(char) nogil  # works on ndarray.byteorder
    bint PyArray_ISNOTSWAPPED(ndarray) nogil
    bint PyArray_ISBYTESWAPPED(ndarray) nogil

    bint PyArray_FLAGSWAP(ndarray, int) nogil

    bint PyArray_ISCARRAY(ndarray) nogil
    bint PyArray_ISCARRAY_RO(ndarray) nogil
    bint PyArray_ISFARRAY(ndarray) nogil
    bint PyArray_ISFARRAY_RO(ndarray) nogil
    bint PyArray_ISBEHAVED(ndarray) nogil
    bint PyArray_ISBEHAVED_RO(ndarray) nogil


    bint PyDataType_ISNOTSWAPPED(dtype) nogil
    bint PyDataType_ISBYTESWAPPED(dtype) nogil

    bint PyArray_DescrCheck(object)

    bint PyArray_Check(object)
    bint PyArray_CheckExact(object)

    # Cannot be supported due to out arg:
    # bint PyArray_HasArrayInterfaceType(object, dtype, object, object&)
    # bint PyArray_HasArrayInterface(op, out)


    bint PyArray_IsZeroDim(object)
    # Cannot be supported due to ## ## in macro:
    # bint PyArray_IsScalar(object, verbatim work)
    bint PyArray_CheckScalar(object)
    bint PyArray_IsPythonNumber(object)
    bint PyArray_IsPythonScalar(object)
    bint PyArray_IsAnyScalar(object)
    bint PyArray_CheckAnyScalar(object)

    ndarray PyArray_GETCONTIGUOUS(ndarray)
    bint PyArray_SAMESHAPE(ndarray, ndarray) nogil
    npy_intp PyArray_SIZE(ndarray) nogil
    npy_intp PyArray_NBYTES(ndarray) nogil

    object PyArray_FROM_O(object)
    object PyArray_FROM_OF(object m, int flags)
    object PyArray_FROM_OT(object m, int type)
    object PyArray_FROM_OTF(object m, int type, int flags)
    object PyArray_FROMANY(object m, int type, int min, int max, int flags)
    object PyArray_ZEROS(int nd, npy_intp* dims, int type, int fortran)
    object PyArray_EMPTY(int nd, npy_intp* dims, int type, int fortran)
    void PyArray_FILLWBYTE(object, int val)
    npy_intp PyArray_REFCOUNT(object)
    object PyArray_ContiguousFromAny(op, int, int min_depth, int max_depth)
    unsigned char PyArray_EquivArrTypes(ndarray a1, ndarray a2)
    bint PyArray_EquivByteorders(int b1, int b2) nogil
    object PyArray_SimpleNew(int nd, npy_intp* dims, int typenum)
    object PyArray_SimpleNewFromData(int nd, npy_intp* dims, int typenum, void* data)
    #object PyArray_SimpleNewFromDescr(int nd, npy_intp* dims, dtype descr)
    object PyArray_ToScalar(void* data, ndarray arr)

    void* PyArray_GETPTR1(ndarray m, npy_intp i) nogil
    void* PyArray_GETPTR2(ndarray m, npy_intp i, npy_intp j) nogil
    void* PyArray_GETPTR3(ndarray m, npy_intp i, npy_intp j, npy_intp k) nogil
    void* PyArray_GETPTR4(ndarray m, npy_intp i, npy_intp j, npy_intp k, npy_intp l) nogil

    void PyArray_XDECREF_ERR(ndarray)
    # Cannot be supported due to out arg
    # void PyArray_DESCR_REPLACE(descr)


    object PyArray_Copy(ndarray)
    object PyArray_FromObject(object op, int type, int min_depth, int max_depth)
    object PyArray_ContiguousFromObject(object op, int type, int min_depth, int max_depth)
    object PyArray_CopyFromObject(object op, int type, int min_depth, int max_depth)

    object PyArray_Cast(ndarray mp, int type_num)
    object PyArray_Take(ndarray ap, object items, int axis)
    object PyArray_Put(ndarray ap, object items, object values)

    void PyArray_ITER_RESET(flatiter it) nogil
    void PyArray_ITER_NEXT(flatiter it) nogil
    void PyArray_ITER_GOTO(flatiter it, npy_intp* destination) nogil
    void PyArray_ITER_GOTO1D(flatiter it, npy_intp ind) nogil
    void* PyArray_ITER_DATA(flatiter it) nogil
    bint PyArray_ITER_NOTDONE(flatiter it) nogil

    void PyArray_MultiIter_RESET(broadcast multi) nogil
    void PyArray_MultiIter_NEXT(broadcast multi) nogil
    void PyArray_MultiIter_GOTO(broadcast multi, npy_intp dest) nogil
    void PyArray_MultiIter_GOTO1D(broadcast multi, npy_intp ind) nogil
    void* PyArray_MultiIter_DATA(broadcast multi, npy_intp i) nogil
    void PyArray_MultiIter_NEXTi(broadcast multi, npy_intp i) nogil
    bint PyArray_MultiIter_NOTDONE(broadcast multi) nogil

    # Functions from __multiarray_api.h

    # Functions taking dtype and returning object/ndarray are disabled
    # for now as they steal dtype references. I'm conservative and disable
    # more than is probably needed until it can be checked further.
    int PyArray_SetNumericOps        (object)
    object PyArray_GetNumericOps ()
    int PyArray_INCREF (ndarray)
    int PyArray_XDECREF (ndarray)
    void PyArray_SetStringFunction (object, int)
    dtype PyArray_DescrFromType (int)
    object PyArray_TypeObjectFromType (int)
    char * PyArray_Zero (ndarray)
    char * PyArray_One (ndarray)
    #object PyArray_CastToType (ndarray, dtype, int)
    int PyArray_CastTo (ndarray, ndarray)
    int PyArray_CastAnyTo (ndarray, ndarray)
    int PyArray_CanCastSafely (int, int)
    npy_bool PyArray_CanCastTo (dtype, dtype)
    int PyArray_ObjectType (object, int)
    dtype PyArray_DescrFromObject (object, dtype)
    #ndarray* PyArray_ConvertToCommonType (object, int *)
    dtype PyArray_DescrFromScalar (object)
    dtype PyArray_DescrFromTypeObject (object)
    npy_intp PyArray_Size (object)
    #object PyArray_Scalar (void *, dtype, object)
    #object PyArray_FromScalar (object, dtype)
    void PyArray_ScalarAsCtype (object, void *)
    #int PyArray_CastScalarToCtype (object, void *, dtype)
    #int PyArray_CastScalarDirect (object, dtype, void *, int)
    object PyArray_ScalarFromObject (object)
    #PyArray_VectorUnaryFunc * PyArray_GetCastFunc (dtype, int)
    object PyArray_FromDims (int, int *, int)
    #object PyArray_FromDimsAndDataAndDescr (int, int *, dtype, char *)
    #object PyArray_FromAny (object, dtype, int, int, int, object)
    object PyArray_EnsureArray (object)
    object PyArray_EnsureAnyArray (object)
    #object PyArray_FromFile (stdio.FILE *, dtype, npy_intp, char *)
    #object PyArray_FromString (char *, npy_intp, dtype, npy_intp, char *)
    #object PyArray_FromBuffer (object, dtype, npy_intp, npy_intp)
    #object PyArray_FromIter (object, dtype, npy_intp)
    object PyArray_Return (ndarray)
    #object PyArray_GetField (ndarray, dtype, int)
    #int PyArray_SetField (ndarray, dtype, int, object)
    object PyArray_Byteswap (ndarray, npy_bool)
    object PyArray_Resize (ndarray, PyArray_Dims *, int, NPY_ORDER)
    int PyArray_MoveInto (ndarray, ndarray)
    int PyArray_CopyInto (ndarray, ndarray)
    int PyArray_CopyAnyInto (ndarray, ndarray)
    int PyArray_CopyObject (ndarray, object)
    object PyArray_NewCopy (ndarray, NPY_ORDER)
    object PyArray_ToList (ndarray)
    object PyArray_ToString (ndarray, NPY_ORDER)
    int PyArray_ToFile (ndarray, stdio.FILE *, char *, char *)
    int PyArray_Dump (object, object, int)
    object PyArray_Dumps (object, int)
    int PyArray_ValidType (int)
    void PyArray_UpdateFlags (ndarray, int)
    object PyArray_New (type, int, npy_intp *, int, npy_intp *, void *, int, int, object)
    #object PyArray_NewFromDescr (type, dtype, int, npy_intp *, npy_intp *, void *, int, object)
    #dtype PyArray_DescrNew (dtype)
    dtype PyArray_DescrNewFromType (int)
    double PyArray_GetPriority (object, double)
    object PyArray_IterNew (object)
    object PyArray_MultiIterNew (int, ...)

    int PyArray_PyIntAsInt (object)
    npy_intp PyArray_PyIntAsIntp (object)
    int PyArray_Broadcast (broadcast)
    void PyArray_FillObjectArray (ndarray, object)
    int PyArray_FillWithScalar (ndarray, object)
    npy_bool PyArray_CheckStrides (int, int, npy_intp, npy_intp, npy_intp *, npy_intp *)
    dtype PyArray_DescrNewByteorder (dtype, char)
    object PyArray_IterAllButAxis (object, int *)
    #object PyArray_CheckFromAny (object, dtype, int, int, int, object)
    #object PyArray_FromArray (ndarray, dtype, int)
    object PyArray_FromInterface (object)
    object PyArray_FromStructInterface (object)
    #object PyArray_FromArrayAttr (object, dtype, object)
    #NPY_SCALARKIND PyArray_ScalarKind (int, ndarray*)
    int PyArray_CanCoerceScalar (int, int, NPY_SCALARKIND)
    object PyArray_NewFlagsObject (object)
    npy_bool PyArray_CanCastScalar (type, type)
    #int PyArray_CompareUCS4 (npy_ucs4 *, npy_ucs4 *, register size_t)
    int PyArray_RemoveSmallest (broadcast)
    int PyArray_ElementStrides (object)
    void PyArray_Item_INCREF (char *, dtype)
    void PyArray_Item_XDECREF (char *, dtype)
    object PyArray_FieldNames (object)
    object PyArray_Transpose (ndarray, PyArray_Dims *)
    object PyArray_TakeFrom (ndarray, object, int, ndarray, NPY_CLIPMODE)
    object PyArray_PutTo (ndarray, object, object, NPY_CLIPMODE)
    object PyArray_PutMask (ndarray, object, object)
    object PyArray_Repeat (ndarray, object, int)
    object PyArray_Choose (ndarray, object, ndarray, NPY_CLIPMODE)
    int PyArray_Sort (ndarray, int, NPY_SORTKIND)
    object PyArray_ArgSort (ndarray, int, NPY_SORTKIND)
    object PyArray_SearchSorted (ndarray, object, NPY_SEARCHSIDE, PyObject *)
    object PyArray_ArgMax (ndarray, int, ndarray)
    object PyArray_ArgMin (ndarray, int, ndarray)
    object PyArray_Reshape (ndarray, object)
    object PyArray_Newshape (ndarray, PyArray_Dims *, NPY_ORDER)
    object PyArray_Squeeze (ndarray)
    #object PyArray_View (ndarray, dtype, type)
    object PyArray_SwapAxes (ndarray, int, int)
    object PyArray_Max (ndarray, int, ndarray)
    object PyArray_Min (ndarray, int, ndarray)
    object PyArray_Ptp (ndarray, int, ndarray)
    object PyArray_Mean (ndarray, int, int, ndarray)
    object PyArray_Trace (ndarray, int, int, int, int, ndarray)
    object PyArray_Diagonal (ndarray, int, int, int)
    object PyArray_Clip (ndarray, object, object, ndarray)
    object PyArray_Conjugate (ndarray, ndarray)
    object PyArray_Nonzero (ndarray)
    object PyArray_Std (ndarray, int, int, ndarray, int)
    object PyArray_Sum (ndarray, int, int, ndarray)
    object PyArray_CumSum (ndarray, int, int, ndarray)
    object PyArray_Prod (ndarray, int, int, ndarray)
    object PyArray_CumProd (ndarray, int, int, ndarray)
    object PyArray_All (ndarray, int, ndarray)
    object PyArray_Any (ndarray, int, ndarray)
    object PyArray_Compress (ndarray, object, int, ndarray)
    object PyArray_Flatten (ndarray, NPY_ORDER)
    object PyArray_Ravel (ndarray, NPY_ORDER)
    npy_intp PyArray_MultiplyList (npy_intp *, int)
    int PyArray_MultiplyIntList (int *, int)
    void * PyArray_GetPtr (ndarray, npy_intp*)
    int PyArray_CompareLists (npy_intp *, npy_intp *, int)
    #int PyArray_AsCArray (object*, void *, npy_intp *, int, dtype)
    #int PyArray_As1D (object*, char **, int *, int)
    #int PyArray_As2D (object*, char ***, int *, int *, int)
    int PyArray_Free (object, void *)
    #int PyArray_Converter (object, object*)
    int PyArray_IntpFromSequence (object, npy_intp *, int)
    object PyArray_Concatenate (object, int)
    object PyArray_InnerProduct (object, object)
    object PyArray_MatrixProduct (object, object)
    object PyArray_CopyAndTranspose (object)
    object PyArray_Correlate (object, object, int)
    int PyArray_TypestrConvert (int, int)
    #int PyArray_DescrConverter (object, dtype*)
    #int PyArray_DescrConverter2 (object, dtype*)
    int PyArray_IntpConverter (object, PyArray_Dims *)
    #int PyArray_BufferConverter (object, chunk)
    int PyArray_AxisConverter (object, int *)
    int PyArray_BoolConverter (object, npy_bool *)
    int PyArray_ByteorderConverter (object, char *)
    int PyArray_OrderConverter (object, NPY_ORDER *)
    unsigned char PyArray_EquivTypes (dtype, dtype)
    #object PyArray_Zeros (int, npy_intp *, dtype, int)
    #object PyArray_Empty (int, npy_intp *, dtype, int)
    object PyArray_Where (object, object, object)
    object PyArray_Arange (double, double, double, int)
    #object PyArray_ArangeObj (object, object, object, dtype)
    int PyArray_SortkindConverter (object, NPY_SORTKIND *)
    object PyArray_LexSort (object, int)
    object PyArray_Round (ndarray, int, ndarray)
    unsigned char PyArray_EquivTypenums (int, int)
    int PyArray_RegisterDataType (dtype)
    int PyArray_RegisterCastFunc (dtype, int, PyArray_VectorUnaryFunc *)
    int PyArray_RegisterCanCast (dtype, int, NPY_SCALARKIND)
    #void PyArray_InitArrFuncs (PyArray_ArrFuncs *)
    object PyArray_IntTupleFromIntp (int, npy_intp *)
    int PyArray_TypeNumFromName (char *)
    int PyArray_ClipmodeConverter (object, NPY_CLIPMODE *)
    #int PyArray_OutputConverter (object, ndarray*)
    object PyArray_BroadcastToShape (object, npy_intp *, int)
    void _PyArray_SigintHandler (int)
    void* _PyArray_GetSigintBuf ()
    #int PyArray_DescrAlignConverter (object, dtype*)
    #int PyArray_DescrAlignConverter2 (object, dtype*)
    int PyArray_SearchsideConverter (object, void *)
    object PyArray_CheckAxis (ndarray, int *, int)
    npy_intp PyArray_OverflowMultiplyList (npy_intp *, int)
    int PyArray_CompareString (char *, char *, size_t)
    int PyArray_SetBaseObject(ndarray, base)  # NOTE: steals a reference to base! Use "set_array_base()" instead.


# Typedefs that matches the runtime dtype objects in
# the numpy module.

# The ones that are commented out needs an IFDEF function
# in Cython to enable them only on the right systems.

ctypedef npy_int8       int8_t
ctypedef npy_int16      int16_t
ctypedef npy_int32      int32_t
ctypedef npy_int64      int64_t
#ctypedef npy_int96      int96_t
#ctypedef npy_int128     int128_t

ctypedef npy_uint8      uint8_t
ctypedef npy_uint16     uint16_t
ctypedef npy_uint32     uint32_t
ctypedef npy_uint64     uint64_t
#ctypedef npy_uint96     uint96_t
#ctypedef npy_uint128    uint128_t

ctypedef npy_float32    float32_t
ctypedef npy_float64    float64_t
#ctypedef npy_float80    float80_t
#ctypedef npy_float128   float128_t

ctypedef float complex  complex64_t
ctypedef double complex complex128_t

# The int types are mapped a bit surprising --
# numpy.int corresponds to 'l' and numpy.long to 'q'
ctypedef npy_long       int_t
ctypedef npy_longlong   long_t
ctypedef npy_longlong   longlong_t

ctypedef npy_ulong      uint_t
ctypedef npy_ulonglong  ulong_t
ctypedef npy_ulonglong  ulonglong_t

ctypedef npy_intp       intp_t
ctypedef npy_uintp      uintp_t

ctypedef npy_double     float_t
ctypedef npy_double     double_t
ctypedef npy_longdouble longdouble_t

ctypedef npy_cfloat      cfloat_t
ctypedef npy_cdouble     cdouble_t
ctypedef npy_clongdouble clongdouble_t

ctypedef npy_cdouble     complex_t

cdef inline object PyArray_MultiIterNew1(a):
    return PyArray_MultiIterNew(1, <void*>a)

cdef inline object PyArray_MultiIterNew2(a, b):
    return PyArray_MultiIterNew(2, <void*>a, <void*>b)

cdef inline object PyArray_MultiIterNew3(a, b, c):
    return PyArray_MultiIterNew(3, <void*>a, <void*>b, <void*> c)

cdef inline object PyArray_MultiIterNew4(a, b, c, d):
    return PyArray_MultiIterNew(4, <void*>a, <void*>b, <void*>c, <void*> d)

cdef inline object PyArray_MultiIterNew5(a, b, c, d, e):
    return PyArray_MultiIterNew(5, <void*>a, <void*>b, <void*>c, <void*> d, <void*> e)

cdef inline tuple PyDataType_SHAPE(dtype d):
    if PyDataType_HASSUBARRAY(d):
        return <tuple>d.subarray.shape
    else:
        return ()


cdef extern from "numpy/ndarrayobject.h":
    PyTypeObject PyTimedeltaArrType_Type
    PyTypeObject PyDatetimeArrType_Type
    ctypedef int64_t npy_timedelta
    ctypedef int64_t npy_datetime

cdef extern from "numpy/ndarraytypes.h":
    ctypedef struct PyArray_DatetimeMetaData:
        NPY_DATETIMEUNIT base
        int64_t num

cdef extern from "numpy/arrayscalars.h":

    # abstract types
    ctypedef class numpy.generic [object PyObject]:
        pass
    ctypedef class numpy.number [object PyObject]:
        pass
    ctypedef class numpy.integer [object PyObject]:
        pass
    ctypedef class numpy.signedinteger [object PyObject]:
        pass
    ctypedef class numpy.unsignedinteger [object PyObject]:
        pass
    ctypedef class numpy.inexact [object PyObject]:
        pass
    ctypedef class numpy.floating [object PyObject]:
        pass
    ctypedef class numpy.complexfloating [object PyObject]:
        pass
    ctypedef class numpy.flexible [object PyObject]:
        pass
    ctypedef class numpy.character [object PyObject]:
        pass

    ctypedef struct PyDatetimeScalarObject:
        # PyObject_HEAD
        npy_datetime obval
        PyArray_DatetimeMetaData obmeta

    ctypedef struct PyTimedeltaScalarObject:
        # PyObject_HEAD
        npy_timedelta obval
        PyArray_DatetimeMetaData obmeta

    ctypedef enum NPY_DATETIMEUNIT:
        NPY_FR_Y
        NPY_FR_M
        NPY_FR_W
        NPY_FR_D
        NPY_FR_B
        NPY_FR_h
        NPY_FR_m
        NPY_FR_s
        NPY_FR_ms
        NPY_FR_us
        NPY_FR_ns
        NPY_FR_ps
        NPY_FR_fs
        NPY_FR_as


#
# ufunc API
#

cdef extern from "numpy/ufuncobject.h":

    ctypedef void (*PyUFuncGenericFunction) (char **, npy_intp *, npy_intp *, void *)

    ctypedef class numpy.ufunc [object PyUFuncObject, check_size ignore]:
        cdef:
            int nin, nout, nargs
            int identity
            PyUFuncGenericFunction *functions
            void **data
            int ntypes
            int check_return
            char *name
            char *types
            char *doc
            void *ptr
            PyObject *obj
            PyObject *userloops

    cdef enum:
        PyUFunc_Zero
        PyUFunc_One
        PyUFunc_None
        UFUNC_ERR_IGNORE
        UFUNC_ERR_WARN
        UFUNC_ERR_RAISE
        UFUNC_ERR_CALL
        UFUNC_ERR_PRINT
        UFUNC_ERR_LOG
        UFUNC_MASK_DIVIDEBYZERO
        UFUNC_MASK_OVERFLOW
        UFUNC_MASK_UNDERFLOW
        UFUNC_MASK_INVALID
        UFUNC_SHIFT_DIVIDEBYZERO
        UFUNC_SHIFT_OVERFLOW
        UFUNC_SHIFT_UNDERFLOW
        UFUNC_SHIFT_INVALID
        UFUNC_FPE_DIVIDEBYZERO
        UFUNC_FPE_OVERFLOW
        UFUNC_FPE_UNDERFLOW
        UFUNC_FPE_INVALID
        UFUNC_ERR_DEFAULT
        UFUNC_ERR_DEFAULT2

    object PyUFunc_FromFuncAndData(PyUFuncGenericFunction *,
          void **, char *, int, int, int, int, char *, char *, int)
    int PyUFunc_RegisterLoopForType(ufunc, int,
                                    PyUFuncGenericFunction, int *, void *)
    void PyUFunc_f_f_As_d_d \
         (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_d_d \
         (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_f_f \
         (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_g_g \
         (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_F_F_As_D_D \
         (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_F_F \
         (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_D_D \
         (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_G_G \
         (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_O_O \
         (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_ff_f_As_dd_d \
         (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_ff_f \
         (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_dd_d \
         (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_gg_g \
         (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_FF_F_As_DD_D \
         (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_DD_D \
         (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_FF_F \
         (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_GG_G \
         (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_OO_O \
         (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_O_O_method \
         (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_OO_O_method \
         (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_On_Om \
         (char **, npy_intp *, npy_intp *, void *)
    int PyUFunc_GetPyValues \
        (char *, int *, int *, PyObject **)
    int PyUFunc_checkfperr \
           (int, PyObject *, int *)
    void PyUFunc_clearfperr()
    int PyUFunc_getfperr()
    int PyUFunc_handlefperr \
        (int, PyObject *, int, int *)
    int PyUFunc_ReplaceLoopBySignature \
        (ufunc, PyUFuncGenericFunction, int *, PyUFuncGenericFunction *)
    object PyUFunc_FromFuncAndDataAndSignature \
             (PyUFuncGenericFunction *, void **, char *, int, int, int,
              int, char *, char *, int, char *)

    int _import_umath() except -1

cdef inline void set_array_base(ndarray arr, object base):
    Py_INCREF(base) # important to do this before stealing the reference below!
    PyArray_SetBaseObject(arr, base)

cdef inline object get_array_base(ndarray arr):
    base = PyArray_BASE(arr)
    if base is NULL:
        return None
    return <object>base

# Versions of the import_* functions which are more suitable for
# Cython code.
cdef inline int import_array() except -1:
    try:
        __pyx_import_array()
    except Exception:
        raise ImportError("numpy.core.multiarray failed to import")

cdef inline int import_umath() except -1:
    try:
        _import_umath()
    except Exception:
        raise ImportError("numpy.core.umath failed to import")

cdef inline int import_ufunc() except -1:
    try:
        _import_umath()
    except Exception:
        raise ImportError("numpy.core.umath failed to import")


cdef inline bint is_timedelta64_object(object obj):
    """
    Cython equivalent of `isinstance(obj, np.timedelta64)`

    Parameters
    ----------
    obj : object

    Returns
    -------
    bool
    """
    return PyObject_TypeCheck(obj, &PyTimedeltaArrType_Type)


cdef inline bint is_datetime64_object(object obj):
    """
    Cython equivalent of `isinstance(obj, np.datetime64)`

    Parameters
    ----------
    obj : object

    Returns
    -------
    bool
    """
    return PyObject_TypeCheck(obj, &PyDatetimeArrType_Type)


cdef inline npy_datetime get_datetime64_value(object obj) nogil:
    """
    returns the int64 value underlying scalar numpy datetime64 object

    Note that to interpret this as a datetime, the corresponding unit is
    also needed.  That can be found using `get_datetime64_unit`.
    """
    return (<PyDatetimeScalarObject*>obj).obval


cdef inline npy_timedelta get_timedelta64_value(object obj) nogil:
    """
    returns the int64 value underlying scalar numpy timedelta64 object
    """
    return (<PyTimedeltaScalarObject*>obj).obval


cdef inline NPY_DATETIMEUNIT get_datetime64_unit(object obj) nogil:
    """
    returns the unit part of the dtype for a numpy datetime64 object.
    """
    return <NPY_DATETIMEUNIT>(<PyDatetimeScalarObject*>obj).obmeta.base
