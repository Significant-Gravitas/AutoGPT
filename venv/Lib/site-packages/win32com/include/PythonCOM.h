/* PythonCOM.h

 Main header for Python COM support.

 This file is involved mainly with client side COM support for
 Python.

 Most COM work put together by Greg Stein and Mark Hammond, with a
 few others starting to come out of the closet.


 --------------------------------------------------------------------
 Thread State Rules
 ------------------
 These rules apply to PythonCOM in general, and not just to
 the client side.

 The rules are quite simple, but it is critical they be followed.
 In general, errors here will be picked up quite quickly, as Python
 will raise a Fatal Error.  However, the Release() issue in particular
 may keep a number of problems well hidden.

 Interfaces:
 -----------
 Before making ANY call out to COM, you MUST release the Python lock.
 This is true to ANY call whatsoever, including the COM call in question,
 but also any calls to "->Release();"

 This is normally achieved with the calls
 PY_INTERFACE_PRECALL and PY_INTERFACE_POSTCALL, which release
 and acquire the Python lock.

 Gateways:
 ---------
 Before doing anything related to Python, gateways MUST acquire the
 Python lock, and must release it before returning.

 This is normally achieved with PY_GATEWAY_METHOD at the top of a
 gateway method.  This macro resolves to a class, which automatically does
 the right thing.

 Release:
 --------
 As mentioned above for Interfaces, EVERY call to Release() must be done
 with the Python lock released.  This is expanded here.

 This is very important, but an error may not be noticed.  The problem will
 only be seen when the Release() is on a Python object and the Release() is the
 final one for the object.  In this case, the Python object will attempt to
 acquire the Python lock before destroying itself, and Python will raise a
 fatal error.

 In many many cases, you will not notice this error, but someday, someone will
 implement the other side in Python, and suddenly FatalErrors will start
 appearing.  Make sure you get this right.

 Eg, this code is correct:
   PY_INTERFACE_PRECALL;
   pSomeObj->SomeFunction(pSomeOtherObject);
   pSomeOtherObject->Release();
   PY_INTERFACE_POSTCALL;

 However, this code is WRONG, but will RARELY FAIL.
   PY_INTERFACE_PRECALL;
   pSomeObj->SomeFunction(pSomeOtherObject);
   PY_INTERFACE_POSTCALL;
   pSomeOtherObject->Release();
--------------------------------------------------------------------
*/
#ifndef __PYTHONCOM_H__
#define __PYTHONCOM_H__

// #define _DEBUG_LIFETIMES // Trace COM object lifetimes.

#ifdef FREEZE_PYTHONCOM
/* The pythoncom module is being included in a frozen .EXE/.DLL */
#define PYCOM_EXPORT
#else
#ifdef BUILD_PYTHONCOM
/* We are building pythoncomxx.dll */
#define PYCOM_EXPORT __declspec(dllexport)
#else
/* This module uses pythoncomxx.dll */
#define PYCOM_EXPORT __declspec(dllimport)
#ifndef _DEBUG
#pragma comment(lib, "pythoncom.lib")
#else
#pragma comment(lib, "pythoncom_d.lib")
#endif
#endif
#endif

#ifdef MS_WINCE
// List of interfaces not supported by CE.
#define NO_PYCOM_IDISPATCHEX
#define NO_PYCOM_IPROVIDECLASSINFO
#define NO_PYCOM_IENUMGUID
#define NO_PYCOM_IENUMCATEGORYINFO
#define NO_PYCOM_ICATINFORMATION
#define NO_PYCOM_ICATREGISTER
#define NO_PYCOM_ISERVICEPROVIDER
#define NO_PYCOM_IPROPERTYSTORAGE
#define NO_PYCOM_IPROPERTYSETSTORAGE
#define NO_PYCOM_ENUMSTATPROPSTG

#include "ocidl.h"
#include "oleauto.h"

#endif  // MS_WINCE

#ifdef __MINGW32__
// Special Mingw32 considerations.
#define NO_PYCOM_ENUMSTATPROPSTG
#define __try try
#define __except catch
#include <olectl.h>

#endif  // __MINGW32__

#include <PyWinTypes.h>  // Standard Win32 Types

#ifndef NO_PYCOM_IDISPATCHEX
#include <dispex.h>  // New header for IDispatchEx interface.
#endif               // NO_PYCOM_IDISPATCHEX

#if defined(MAINWIN)
// Mainwin seems to have 1/2 the VT_RECORD infrastructure in place
#if !defined(VT_RECORD)
#define VT_RECORD 36
#define V_RECORDINFO(X) ((X)->brecVal.pRecInfo)
#define V_RECORD(X) ((X)->brecVal.pvRecord)
#else
#pragma message(                                       \
    "MAINWIN appears to have grown correct VT_RECORD " \
    "support. Please update PythonCOM.h accordingly")
#endif  // VT_RECORD
#endif  // MAINWIN

class PyIUnknown;
// To make life interesting/complicated, I use C++ classes for
// all Python objects.  The main advantage is that I can derive
// a PyIDispatch object from a PyIUnknown, etc.  This provides a
// clean C++ interface, and "automatically" provides all base
// Python methods to "derived" Python types.
//
// Main disadvantage is that any extension DLLs will need to include
// these headers, and link with this .lib
//
// Base class for (most of) the type objects.

class PYCOM_EXPORT PyComTypeObject : public PyTypeObject {
   public:
    PyComTypeObject(const char *name, PyComTypeObject *pBaseType, Py_ssize_t typeSize, struct PyMethodDef *methodList,
                    PyIUnknown *(*thector)(IUnknown *));
    ~PyComTypeObject();

    // is the given object an interface type object? (e.g. PyIUnknown)
    static BOOL is_interface_type(PyObject *ob);

   public:
    PyIUnknown *(*ctor)(IUnknown *);
};

// A type used for interfaces that can automatically provide enumerators
// (ie, they themselves aren't enumerable, but do have a suitable default
// method that returns a PyIEnum object
class PYCOM_EXPORT PyComEnumProviderTypeObject : public PyComTypeObject {
   public:
    PyComEnumProviderTypeObject(const char *name, PyComTypeObject *pBaseType, Py_ssize_t typeSize,
                                struct PyMethodDef *methodList, PyIUnknown *(*thector)(IUnknown *),
                                const char *enum_method_name);
    static PyObject *iter(PyObject *self);
    const char *enum_method_name;
};

// A type used for PyIEnum interfaces
class PYCOM_EXPORT PyComEnumTypeObject : public PyComTypeObject {
   public:
    static PyObject *iter(PyObject *self);
    static PyObject *iternext(PyObject *self);
    PyComEnumTypeObject(const char *name, PyComTypeObject *pBaseType, Py_ssize_t typeSize, struct PyMethodDef *methodList,
                        PyIUnknown *(*thector)(IUnknown *));
};

// Very very base class - not COM specific - Should exist in the
// Python core somewhere, IMO.
class PYCOM_EXPORT PyIBase : public PyObject {
   public:
    // virtuals for Python support
    virtual PyObject *getattr(char *name);
    virtual int setattr(char *name, PyObject *v);
    virtual PyObject *repr();
    virtual int compare(PyObject *other)
    {
        if (this == other)
            return 0;
        if (this < other)
            return -1;
        return 1;
    }
    // These iter are a little special, in that returning NULL means
    // use the implementation in the type
    virtual PyObject *iter() { return NULL; }
    virtual PyObject *iternext() { return NULL; }

   protected:
    PyIBase();
    virtual ~PyIBase();

   public:
    static BOOL is_object(PyObject *, PyComTypeObject *which);
    BOOL is_object(PyComTypeObject *which);
    static void dealloc(PyObject *ob);
    static PyObject *repr(PyObject *ob);
    static PyObject *getattro(PyObject *self, PyObject *name);
    static int setattro(PyObject *op, PyObject *obname, PyObject *v);
    static int cmp(PyObject *ob1, PyObject *ob2);
    static PyObject *richcmp(PyObject *ob1, PyObject *ob2, int op);
};

/* Special Type objects */
extern PYCOM_EXPORT PyTypeObject PyOleEmptyType;        // equivalent to VT_EMPTY
extern PYCOM_EXPORT PyTypeObject PyOleMissingType;      // special Python handling.
extern PYCOM_EXPORT PyTypeObject PyOleArgNotFoundType;  // special VT_ERROR value
extern PYCOM_EXPORT PyTypeObject PyOleNothingType;      // special VT_ERROR value

// ALL of these set an appropriate Python error on bad return.

// Given a Python object that is a registered COM type, return a given
// interface pointer on its underlying object, with a new reference added.
PYCOM_EXPORT BOOL PyCom_InterfaceFromPyObject(PyObject *ob, REFIID iid, LPVOID *ppv, BOOL bNoneOK = TRUE);

// As above, but allows instance with "_oleobj_" attribute.
PYCOM_EXPORT BOOL PyCom_InterfaceFromPyInstanceOrObject(PyObject *ob, REFIID iid, LPVOID *ppv, BOOL bNoneOK = TRUE);

// Release an arbitary COM pointer.
// NOTE: the PRECALL/POSTCALL stuff is probably not strictly necessary
// since the PyGILSTATE stuff has been in place (and even then, it only
// mattered when it was the last Release() on a Python implemented object)
#define PYCOM_RELEASE(pUnk)        \
    {                              \
        if (pUnk) {                \
            PY_INTERFACE_PRECALL;  \
            (pUnk)->Release();     \
            PY_INTERFACE_POSTCALL; \
        }                          \
    }

// Given an IUnknown and an Interface ID, create and return an object
// of the appropriate type. eg IID_Unknown->PyIUnknown,
// IID_IDispatch->PyIDispatch, etc.
// Uses a map that external extension DLLs can populate with their IID/type.
// Under the principal of least surprise, this will return Py_None is punk is NULL.
//  Otherwise, a valid PyI*, but with NULL m_obj (and therefore totally useless)
//  object would be created.
// BOOL bAddRef indicates if a COM reference count should be added to the IUnknown.
//  This depends purely on the context in which it is called.  If the IUnknown is obtained
//  from a function that creates a new ref (eg, CoCreateInstance()) then you should use
//  FALSE.  If you receive the pointer as (eg) a param to a gateway function, then
//  you normally need to pass TRUE, as this is truly a new reference.
//  *** ALWAYS take the time to get this right. ***
PYCOM_EXPORT PyObject *PyCom_PyObjectFromIUnknown(IUnknown *punk, REFIID riid, BOOL bAddRef = FALSE);

// VARIANT <-> PyObject conversion utilities.
PYCOM_EXPORT BOOL PyCom_VariantFromPyObject(PyObject *obj, VARIANT *var);
PYCOM_EXPORT PyObject *PyCom_PyObjectFromVariant(const VARIANT *var);

// PROPVARIANT
PYCOM_EXPORT PyObject *PyObject_FromPROPVARIANT(PROPVARIANT *pVar);
PYCOM_EXPORT PyObject *PyObject_FromPROPVARIANTs(PROPVARIANT *pVars, ULONG cVars);
PYCOM_EXPORT BOOL PyObject_AsPROPVARIANT(PyObject *ob, PROPVARIANT *pVar);

// Other conversion helpers...
PYCOM_EXPORT PyObject *PyCom_PyObjectFromSTATSTG(STATSTG *pStat);
PYCOM_EXPORT BOOL PyCom_PyObjectAsSTATSTG(PyObject *ob, STATSTG *pStat, DWORD flags = 0);
PYCOM_EXPORT BOOL PyCom_SAFEARRAYFromPyObject(PyObject *obj, SAFEARRAY **ppSA, VARENUM vt = VT_VARIANT);
PYCOM_EXPORT PyObject *PyCom_PyObjectFromSAFEARRAY(SAFEARRAY *psa, VARENUM vt = VT_VARIANT);
#ifndef NO_PYCOM_STGOPTIONS
PYCOM_EXPORT BOOL PyCom_PyObjectAsSTGOPTIONS(PyObject *obstgoptions, STGOPTIONS **ppstgoptions, TmpWCHAR *tmpw_shelve);
#endif
PYCOM_EXPORT PyObject *PyCom_PyObjectFromSTATPROPSETSTG(STATPROPSETSTG *pStat);
PYCOM_EXPORT BOOL PyCom_PyObjectAsSTATPROPSETSTG(PyObject *, STATPROPSETSTG *);

// Currency support.
PYCOM_EXPORT PyObject *PyObject_FromCurrency(CURRENCY &cy);
PYCOM_EXPORT BOOL PyObject_AsCurrency(PyObject *ob, CURRENCY *pcy);

// OLEMENUGROUPWIDTHS are used by axcontrol, shell, etc
PYCOM_EXPORT BOOL PyObject_AsOLEMENUGROUPWIDTHS(PyObject *oblpMenuWidths, OLEMENUGROUPWIDTHS *pWidths);
PYCOM_EXPORT PyObject *PyObject_FromOLEMENUGROUPWIDTHS(const OLEMENUGROUPWIDTHS *pWidths);

/* Functions for Initializing COM, and also letting the core know about it!
 */
PYCOM_EXPORT HRESULT PyCom_CoInitializeEx(LPVOID reserved, DWORD dwInit);
PYCOM_EXPORT HRESULT PyCom_CoInitialize(LPVOID reserved);
PYCOM_EXPORT void PyCom_CoUninitialize();

///////////////////////////////////////////////////////////////////
// Error related functions

// Client related functions - generally called by interfaces before
// they return NULL back to Python to indicate the error.
// All these functions return NULL so interfaces can generally
// just "return PyCom_BuildPyException(hr, punk, IID_IWhatever)"

// Uses the HRESULT, and IErrorInfo interfaces if available to
// create and set a pythoncom.com_error.
PYCOM_EXPORT PyObject *PyCom_BuildPyException(HRESULT hr, IUnknown *pUnk = NULL, REFIID iid = IID_NULL);

// Uses the HRESULT and an EXCEPINFO structure to create and
// set a pythoncom.com_error.
PYCOM_EXPORT PyObject *PyCom_BuildPyExceptionFromEXCEPINFO(HRESULT hr, EXCEPINFO *pexcepInfo, UINT nArgErr = (UINT)-1);

// Sets a pythoncom.internal_error - no one should ever see these!
PYCOM_EXPORT PyObject *PyCom_BuildInternalPyException(char *msg);

// Log an error to a Python logger object if one can be found, or
// to stderr if no log available.
// If logProvider is not NULL, we will call a "_GetLogger_()" method on it.
// If logProvider is NULL, we attempt to fetch "win32com.logger".
// If they do not exist, return None, or raise an error fetching them
// (or even writing to them once fetched), the message still goes to stderr.
// NOTE: By default, win32com does *not* provide a logger, so default is that
// all errors are written to stdout.
// This will *not* write a record if a COM Server error is current.
PYCOM_EXPORT void PyCom_LoggerNonServerException(PyObject *logProvider, const WCHAR *fmt, ...);

// Write an error record, including exception.  This will write an error
// record even if a COM server error is current.
PYCOM_EXPORT void PyCom_LoggerException(PyObject *logProvider, const WCHAR *fmt, ...);

// Write a warning record - in general this does *not* mean a call failed, but
// still is something in the programmers control that they should change.
// XXX - if an exception is pending when this is called, the traceback will
// also be written.  This is undesirable and will be changed should this
// start being a problem.
PYCOM_EXPORT void PyCom_LoggerWarning(PyObject *logProvider, const WCHAR *fmt, ...);

// Server related error functions
// These are supplied so that any Python errors we detect can be
// converted into COM error information.  The HRESULT returned should
// be returned by the COM function, and these functions also set the
// IErrorInfo interfaces, so the caller can extract more detailed
// information about the Python exception.

// Set a COM exception, logging the exception if not an explicitly raised 'server' exception
PYCOM_EXPORT HRESULT PyCom_SetAndLogCOMErrorFromPyException(const char *methodName, REFIID riid /* = IID_NULL */);
PYCOM_EXPORT HRESULT PyCom_SetAndLogCOMErrorFromPyExceptionEx(PyObject *provider, const char *methodName,
                                                              REFIID riid /* = IID_NULL */);

// Used in gateways to SetErrorInfo() with a simple HRESULT, then return it.
// The description is generally only useful for debugging purposes,
// and if you are debugging via a server that supports IErrorInfo (like Python :-)
// NOTE: this function is usuable from outside the Python context
PYCOM_EXPORT HRESULT PyCom_SetCOMErrorFromSimple(HRESULT hr, REFIID riid = IID_NULL, const WCHAR *description = NULL);

// Used in gateways to check if an IEnum*'s Next() or Clone() method worked.
PYCOM_EXPORT HRESULT PyCom_CheckIEnumNextResult(HRESULT hr, REFIID riid);

// Used in gateways when an enumerator expected a sequence but didn't get it.
PYCOM_EXPORT HRESULT PyCom_HandleIEnumNoSequence(REFIID riid);

// Used in gateways to SetErrorInfo() the current Python exception, and
// (assuming not a server error explicitly raised) also logs an error
// to stdout/win32com.logger.
// NOTE: this function assumes GIL held
PYCOM_EXPORT HRESULT PyCom_SetCOMErrorFromPyException(REFIID riid = IID_NULL);

// A couple of EXCEPINFO helpers - could be private to IDispatch
// if it wasnt for the AXScript support (and ITypeInfo if we get around to that :-)
// These functions do not set any error states to either Python or
// COM - they simply convert to/from PyObjects and EXCEPINFOs

// Use the current Python exception to fill an EXCEPINFO structure.
PYCOM_EXPORT void PyCom_ExcepInfoFromPyException(EXCEPINFO *pExcepInfo);

// Fill in an EXCEPINFO structure from a Python instance or tuple object.
// (ie, similar to the above, except the Python exception object is specified,
// rather than using the "current"
PYCOM_EXPORT BOOL PyCom_ExcepInfoFromPyObject(PyObject *obExcepInfo, EXCEPINFO *pexcepInfo, HRESULT *phresult = NULL);

// Create a Python object holding the exception information.  The exception
// information is *not* freed by this function.  Python exceptions are
// raised and NULL is returned if an error occurs.
PYCOM_EXPORT PyObject *PyCom_PyObjectFromExcepInfo(const EXCEPINFO *pexcepInfo);

///////////////////////////////////////////////////////////////////
//
// External C++ helpers - these helpers are for other DLLs which
// may need similar functionality, but dont want to duplicate all

// This helper is for an application that has an IDispatch, and COM arguments
// and wants to call a Python function.  It is assumed the caller can map the IDispatch
// to a Python object, so the Python handler is passed.
// Args:
//   handler : A Python callable object.
//   dispparms : the COM arguments.
//   pVarResult : The variant for the return value of the Python call.
//   pexcepinfo : Exception info the helper may fill out.
//   puArgErr : Argument error the helper may fill out on exception
//   addnArgs : Any additional arguments to the Python function.  May be NULL.
// If addnArgs is NULL, then it is assumed the Python call should be native -
// ie, the COM args are packed as normal Python args to the call.
// If addnArgs is NOT NULL, it is assumed the Python function itself is
// a helper.  This Python function will be called with 2 arguments - both
// tuples - first one is the COM args, second is the addn args.
PYCOM_EXPORT BOOL PyCom_MakeOlePythonCall(PyObject *handler, DISPPARAMS FAR *params, VARIANT FAR *pVarResult,
                                          EXCEPINFO FAR *pexcepinfo, UINT FAR *puArgErr, PyObject *addnlArgs);

/////////////////////////////////////////////////////////////////////////////
// Various special purpose singletons
class PYCOM_EXPORT PyOleEmpty : public PyObject {
   public:
    PyOleEmpty();
};

class PYCOM_EXPORT PyOleMissing : public PyObject {
   public:
    PyOleMissing();
};

class PYCOM_EXPORT PyOleArgNotFound : public PyObject {
   public:
    PyOleArgNotFound();
};

class PYCOM_EXPORT PyOleNothing : public PyObject {
   public:
    PyOleNothing();
};

// We need to dynamically create C++ Python objects
// These helpers allow each type object to create it.
#define MAKE_PYCOM_CTOR(classname) \
    static PyIUnknown *PyObConstruct(IUnknown *pInitObj) { return new classname(pInitObj); }
#define MAKE_PYCOM_CTOR_ERRORINFO(classname, iid)                                                       \
    static PyIUnknown *PyObConstruct(IUnknown *pInitObj) { return new classname(pInitObj); }            \
    static PyObject *SetPythonCOMError(PyObject *self, HRESULT hr)                                      \
    {                                                                                                   \
        return PyCom_BuildPyException(hr, GetI(self), iid);                                             \
    }
#define GET_PYCOM_CTOR(classname) classname::PyObConstruct

// Macros that interfaces should use.  PY_INTERFACE_METHOD at the top of the method
// The other 2 wrap directly around the underlying method call.
#define PY_INTERFACE_METHOD
// Identical to Py_BEGIN_ALLOW_THREADS except no { !!!
#define PY_INTERFACE_PRECALL PyThreadState *_save = PyEval_SaveThread();
#define PY_INTERFACE_POSTCALL PyEval_RestoreThread(_save);

/////////////////////////////////////////////////////////////////////////////
// class PyIUnknown
class PYCOM_EXPORT PyIUnknown : public PyIBase {
   public:
    MAKE_PYCOM_CTOR(PyIUnknown);
    virtual PyObject *repr();
    virtual int compare(PyObject *other);

    static IUnknown *GetI(PyObject *self);
    IUnknown *m_obj;
    static char *szErrMsgObjectReleased;
    static void SafeRelease(PyIUnknown *ob);
    static PyComTypeObject type;

    // The Python methods
    static PyObject *QueryInterface(PyObject *self, PyObject *args);
    static PyObject *SafeRelease(PyObject *self, PyObject *args);

   protected:
    PyIUnknown(IUnknown *punk);
    ~PyIUnknown();
};

/////////////////////////////////////////////////////////////////////////////
// class PyIDispatch

class PYCOM_EXPORT PyIDispatch : public PyIUnknown {
   public:
    MAKE_PYCOM_CTOR(PyIDispatch);
    static IDispatch *GetI(PyObject *self);
    static PyComTypeObject type;

    // The Python methods
    static PyObject *Invoke(PyObject *self, PyObject *args);
    static PyObject *InvokeTypes(PyObject *self, PyObject *args);
    static PyObject *GetIDsOfNames(PyObject *self, PyObject *args);
    static PyObject *GetTypeInfo(PyObject *self, PyObject *args);
    static PyObject *GetTypeInfoCount(PyObject *self, PyObject *args);

   protected:
    PyIDispatch(IUnknown *pdisp);
    ~PyIDispatch();
};

#ifndef NO_PYCOM_IDISPATCHEX
/////////////////////////////////////////////////////////////////////////////
// class PyIDispatchEx

class PYCOM_EXPORT PyIDispatchEx : public PyIDispatch {
   public:
    MAKE_PYCOM_CTOR_ERRORINFO(PyIDispatchEx, IID_IDispatchEx);
    static IDispatchEx *GetI(PyObject *self);
    static PyComTypeObject type;

    // The Python methods
    static PyObject *GetDispID(PyObject *self, PyObject *args);
    static PyObject *InvokeEx(PyObject *self, PyObject *args);
    static PyObject *DeleteMemberByName(PyObject *self, PyObject *args);
    static PyObject *DeleteMemberByDispID(PyObject *self, PyObject *args);
    static PyObject *GetMemberProperties(PyObject *self, PyObject *args);
    static PyObject *GetMemberName(PyObject *self, PyObject *args);
    static PyObject *GetNextDispID(PyObject *self, PyObject *args);

   protected:
    PyIDispatchEx(IUnknown *pdisp);
    ~PyIDispatchEx();
};
#endif  // NO_PYCOM_IDISPATCHEX

/////////////////////////////////////////////////////////////////////////////
// class PyIClassFactory

class PYCOM_EXPORT PyIClassFactory : public PyIUnknown {
   public:
    MAKE_PYCOM_CTOR(PyIClassFactory);
    static IClassFactory *GetI(PyObject *self);
    static PyComTypeObject type;

    // The Python methods
    static PyObject *CreateInstance(PyObject *self, PyObject *args);
    static PyObject *LockServer(PyObject *self, PyObject *args);

   protected:
    PyIClassFactory(IUnknown *pdisp);
    ~PyIClassFactory();
};

#ifndef NO_PYCOM_IPROVIDECLASSINFO

/////////////////////////////////////////////////////////////////////////////
// class PyIProvideTypeInfo

class PYCOM_EXPORT PyIProvideClassInfo : public PyIUnknown {
   public:
    MAKE_PYCOM_CTOR(PyIProvideClassInfo);
    static IProvideClassInfo *GetI(PyObject *self);
    static PyComTypeObject type;

    // The Python methods
    static PyObject *GetClassInfo(PyObject *self, PyObject *args);

   protected:
    PyIProvideClassInfo(IUnknown *pdisp);
    ~PyIProvideClassInfo();
};

class PYCOM_EXPORT PyIProvideClassInfo2 : public PyIProvideClassInfo {
   public:
    MAKE_PYCOM_CTOR(PyIProvideClassInfo2);
    static IProvideClassInfo2 *GetI(PyObject *self);
    static PyComTypeObject type;

    // The Python methods
    static PyObject *GetGUID(PyObject *self, PyObject *args);

   protected:
    PyIProvideClassInfo2(IUnknown *pdisp);
    ~PyIProvideClassInfo2();
};
#endif  // NO_PYCOM_IPROVIDECLASSINFO

/////////////////////////////////////////////////////////////////////////////
// class PyITypeInfo
class PYCOM_EXPORT PyITypeInfo : public PyIUnknown {
   public:
    MAKE_PYCOM_CTOR(PyITypeInfo);
    static PyComTypeObject type;
    static ITypeInfo *GetI(PyObject *self);

    PyObject *GetContainingTypeLib();
    PyObject *GetDocumentation(MEMBERID);
    PyObject *GetRefTypeInfo(HREFTYPE href);
    PyObject *GetRefTypeOfImplType(int index);
    PyObject *GetFuncDesc(int pos);
    PyObject *GetIDsOfNames(OLECHAR FAR *FAR *, int);
    PyObject *GetNames(MEMBERID);
    PyObject *GetTypeAttr();
    PyObject *GetVarDesc(int pos);
    PyObject *GetImplTypeFlags(int index);
    PyObject *GetTypeComp();

   protected:
    PyITypeInfo(IUnknown *);
    ~PyITypeInfo();
};

/////////////////////////////////////////////////////////////////////////////
// class PyITypeComp
class PYCOM_EXPORT PyITypeComp : public PyIUnknown {
   public:
    MAKE_PYCOM_CTOR(PyITypeComp);
    static PyComTypeObject type;
    static ITypeComp *GetI(PyObject *self);

    PyObject *Bind(OLECHAR *szName, unsigned short wflags);
    PyObject *BindType(OLECHAR *szName);

   protected:
    PyITypeComp(IUnknown *);
    ~PyITypeComp();
};

/////////////////////////////////////////////////////////////////////////////
// class CPyTypeLib

class PYCOM_EXPORT PyITypeLib : public PyIUnknown {
   public:
    MAKE_PYCOM_CTOR(PyITypeLib);
    static PyComTypeObject type;
    static ITypeLib *GetI(PyObject *self);

    PyObject *GetLibAttr();
    PyObject *GetDocumentation(int pos);
    PyObject *GetTypeInfo(int pos);
    PyObject *GetTypeInfoCount();
    PyObject *GetTypeInfoOfGuid(REFGUID guid);
    PyObject *GetTypeInfoType(int pos);
    PyObject *GetTypeComp();

   protected:
    PyITypeLib(IUnknown *);
    ~PyITypeLib();
};

/////////////////////////////////////////////////////////////////////////////
// class PyIConnectionPoint

class PYCOM_EXPORT PyIConnectionPoint : public PyIUnknown {
   public:
    MAKE_PYCOM_CTOR_ERRORINFO(PyIConnectionPoint, IID_IConnectionPoint);
    static PyComTypeObject type;
    static IConnectionPoint *GetI(PyObject *self);

    static PyObject *GetConnectionInterface(PyObject *self, PyObject *args);
    static PyObject *GetConnectionPointContainer(PyObject *self, PyObject *args);
    static PyObject *Advise(PyObject *self, PyObject *args);
    static PyObject *Unadvise(PyObject *self, PyObject *args);
    static PyObject *EnumConnections(PyObject *self, PyObject *args);

   protected:
    PyIConnectionPoint(IUnknown *);
    ~PyIConnectionPoint();
};

class PYCOM_EXPORT PyIConnectionPointContainer : public PyIUnknown {
   public:
    MAKE_PYCOM_CTOR_ERRORINFO(PyIConnectionPointContainer, IID_IConnectionPointContainer);
    static PyComTypeObject type;
    static IConnectionPointContainer *GetI(PyObject *self);

    static PyObject *EnumConnectionPoints(PyObject *self, PyObject *args);
    static PyObject *FindConnectionPoint(PyObject *self, PyObject *args);

   protected:
    PyIConnectionPointContainer(IUnknown *);
    ~PyIConnectionPointContainer();
};

/////////////////////////////////////////////////////////////////////////////
// class PythonOleArgHelper
//
// A PythonOleArgHelper is used primarily to help out Python helpers
// which need to convert from a Python object when the specific OLE
// type is known - eg, when a TypeInfo is available.
//
// The type of conversion determines who owns what buffers etc.  I wish BYREF didnt exist :-)
typedef enum {
    // We dont know what sort of conversion it is yet.
    POAH_CONVERT_UNKNOWN,
    // A PyObject is given, we convert to a VARIANT, make the COM call, then BYREFs back to a PyObject
    // ie, this is typically a "normal" COM call, where Python initiates the call
    POAH_CONVERT_FROM_PYOBJECT,
    // A VARIANT is given, we convert to a PyObject, make the Python call, then BYREFs back to a VARIANT.
    // ie, this is typically handling a COM event, where COM itself initiates the call.
    POAH_CONVERT_FROM_VARIANT,
} POAH_CONVERT_DIRECTION;

class PYCOM_EXPORT PythonOleArgHelper {
   public:
    PythonOleArgHelper();
    ~PythonOleArgHelper();
    BOOL ParseTypeInformation(PyObject *reqdObjectTuple);

    // Using this call with reqdObject != NULL will check the existing
    // VT_ of the variant.  If not VT_EMPTY, then the result will be coerced to
    // that type.  This contrasts with PyCom_PyObjectToVariant which just
    // uses the Python type to determine the variant type.
    BOOL MakeObjToVariant(PyObject *obj, VARIANT *var, PyObject *reqdObjectTuple = NULL);
    PyObject *MakeVariantToObj(VARIANT *var);

    VARTYPE m_reqdType;
    BOOL m_bParsedTypeInfo;
    BOOL m_bIsOut;
    POAH_CONVERT_DIRECTION m_convertDirection;
    PyObject *m_pyVariant;  // if non-null, a win32com.client.VARIANT
    union {
        void *m_pValueHolder;
        short m_sBuf;
        long m_lBuf;
        LONGLONG m_llBuf;
        VARIANT_BOOL m_boolBuf;
        double m_dBuf;
        float m_fBuf;
        IDispatch *m_dispBuf;
        IUnknown *m_unkBuf;
        SAFEARRAY *m_arrayBuf;
        VARIANT *m_varBuf;
        DATE m_dateBuf;
        CY m_cyBuf;
    };
};

/////////////////////////////////////////////////////////////////////////////
// global functions and variables
PYCOM_EXPORT BOOL MakePythonArgumentTuples(PyObject **pArgs, PythonOleArgHelper **ppHelpers, PyObject **pNamedArgs,
                                           PythonOleArgHelper **ppNamedHelpers, DISPPARAMS FAR *params);

// Convert a Python object to a BSTR - allow embedded NULLs, None, etc.
PYCOM_EXPORT BOOL PyCom_BstrFromPyObject(PyObject *stringObject, BSTR *pResult, BOOL bNoneOK = FALSE);

// MakeBstrToObj - convert a BSTR into a Python string.
//
// ONLY USE THIS FOR TRUE BSTR's - Use the fn below for OLECHAR *'s.
// NOTE - does not use standard macros, so NULLs get through!
PYCOM_EXPORT PyObject *MakeBstrToObj(const BSTR bstr);

// Size info is available (eg, a fn returns a string and also fills in a size variable)
PYCOM_EXPORT PyObject *MakeOLECHARToObj(const OLECHAR *str, int numChars);

// No size info avail.
PYCOM_EXPORT PyObject *MakeOLECHARToObj(const OLECHAR *str);

PYCOM_EXPORT void PyCom_LogF(const WCHAR *fmt, ...);

// Generic conversion from python sequence to VT_VECTOR array
// Resulting array must be freed with CoTaskMemFree
template <typename arraytype>
BOOL SeqToVector(PyObject *ob, arraytype **pA, ULONG *pcount, BOOL (*converter)(PyObject *, arraytype *))
{
    TmpPyObject seq = PyWinSequence_Tuple(ob, pcount);
    if (seq == NULL)
        return FALSE;
    *pA = (arraytype *)CoTaskMemAlloc(*pcount * sizeof(arraytype));
    if (*pA == NULL) {
        PyErr_NoMemory();
        return FALSE;
    }
    for (ULONG i = 0; i < *pcount; i++) {
        PyObject *item = PyTuple_GET_ITEM((PyObject *)seq, i);
        if (!(*converter)(item, &(*pA)[i]))
            return FALSE;
    }
    return TRUE;
}

#endif  // __PYTHONCOM_H__
