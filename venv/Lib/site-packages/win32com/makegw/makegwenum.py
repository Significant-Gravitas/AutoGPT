"""Utility file for generating PyIEnum support.

This is almost a 'template' file.  It simplay contains almost full
C++ source code for PyIEnum* support, and the Python code simply
substitutes the appropriate interface name.

This module is notmally not used directly - the @makegw@ module
automatically calls this.
"""
#
# INTERNAL FUNCTIONS
#
#
import string


def is_interface_enum(enumtype):
    return not (enumtype[0] in string.uppercase and enumtype[2] in string.uppercase)


def _write_enumifc_cpp(f, interface):
    enumtype = interface.name[5:]
    if is_interface_enum(enumtype):
        # Assume an interface.
        enum_interface = "I" + enumtype[:-1]
        converter = (
            "PyObject *ob = PyCom_PyObjectFromIUnknown(rgVar[i], IID_%(enum_interface)s, FALSE);"
            % locals()
        )
        arraydeclare = (
            "%(enum_interface)s **rgVar = new %(enum_interface)s *[celt];" % locals()
        )
    else:
        # Enum of a simple structure
        converter = (
            "PyObject *ob = PyCom_PyObjectFrom%(enumtype)s(&rgVar[i]);" % locals()
        )
        arraydeclare = "%(enumtype)s *rgVar = new %(enumtype)s[celt];" % locals()

    f.write(
        """
// ---------------------------------------------------
//
// Interface Implementation

PyIEnum%(enumtype)s::PyIEnum%(enumtype)s(IUnknown *pdisp):
	PyIUnknown(pdisp)
{
	ob_type = &type;
}

PyIEnum%(enumtype)s::~PyIEnum%(enumtype)s()
{
}

/* static */ IEnum%(enumtype)s *PyIEnum%(enumtype)s::GetI(PyObject *self)
{
	return (IEnum%(enumtype)s *)PyIUnknown::GetI(self);
}

// @pymethod object|PyIEnum%(enumtype)s|Next|Retrieves a specified number of items in the enumeration sequence.
PyObject *PyIEnum%(enumtype)s::Next(PyObject *self, PyObject *args)
{
	long celt = 1;
	// @pyparm int|num|1|Number of items to retrieve.
	if ( !PyArg_ParseTuple(args, "|l:Next", &celt) )
		return NULL;

	IEnum%(enumtype)s *pIE%(enumtype)s = GetI(self);
	if ( pIE%(enumtype)s == NULL )
		return NULL;

	%(arraydeclare)s
	if ( rgVar == NULL ) {
		PyErr_SetString(PyExc_MemoryError, "allocating result %(enumtype)ss");
		return NULL;
	}

	int i;
/*	for ( i = celt; i--; )
		// *** possibly init each structure element???
*/

	ULONG celtFetched = 0;
	PY_INTERFACE_PRECALL;
	HRESULT hr = pIE%(enumtype)s->Next(celt, rgVar, &celtFetched);
	PY_INTERFACE_POSTCALL;
	if (  HRESULT_CODE(hr) != ERROR_NO_MORE_ITEMS && FAILED(hr) )
	{
		delete [] rgVar;
		return PyCom_BuildPyException(hr,pIE%(enumtype)s, IID_IE%(enumtype)s);
	}

	PyObject *result = PyTuple_New(celtFetched);
	if ( result != NULL )
	{
		for ( i = celtFetched; i--; )
		{
			%(converter)s
			if ( ob == NULL )
			{
				Py_DECREF(result);
				result = NULL;
				break;
			}
			PyTuple_SET_ITEM(result, i, ob);
		}
	}

/*	for ( i = celtFetched; i--; )
		// *** possibly cleanup each structure element???
*/
	delete [] rgVar;
	return result;
}

// @pymethod |PyIEnum%(enumtype)s|Skip|Skips over the next specified elementes.
PyObject *PyIEnum%(enumtype)s::Skip(PyObject *self, PyObject *args)
{
	long celt;
	if ( !PyArg_ParseTuple(args, "l:Skip", &celt) )
		return NULL;

	IEnum%(enumtype)s *pIE%(enumtype)s = GetI(self);
	if ( pIE%(enumtype)s == NULL )
		return NULL;

	PY_INTERFACE_PRECALL;
	HRESULT hr = pIE%(enumtype)s->Skip(celt);
	PY_INTERFACE_POSTCALL;
	if ( FAILED(hr) )
		return PyCom_BuildPyException(hr, pIE%(enumtype)s, IID_IE%(enumtype)s);

	Py_INCREF(Py_None);
	return Py_None;
}

// @pymethod |PyIEnum%(enumtype)s|Reset|Resets the enumeration sequence to the beginning.
PyObject *PyIEnum%(enumtype)s::Reset(PyObject *self, PyObject *args)
{
	if ( !PyArg_ParseTuple(args, ":Reset") )
		return NULL;

	IEnum%(enumtype)s *pIE%(enumtype)s = GetI(self);
	if ( pIE%(enumtype)s == NULL )
		return NULL;

	PY_INTERFACE_PRECALL;
	HRESULT hr = pIE%(enumtype)s->Reset();
	PY_INTERFACE_POSTCALL;
	if ( FAILED(hr) )
		return PyCom_BuildPyException(hr, pIE%(enumtype)s, IID_IE%(enumtype)s);

	Py_INCREF(Py_None);
	return Py_None;
}

// @pymethod <o PyIEnum%(enumtype)s>|PyIEnum%(enumtype)s|Clone|Creates another enumerator that contains the same enumeration state as the current one
PyObject *PyIEnum%(enumtype)s::Clone(PyObject *self, PyObject *args)
{
	if ( !PyArg_ParseTuple(args, ":Clone") )
		return NULL;

	IEnum%(enumtype)s *pIE%(enumtype)s = GetI(self);
	if ( pIE%(enumtype)s == NULL )
		return NULL;

	IEnum%(enumtype)s *pClone;
	PY_INTERFACE_PRECALL;
	HRESULT hr = pIE%(enumtype)s->Clone(&pClone);
	PY_INTERFACE_POSTCALL;
	if ( FAILED(hr) )
		return PyCom_BuildPyException(hr, pIE%(enumtype)s, IID_IE%(enumtype)s);

	return PyCom_PyObjectFromIUnknown(pClone, IID_IEnum%(enumtype)s, FALSE);
}

// @object PyIEnum%(enumtype)s|A Python interface to IEnum%(enumtype)s
static struct PyMethodDef PyIEnum%(enumtype)s_methods[] =
{
	{ "Next", PyIEnum%(enumtype)s::Next, 1 },    // @pymeth Next|Retrieves a specified number of items in the enumeration sequence.
	{ "Skip", PyIEnum%(enumtype)s::Skip, 1 },	// @pymeth Skip|Skips over the next specified elementes.
	{ "Reset", PyIEnum%(enumtype)s::Reset, 1 },	// @pymeth Reset|Resets the enumeration sequence to the beginning.
	{ "Clone", PyIEnum%(enumtype)s::Clone, 1 },	// @pymeth Clone|Creates another enumerator that contains the same enumeration state as the current one.
	{ NULL }
};

PyComEnumTypeObject PyIEnum%(enumtype)s::type("PyIEnum%(enumtype)s",
		&PyIUnknown::type,
		sizeof(PyIEnum%(enumtype)s),
		PyIEnum%(enumtype)s_methods,
		GET_PYCOM_CTOR(PyIEnum%(enumtype)s));
"""
        % locals()
    )


def _write_enumgw_cpp(f, interface):
    enumtype = interface.name[5:]
    if is_interface_enum(enumtype):
        # Assume an interface.
        enum_interface = "I" + enumtype[:-1]
        converter = (
            "if ( !PyCom_InterfaceFromPyObject(ob, IID_%(enum_interface)s, (void **)&rgVar[i], FALSE) )"
            % locals()
        )
        argdeclare = "%(enum_interface)s __RPC_FAR * __RPC_FAR *rgVar" % locals()
    else:
        argdeclare = "%(enumtype)s __RPC_FAR *rgVar" % locals()
        converter = "if ( !PyCom_PyObjectAs%(enumtype)s(ob, &rgVar[i]) )" % locals()
    f.write(
        """
// ---------------------------------------------------
//
// Gateway Implementation

// Std delegation
STDMETHODIMP_(ULONG) PyGEnum%(enumtype)s::AddRef(void) {return PyGatewayBase::AddRef();}
STDMETHODIMP_(ULONG) PyGEnum%(enumtype)s::Release(void) {return PyGatewayBase::Release();}
STDMETHODIMP PyGEnum%(enumtype)s::QueryInterface(REFIID iid, void ** obj) {return PyGatewayBase::QueryInterface(iid, obj);}
STDMETHODIMP PyGEnum%(enumtype)s::GetTypeInfoCount(UINT FAR* pctInfo) {return PyGatewayBase::GetTypeInfoCount(pctInfo);}
STDMETHODIMP PyGEnum%(enumtype)s::GetTypeInfo(UINT itinfo, LCID lcid, ITypeInfo FAR* FAR* pptInfo) {return PyGatewayBase::GetTypeInfo(itinfo, lcid, pptInfo);}
STDMETHODIMP PyGEnum%(enumtype)s::GetIDsOfNames(REFIID refiid, OLECHAR FAR* FAR* rgszNames, UINT cNames, LCID lcid, DISPID FAR* rgdispid) {return PyGatewayBase::GetIDsOfNames( refiid, rgszNames, cNames, lcid, rgdispid);}
STDMETHODIMP PyGEnum%(enumtype)s::Invoke(DISPID dispid, REFIID riid, LCID lcid, WORD wFlags, DISPPARAMS FAR* params, VARIANT FAR* pVarResult, EXCEPINFO FAR* pexcepinfo, UINT FAR* puArgErr) {return PyGatewayBase::Invoke( dispid, riid, lcid, wFlags, params, pVarResult, pexcepinfo, puArgErr);}

STDMETHODIMP PyGEnum%(enumtype)s::Next( 
            /* [in] */ ULONG celt,
            /* [length_is][size_is][out] */ %(argdeclare)s,
            /* [out] */ ULONG __RPC_FAR *pCeltFetched)
{
	PY_GATEWAY_METHOD;
	PyObject *result;
	HRESULT hr = InvokeViaPolicy("Next", &result, "i", celt);
	if ( FAILED(hr) )
		return hr;

	if ( !PySequence_Check(result) )
		goto error;
	int len;
	len = PyObject_Length(result);
	if ( len == -1 )
		goto error;
	if ( len > (int)celt)
		len = celt;

	if ( pCeltFetched )
		*pCeltFetched = len;

	int i;
	for ( i = 0; i < len; ++i )
	{
		PyObject *ob = PySequence_GetItem(result, i);
		if ( ob == NULL )
			goto error;

		%(converter)s
		{
			Py_DECREF(result);
			return PyCom_SetCOMErrorFromPyException(IID_IEnum%(enumtype)s);
		}
	}

	Py_DECREF(result);

	return len < (int)celt ? S_FALSE : S_OK;

  error:
	PyErr_Clear();	// just in case
	Py_DECREF(result);
	return PyCom_HandleIEnumNoSequence(IID_IEnum%(enumtype)s);
}

STDMETHODIMP PyGEnum%(enumtype)s::Skip( 
            /* [in] */ ULONG celt)
{
	PY_GATEWAY_METHOD;
	return InvokeViaPolicy("Skip", NULL, "i", celt);
}

STDMETHODIMP PyGEnum%(enumtype)s::Reset(void)
{
	PY_GATEWAY_METHOD;
	return InvokeViaPolicy("Reset");
}

STDMETHODIMP PyGEnum%(enumtype)s::Clone( 
            /* [out] */ IEnum%(enumtype)s __RPC_FAR *__RPC_FAR *ppEnum)
{
	PY_GATEWAY_METHOD;
	PyObject * result;
	HRESULT hr = InvokeViaPolicy("Clone", &result);
	if ( FAILED(hr) )
		return hr;

	/*
	** Make sure we have the right kind of object: we should have some kind
	** of IUnknown subclass wrapped into a PyIUnknown instance.
	*/
	if ( !PyIBase::is_object(result, &PyIUnknown::type) )
	{
		/* the wrong kind of object was returned to us */
		Py_DECREF(result);
		return PyCom_SetCOMErrorFromSimple(E_FAIL, IID_IEnum%(enumtype)s);
	}

	/*
	** Get the IUnknown out of the thing. note that the Python ob maintains
	** a reference, so we don't have to explicitly AddRef() here.
	*/
	IUnknown *punk = ((PyIUnknown *)result)->m_obj;
	if ( !punk )
	{
		/* damn. the object was released. */
		Py_DECREF(result);
		return PyCom_SetCOMErrorFromSimple(E_FAIL, IID_IEnum%(enumtype)s);
	}

	/*
	** Get the interface we want. note it is returned with a refcount.
	** This QI is actually going to instantiate a PyGEnum%(enumtype)s.
	*/
	hr = punk->QueryInterface(IID_IEnum%(enumtype)s, (LPVOID *)ppEnum);

	/* done with the result; this DECREF is also for <punk> */
	Py_DECREF(result);

	return PyCom_CheckIEnumNextResult(hr, IID_IEnum%(enumtype)s);
}
"""
        % locals()
    )
