"""Utilities for makegw - Parse a header file to build an interface

 This module contains the core code for parsing a header file describing a
 COM interface, and building it into an "Interface" structure.

 Each Interface has methods, and each method has arguments.

 Each argument knows how to use Py_BuildValue or Py_ParseTuple to
 exchange itself with Python.
 
 See the @win32com.makegw@ module for information in building a COM
 interface
"""
import re
import traceback


class error_not_found(Exception):
    def __init__(self, msg="The requested item could not be found"):
        super(error_not_found, self).__init__(msg)


class error_not_supported(Exception):
    def __init__(self, msg="The required functionality is not supported"):
        super(error_not_supported, self).__init__(msg)


VERBOSE = 0
DEBUG = 0

## NOTE : For interfaces as params to work correctly, you must
## make sure any PythonCOM extensions which expose the interface are loaded
## before generating.


class ArgFormatter:
    """An instance for a specific type of argument.	 Knows how to convert itself"""

    def __init__(self, arg, builtinIndirection, declaredIndirection=0):
        # print 'init:', arg.name, builtinIndirection, declaredIndirection, arg.indirectionLevel
        self.arg = arg
        self.builtinIndirection = builtinIndirection
        self.declaredIndirection = declaredIndirection
        self.gatewayMode = 0

    def _IndirectPrefix(self, indirectionFrom, indirectionTo):
        """Given the indirection level I was declared at (0=Normal, 1=*, 2=**)
        return a string prefix so I can pass to a function with the
        required indirection (where the default is the indirection of the method's param.

        eg, assuming my arg has indirection level of 2, if this function was passed 1
        it would return "&", so that a variable declared with indirection of 1
        can be prefixed with this to turn it into the indirection level required of 2
        """
        dif = indirectionFrom - indirectionTo
        if dif == 0:
            return ""
        elif dif == -1:
            return "&"
        elif dif == 1:
            return "*"
        else:
            return "?? (%d)" % (dif,)
            raise error_not_supported("Can't indirect this far - please fix me :-)")

    def GetIndirectedArgName(self, indirectFrom, indirectionTo):
        # print 'get:',self.arg.name, indirectFrom,self._GetDeclaredIndirection() + self.builtinIndirection, indirectionTo, self.arg.indirectionLevel

        if indirectFrom is None:
            ### ACK! this does not account for [in][out] variables.
            ### when this method is called, we need to know which
            indirectFrom = self._GetDeclaredIndirection() + self.builtinIndirection

        return self._IndirectPrefix(indirectFrom, indirectionTo) + self.arg.name

    def GetBuildValueArg(self):
        "Get the argument to be passes to Py_BuildValue"
        return self.arg.name

    def GetParseTupleArg(self):
        "Get the argument to be passed to PyArg_ParseTuple"
        if self.gatewayMode:
            # use whatever they were declared with
            return self.GetIndirectedArgName(None, 1)
        # local declarations have just their builtin indirection
        return self.GetIndirectedArgName(self.builtinIndirection, 1)

    def GetInterfaceCppObjectInfo(self):
        """Provide information about the C++ object used.

        Simple variables (such as integers) can declare their type (eg an integer)
        and use it as the target of both PyArg_ParseTuple and the COM function itself.

        More complex types require a PyObject * declared as the target of PyArg_ParseTuple,
        then some conversion routine to the C++ object which is actually passed to COM.

        This method provides the name, and optionally the type of that C++ variable.
        If the type if provided, the caller will likely generate a variable declaration.
        The name must always be returned.

        Result is a tuple of (variableName, [DeclareType|None|""])
        """

        # the first return element is the variable to be passed as
        # 	 an argument to an interface method. the variable was
        # 	 declared with only its builtin indirection level. when
        # 	 we pass it, we'll need to pass in whatever amount of
        # 	 indirection was applied (plus the builtin amount)
        # the second return element is the variable declaration; it
        # 	 should simply be builtin indirection
        return self.GetIndirectedArgName(
            self.builtinIndirection, self.arg.indirectionLevel + self.builtinIndirection
        ), "%s %s" % (self.GetUnconstType(), self.arg.name)

    def GetInterfaceArgCleanup(self):
        "Return cleanup code for C++ args passed to the interface method."
        if DEBUG:
            return "/* GetInterfaceArgCleanup output goes here: %s */\n" % self.arg.name
        else:
            return ""

    def GetInterfaceArgCleanupGIL(self):
        """Return cleanup code for C++ args passed to the interface
        method that must be executed with the GIL held"""
        if DEBUG:
            return (
                "/* GetInterfaceArgCleanup (GIL held) output goes here: %s */\n"
                % self.arg.name
            )
        else:
            return ""

    def GetUnconstType(self):
        return self.arg.unc_type

    def SetGatewayMode(self):
        self.gatewayMode = 1

    def _GetDeclaredIndirection(self):
        return self.arg.indirectionLevel
        print("declared:", self.arg.name, self.gatewayMode)
        if self.gatewayMode:
            return self.arg.indirectionLevel
        else:
            return self.declaredIndirection

    def DeclareParseArgTupleInputConverter(self):
        "Declare the variable used as the PyArg_ParseTuple param for a gateway"
        # Only declare it??
        # if self.arg.indirectionLevel==0:
        # 	return "\t%s %s;\n" % (self.arg.type, self.arg.name)
        # else:
        if DEBUG:
            return (
                "/* Declare ParseArgTupleInputConverter goes here: %s */\n"
                % self.arg.name
            )
        else:
            return ""

    def GetParsePostCode(self):
        "Get a string of C++ code to be executed after (ie, to finalise) the PyArg_ParseTuple conversion"
        if DEBUG:
            return "/* GetParsePostCode code goes here: %s */\n" % self.arg.name
        else:
            return ""

    def GetBuildForInterfacePreCode(self):
        "Get a string of C++ code to be executed before (ie, to initialise) the Py_BuildValue conversion for Interfaces"
        if DEBUG:
            return "/* GetBuildForInterfacePreCode goes here: %s */\n" % self.arg.name
        else:
            return ""

    def GetBuildForGatewayPreCode(self):
        "Get a string of C++ code to be executed before (ie, to initialise) the Py_BuildValue conversion for Gateways"
        s = self.GetBuildForInterfacePreCode()  # Usually the same
        if DEBUG:
            if s[:4] == "/* G":
                s = "/* GetBuildForGatewayPreCode goes here: %s */\n" % self.arg.name
        return s

    def GetBuildForInterfacePostCode(self):
        "Get a string of C++ code to be executed after (ie, to finalise) the Py_BuildValue conversion for Interfaces"
        if DEBUG:
            return "/* GetBuildForInterfacePostCode goes here: %s */\n" % self.arg.name
        return ""

    def GetBuildForGatewayPostCode(self):
        "Get a string of C++ code to be executed after (ie, to finalise) the Py_BuildValue conversion for Gateways"
        s = self.GetBuildForInterfacePostCode()  # Usually the same
        if DEBUG:
            if s[:4] == "/* G":
                s = "/* GetBuildForGatewayPostCode goes here: %s */\n" % self.arg.name
        return s

    def GetAutoduckString(self):
        return "// @pyparm %s|%s||Description for %s" % (
            self._GetPythonTypeDesc(),
            self.arg.name,
            self.arg.name,
        )

    def _GetPythonTypeDesc(self):
        "Returns a string with the description of the type.	 Used for doco purposes"
        return None

    def NeedUSES_CONVERSION(self):
        "Determines if this arg forces a USES_CONVERSION macro"
        return 0


# Special formatter for floats since they're smaller than Python floats.
class ArgFormatterFloat(ArgFormatter):
    def GetFormatChar(self):
        return "f"

    def DeclareParseArgTupleInputConverter(self):
        # Declare a double variable
        return "\tdouble dbl%s;\n" % self.arg.name

    def GetParseTupleArg(self):
        return "&dbl" + self.arg.name

    def _GetPythonTypeDesc(self):
        return "float"

    def GetBuildValueArg(self):
        return "&dbl" + self.arg.name

    def GetBuildForInterfacePreCode(self):
        return "\tdbl" + self.arg.name + " = " + self.arg.name + ";\n"

    def GetBuildForGatewayPreCode(self):
        return (
            "\tdbl%s = " % self.arg.name
            + self._IndirectPrefix(self._GetDeclaredIndirection(), 0)
            + self.arg.name
            + ";\n"
        )

    def GetParsePostCode(self):
        s = "\t"
        if self.gatewayMode:
            s = s + self._IndirectPrefix(self._GetDeclaredIndirection(), 0)
        s = s + self.arg.name
        s = s + " = (float)dbl%s;\n" % self.arg.name
        return s


# Special formatter for Shorts because they're
# a different size than Python ints!
class ArgFormatterShort(ArgFormatter):
    def GetFormatChar(self):
        return "i"

    def DeclareParseArgTupleInputConverter(self):
        # Declare a double variable
        return "\tINT i%s;\n" % self.arg.name

    def GetParseTupleArg(self):
        return "&i" + self.arg.name

    def _GetPythonTypeDesc(self):
        return "int"

    def GetBuildValueArg(self):
        return "&i" + self.arg.name

    def GetBuildForInterfacePreCode(self):
        return "\ti" + self.arg.name + " = " + self.arg.name + ";\n"

    def GetBuildForGatewayPreCode(self):
        return (
            "\ti%s = " % self.arg.name
            + self._IndirectPrefix(self._GetDeclaredIndirection(), 0)
            + self.arg.name
            + ";\n"
        )

    def GetParsePostCode(self):
        s = "\t"
        if self.gatewayMode:
            s = s + self._IndirectPrefix(self._GetDeclaredIndirection(), 0)
        s = s + self.arg.name
        s = s + " = i%s;\n" % self.arg.name
        return s


# for types which are 64bits on AMD64 - eg, HWND
class ArgFormatterLONG_PTR(ArgFormatter):
    def GetFormatChar(self):
        return "O"

    def DeclareParseArgTupleInputConverter(self):
        # Declare a PyObject variable
        return "\tPyObject *ob%s;\n" % self.arg.name

    def GetParseTupleArg(self):
        return "&ob" + self.arg.name

    def _GetPythonTypeDesc(self):
        return "int/long"

    def GetBuildValueArg(self):
        return "ob" + self.arg.name

    def GetBuildForInterfacePostCode(self):
        return "\tPy_XDECREF(ob%s);\n" % self.arg.name

    def GetParsePostCode(self):
        return (
            "\tif (bPythonIsHappy && !PyWinLong_AsULONG_PTR(ob%s, (ULONG_PTR *)%s)) bPythonIsHappy = FALSE;\n"
            % (self.arg.name, self.GetIndirectedArgName(None, 2))
        )

    def GetBuildForInterfacePreCode(self):
        notdirected = self.GetIndirectedArgName(None, 1)
        return "\tob%s = PyWinObject_FromULONG_PTR(%s);\n" % (
            self.arg.name,
            notdirected,
        )

    def GetBuildForGatewayPostCode(self):
        return "\tPy_XDECREF(ob%s);\n" % self.arg.name


class ArgFormatterPythonCOM(ArgFormatter):
    """An arg formatter for types exposed in the PythonCOM module"""

    def GetFormatChar(self):
        return "O"

    # def GetInterfaceCppObjectInfo(self):
    # 	return ArgFormatter.GetInterfaceCppObjectInfo(self)[0], \
    # 		"%s %s%s" % (self.arg.unc_type, "*" * self._GetDeclaredIndirection(), self.arg.name)
    def DeclareParseArgTupleInputConverter(self):
        # Declare a PyObject variable
        return "\tPyObject *ob%s;\n" % self.arg.name

    def GetParseTupleArg(self):
        return "&ob" + self.arg.name

    def _GetPythonTypeDesc(self):
        return "<o Py%s>" % self.arg.type

    def GetBuildValueArg(self):
        return "ob" + self.arg.name

    def GetBuildForInterfacePostCode(self):
        return "\tPy_XDECREF(ob%s);\n" % self.arg.name


class ArgFormatterBSTR(ArgFormatterPythonCOM):
    def _GetPythonTypeDesc(self):
        return "<o unicode>"

    def GetParsePostCode(self):
        return (
            "\tif (bPythonIsHappy && !PyWinObject_AsBstr(ob%s, %s)) bPythonIsHappy = FALSE;\n"
            % (self.arg.name, self.GetIndirectedArgName(None, 2))
        )

    def GetBuildForInterfacePreCode(self):
        notdirected = self.GetIndirectedArgName(None, 1)
        return "\tob%s = MakeBstrToObj(%s);\n" % (self.arg.name, notdirected)

    def GetBuildForInterfacePostCode(self):
        return "\tSysFreeString(%s);\n" % (
            self.arg.name,
        ) + ArgFormatterPythonCOM.GetBuildForInterfacePostCode(self)

    def GetBuildForGatewayPostCode(self):
        return "\tPy_XDECREF(ob%s);\n" % self.arg.name


class ArgFormatterOLECHAR(ArgFormatterPythonCOM):
    def _GetPythonTypeDesc(self):
        return "<o unicode>"

    def GetUnconstType(self):
        if self.arg.type[:3] == "LPC":
            return self.arg.type[:2] + self.arg.type[3:]
        else:
            return self.arg.unc_type

    def GetParsePostCode(self):
        return (
            "\tif (bPythonIsHappy && !PyWinObject_AsBstr(ob%s, %s)) bPythonIsHappy = FALSE;\n"
            % (self.arg.name, self.GetIndirectedArgName(None, 2))
        )

    def GetInterfaceArgCleanup(self):
        return "\tSysFreeString(%s);\n" % self.GetIndirectedArgName(None, 1)

    def GetBuildForInterfacePreCode(self):
        # the variable was declared with just its builtin indirection
        notdirected = self.GetIndirectedArgName(self.builtinIndirection, 1)
        return "\tob%s = MakeOLECHARToObj(%s);\n" % (self.arg.name, notdirected)

    def GetBuildForInterfacePostCode(self):
        # memory returned into an OLECHAR should be freed
        return "\tCoTaskMemFree(%s);\n" % (
            self.arg.name,
        ) + ArgFormatterPythonCOM.GetBuildForInterfacePostCode(self)

    def GetBuildForGatewayPostCode(self):
        return "\tPy_XDECREF(ob%s);\n" % self.arg.name


class ArgFormatterTCHAR(ArgFormatterPythonCOM):
    def _GetPythonTypeDesc(self):
        return "string/<o unicode>"

    def GetUnconstType(self):
        if self.arg.type[:3] == "LPC":
            return self.arg.type[:2] + self.arg.type[3:]
        else:
            return self.arg.unc_type

    def GetParsePostCode(self):
        return (
            "\tif (bPythonIsHappy && !PyWinObject_AsTCHAR(ob%s, %s)) bPythonIsHappy = FALSE;\n"
            % (self.arg.name, self.GetIndirectedArgName(None, 2))
        )

    def GetInterfaceArgCleanup(self):
        return "\tPyWinObject_FreeTCHAR(%s);\n" % self.GetIndirectedArgName(None, 1)

    def GetBuildForInterfacePreCode(self):
        # the variable was declared with just its builtin indirection
        notdirected = self.GetIndirectedArgName(self.builtinIndirection, 1)
        return "\tob%s = PyWinObject_FromTCHAR(%s);\n" % (self.arg.name, notdirected)

    def GetBuildForInterfacePostCode(self):
        return "// ??? - TCHAR post code\n"

    def GetBuildForGatewayPostCode(self):
        return "\tPy_XDECREF(ob%s);\n" % self.arg.name


class ArgFormatterIID(ArgFormatterPythonCOM):
    def _GetPythonTypeDesc(self):
        return "<o PyIID>"

    def GetParsePostCode(self):
        return "\tif (!PyWinObject_AsIID(ob%s, &%s)) bPythonIsHappy = FALSE;\n" % (
            self.arg.name,
            self.arg.name,
        )

    def GetBuildForInterfacePreCode(self):
        # 		notdirected = self.GetIndirectedArgName(self.arg.indirectionLevel, 0)
        notdirected = self.GetIndirectedArgName(None, 0)
        return "\tob%s = PyWinObject_FromIID(%s);\n" % (self.arg.name, notdirected)

    def GetInterfaceCppObjectInfo(self):
        return self.arg.name, "IID %s" % (self.arg.name)


class ArgFormatterTime(ArgFormatterPythonCOM):
    def __init__(self, arg, builtinIndirection, declaredIndirection=0):
        # we don't want to declare LPSYSTEMTIME / LPFILETIME objects
        if arg.indirectionLevel == 0 and arg.unc_type[:2] == "LP":
            arg.unc_type = arg.unc_type[2:]
            # reduce the builtin and increment the declaration
            arg.indirectionLevel = arg.indirectionLevel + 1
            builtinIndirection = 0
        ArgFormatterPythonCOM.__init__(
            self, arg, builtinIndirection, declaredIndirection
        )

    def _GetPythonTypeDesc(self):
        return "<o PyDateTime>"

    def GetParsePostCode(self):
        # variable was declared with only the builtinIndirection
        ### NOTE: this is an [in] ... so use only builtin
        return (
            '\tif (!PyTime_Check(ob%s)) {\n\t\tPyErr_SetString(PyExc_TypeError, "The argument must be a PyTime object");\n\t\tbPythonIsHappy = FALSE;\n\t}\n\tif (!((PyTime *)ob%s)->GetTime(%s)) bPythonIsHappy = FALSE;\n'
            % (
                self.arg.name,
                self.arg.name,
                self.GetIndirectedArgName(self.builtinIndirection, 1),
            )
        )

    def GetBuildForInterfacePreCode(self):
        ### use just the builtinIndirection again...
        notdirected = self.GetIndirectedArgName(self.builtinIndirection, 0)
        return "\tob%s = new PyTime(%s);\n" % (self.arg.name, notdirected)

    def GetBuildForInterfacePostCode(self):
        ### hack to determine if we need to free stuff
        ret = ""
        if self.builtinIndirection + self.arg.indirectionLevel > 1:
            # memory returned into an OLECHAR should be freed
            ret = "\tCoTaskMemFree(%s);\n" % self.arg.name
        return ret + ArgFormatterPythonCOM.GetBuildForInterfacePostCode(self)


class ArgFormatterSTATSTG(ArgFormatterPythonCOM):
    def _GetPythonTypeDesc(self):
        return "<o STATSTG>"

    def GetParsePostCode(self):
        return (
            "\tif (!PyCom_PyObjectAsSTATSTG(ob%s, %s, 0/*flags*/)) bPythonIsHappy = FALSE;\n"
            % (self.arg.name, self.GetIndirectedArgName(None, 1))
        )

    def GetBuildForInterfacePreCode(self):
        notdirected = self.GetIndirectedArgName(None, 1)
        return (
            "\tob%s = PyCom_PyObjectFromSTATSTG(%s);\n\t// STATSTG doco says our responsibility to free\n\tif ((%s).pwcsName) CoTaskMemFree((%s).pwcsName);\n"
            % (
                self.arg.name,
                self.GetIndirectedArgName(None, 1),
                notdirected,
                notdirected,
            )
        )


class ArgFormatterGeneric(ArgFormatterPythonCOM):
    def _GetPythonTypeDesc(self):
        return "<o %s>" % self.arg.type

    def GetParsePostCode(self):
        return "\tif (!PyObject_As%s(ob%s, &%s) bPythonIsHappy = FALSE;\n" % (
            self.arg.type,
            self.arg.name,
            self.GetIndirectedArgName(None, 1),
        )

    def GetInterfaceArgCleanup(self):
        return "\tPyObject_Free%s(%s);\n" % (self.arg.type, self.arg.name)

    def GetBuildForInterfacePreCode(self):
        notdirected = self.GetIndirectedArgName(None, 1)
        return "\tob%s = PyObject_From%s(%s);\n" % (
            self.arg.name,
            self.arg.type,
            self.GetIndirectedArgName(None, 1),
        )


class ArgFormatterIDLIST(ArgFormatterPythonCOM):
    def _GetPythonTypeDesc(self):
        return "<o PyIDL>"

    def GetParsePostCode(self):
        return (
            "\tif (bPythonIsHappy && !PyObject_AsPIDL(ob%s, &%s)) bPythonIsHappy = FALSE;\n"
            % (self.arg.name, self.GetIndirectedArgName(None, 1))
        )

    def GetInterfaceArgCleanup(self):
        return "\tPyObject_FreePIDL(%s);\n" % (self.arg.name,)

    def GetBuildForInterfacePreCode(self):
        notdirected = self.GetIndirectedArgName(None, 1)
        return "\tob%s = PyObject_FromPIDL(%s);\n" % (
            self.arg.name,
            self.GetIndirectedArgName(None, 1),
        )


class ArgFormatterHANDLE(ArgFormatterPythonCOM):
    def _GetPythonTypeDesc(self):
        return "<o PyHANDLE>"

    def GetParsePostCode(self):
        return (
            "\tif (!PyWinObject_AsHANDLE(ob%s, &%s, FALSE) bPythonIsHappy = FALSE;\n"
            % (self.arg.name, self.GetIndirectedArgName(None, 1))
        )

    def GetBuildForInterfacePreCode(self):
        notdirected = self.GetIndirectedArgName(None, 1)
        return "\tob%s = PyWinObject_FromHANDLE(%s);\n" % (
            self.arg.name,
            self.GetIndirectedArgName(None, 0),
        )


class ArgFormatterLARGE_INTEGER(ArgFormatterPythonCOM):
    def GetKeyName(self):
        return "LARGE_INTEGER"

    def _GetPythonTypeDesc(self):
        return "<o %s>" % self.GetKeyName()

    def GetParsePostCode(self):
        return "\tif (!PyWinObject_As%s(ob%s, %s)) bPythonIsHappy = FALSE;\n" % (
            self.GetKeyName(),
            self.arg.name,
            self.GetIndirectedArgName(None, 1),
        )

    def GetBuildForInterfacePreCode(self):
        notdirected = self.GetIndirectedArgName(None, 0)
        return "\tob%s = PyWinObject_From%s(%s);\n" % (
            self.arg.name,
            self.GetKeyName(),
            notdirected,
        )


class ArgFormatterULARGE_INTEGER(ArgFormatterLARGE_INTEGER):
    def GetKeyName(self):
        return "ULARGE_INTEGER"


class ArgFormatterInterface(ArgFormatterPythonCOM):
    def GetInterfaceCppObjectInfo(self):
        return self.GetIndirectedArgName(1, self.arg.indirectionLevel), "%s * %s" % (
            self.GetUnconstType(),
            self.arg.name,
        )

    def GetParsePostCode(self):
        # This gets called for out params in gateway mode
        if self.gatewayMode:
            sArg = self.GetIndirectedArgName(None, 2)
        else:
            # vs. in params for interface mode.
            sArg = self.GetIndirectedArgName(1, 2)
        return (
            "\tif (bPythonIsHappy && !PyCom_InterfaceFromPyInstanceOrObject(ob%s, IID_%s, (void **)%s, TRUE /* bNoneOK */))\n\t\t bPythonIsHappy = FALSE;\n"
            % (self.arg.name, self.arg.type, sArg)
        )

    def GetBuildForInterfacePreCode(self):
        return "\tob%s = PyCom_PyObjectFromIUnknown(%s, IID_%s, FALSE);\n" % (
            self.arg.name,
            self.arg.name,
            self.arg.type,
        )

    def GetBuildForGatewayPreCode(self):
        sPrefix = self._IndirectPrefix(self._GetDeclaredIndirection(), 1)
        return "\tob%s = PyCom_PyObjectFromIUnknown(%s%s, IID_%s, TRUE);\n" % (
            self.arg.name,
            sPrefix,
            self.arg.name,
            self.arg.type,
        )

    def GetInterfaceArgCleanup(self):
        return "\tif (%s) %s->Release();\n" % (self.arg.name, self.arg.name)


class ArgFormatterVARIANT(ArgFormatterPythonCOM):
    def GetParsePostCode(self):
        return (
            "\tif ( !PyCom_VariantFromPyObject(ob%s, %s) )\n\t\tbPythonIsHappy = FALSE;\n"
            % (self.arg.name, self.GetIndirectedArgName(None, 1))
        )

    def GetBuildForGatewayPreCode(self):
        notdirected = self.GetIndirectedArgName(None, 1)
        return "\tob%s = PyCom_PyObjectFromVariant(%s);\n" % (
            self.arg.name,
            notdirected,
        )

    def GetBuildForGatewayPostCode(self):
        return "\tPy_XDECREF(ob%s);\n" % self.arg.name

        # Key :		, Python Type Description, ParseTuple format char


ConvertSimpleTypes = {
    "BOOL": ("BOOL", "int", "i"),
    "UINT": ("UINT", "int", "i"),
    "BYTE": ("BYTE", "int", "i"),
    "INT": ("INT", "int", "i"),
    "DWORD": ("DWORD", "int", "l"),
    "HRESULT": ("HRESULT", "int", "l"),
    "ULONG": ("ULONG", "int", "l"),
    "LONG": ("LONG", "int", "l"),
    "int": ("int", "int", "i"),
    "long": ("long", "int", "l"),
    "DISPID": ("DISPID", "long", "l"),
    "APPBREAKFLAGS": ("int", "int", "i"),
    "BREAKRESUMEACTION": ("int", "int", "i"),
    "ERRORRESUMEACTION": ("int", "int", "i"),
    "BREAKREASON": ("int", "int", "i"),
    "BREAKPOINT_STATE": ("int", "int", "i"),
    "BREAKRESUME_ACTION": ("int", "int", "i"),
    "SOURCE_TEXT_ATTR": ("int", "int", "i"),
    "TEXT_DOC_ATTR": ("int", "int", "i"),
    "QUERYOPTION": ("int", "int", "i"),
    "PARSEACTION": ("int", "int", "i"),
}


class ArgFormatterSimple(ArgFormatter):
    """An arg formatter for simple integer etc types"""

    def GetFormatChar(self):
        return ConvertSimpleTypes[self.arg.type][2]

    def _GetPythonTypeDesc(self):
        return ConvertSimpleTypes[self.arg.type][1]


AllConverters = {
    "const OLECHAR": (ArgFormatterOLECHAR, 0, 1),
    "WCHAR": (ArgFormatterOLECHAR, 0, 1),
    "OLECHAR": (ArgFormatterOLECHAR, 0, 1),
    "LPCOLESTR": (ArgFormatterOLECHAR, 1, 1),
    "LPOLESTR": (ArgFormatterOLECHAR, 1, 1),
    "LPCWSTR": (ArgFormatterOLECHAR, 1, 1),
    "LPWSTR": (ArgFormatterOLECHAR, 1, 1),
    "LPCSTR": (ArgFormatterOLECHAR, 1, 1),
    "LPTSTR": (ArgFormatterTCHAR, 1, 1),
    "LPCTSTR": (ArgFormatterTCHAR, 1, 1),
    "HANDLE": (ArgFormatterHANDLE, 0),
    "BSTR": (ArgFormatterBSTR, 1, 0),
    "const IID": (ArgFormatterIID, 0),
    "CLSID": (ArgFormatterIID, 0),
    "IID": (ArgFormatterIID, 0),
    "GUID": (ArgFormatterIID, 0),
    "const GUID": (ArgFormatterIID, 0),
    "const IID": (ArgFormatterIID, 0),
    "REFCLSID": (ArgFormatterIID, 0),
    "REFIID": (ArgFormatterIID, 0),
    "REFGUID": (ArgFormatterIID, 0),
    "const FILETIME": (ArgFormatterTime, 0),
    "const SYSTEMTIME": (ArgFormatterTime, 0),
    "const LPSYSTEMTIME": (ArgFormatterTime, 1, 1),
    "LPSYSTEMTIME": (ArgFormatterTime, 1, 1),
    "FILETIME": (ArgFormatterTime, 0),
    "SYSTEMTIME": (ArgFormatterTime, 0),
    "STATSTG": (ArgFormatterSTATSTG, 0),
    "LARGE_INTEGER": (ArgFormatterLARGE_INTEGER, 0),
    "ULARGE_INTEGER": (ArgFormatterULARGE_INTEGER, 0),
    "VARIANT": (ArgFormatterVARIANT, 0),
    "float": (ArgFormatterFloat, 0),
    "single": (ArgFormatterFloat, 0),
    "short": (ArgFormatterShort, 0),
    "WORD": (ArgFormatterShort, 0),
    "VARIANT_BOOL": (ArgFormatterShort, 0),
    "HWND": (ArgFormatterLONG_PTR, 1),
    "HMENU": (ArgFormatterLONG_PTR, 1),
    "HOLEMENU": (ArgFormatterLONG_PTR, 1),
    "HICON": (ArgFormatterLONG_PTR, 1),
    "HDC": (ArgFormatterLONG_PTR, 1),
    "LPARAM": (ArgFormatterLONG_PTR, 1),
    "WPARAM": (ArgFormatterLONG_PTR, 1),
    "LRESULT": (ArgFormatterLONG_PTR, 1),
    "UINT": (ArgFormatterShort, 0),
    "SVSIF": (ArgFormatterShort, 0),
    "Control": (ArgFormatterInterface, 0, 1),
    "DataObject": (ArgFormatterInterface, 0, 1),
    "_PropertyBag": (ArgFormatterInterface, 0, 1),
    "AsyncProp": (ArgFormatterInterface, 0, 1),
    "DataSource": (ArgFormatterInterface, 0, 1),
    "DataFormat": (ArgFormatterInterface, 0, 1),
    "void **": (ArgFormatterInterface, 2, 2),
    "ITEMIDLIST": (ArgFormatterIDLIST, 0, 0),
    "LPITEMIDLIST": (ArgFormatterIDLIST, 0, 1),
    "LPCITEMIDLIST": (ArgFormatterIDLIST, 0, 1),
    "const ITEMIDLIST": (ArgFormatterIDLIST, 0, 1),
}

# Auto-add all the simple types
for key in ConvertSimpleTypes.keys():
    AllConverters[key] = ArgFormatterSimple, 0


def make_arg_converter(arg):
    try:
        clz = AllConverters[arg.type][0]
        bin = AllConverters[arg.type][1]
        decl = 0
        if len(AllConverters[arg.type]) > 2:
            decl = AllConverters[arg.type][2]
        return clz(arg, bin, decl)
    except KeyError:
        if arg.type[0] == "I":
            return ArgFormatterInterface(arg, 0, 1)

        raise error_not_supported(
            "The type '%s' (%s) is unknown." % (arg.type, arg.name)
        )


#############################################################
#
# The instances that represent the args, methods and interface
class Argument:
    """A representation of an argument to a COM method

    This class contains information about a specific argument to a method.
    In addition, methods exist so that an argument knows how to convert itself
    to/from Python arguments.
    """

    # 									  in,out					  type			  name			 [	]
    # 								   --------------				--------	  ------------		------
    regex = re.compile(r"/\* \[([^\]]*.*?)] \*/[ \t](.*[* ]+)(\w+)(\[ *])?[\),]")

    def __init__(self, good_interface_names):
        self.good_interface_names = good_interface_names
        self.inout = self.name = self.type = None
        self.const = 0
        self.arrayDecl = 0

    def BuildFromFile(self, file):
        """Parse and build my data from a file

        Reads the next line in the file, and matches it as an argument
        description.  If not a valid argument line, an error_not_found exception
        is raised.
        """
        line = file.readline()
        mo = self.regex.search(line)
        if not mo:
            raise error_not_found
        self.name = mo.group(3)
        self.inout = mo.group(1).split("][")
        typ = mo.group(2).strip()
        self.raw_type = typ
        self.indirectionLevel = 0
        if mo.group(4):  # Has "[ ]" decl
            self.arrayDecl = 1
            try:
                pos = typ.rindex("__RPC_FAR")
                self.indirectionLevel = self.indirectionLevel + 1
                typ = typ[:pos].strip()
            except ValueError:
                pass

        typ = typ.replace("__RPC_FAR", "")
        while 1:
            try:
                pos = typ.rindex("*")
                self.indirectionLevel = self.indirectionLevel + 1
                typ = typ[:pos].strip()
            except ValueError:
                break
        self.type = typ
        if self.type[:6] == "const ":
            self.unc_type = self.type[6:]
        else:
            self.unc_type = self.type

        if VERBOSE:
            print(
                "	   Arg %s of type %s%s (%s)"
                % (self.name, self.type, "*" * self.indirectionLevel, self.inout)
            )

    def HasAttribute(self, typ):
        """Determines if the argument has the specific attribute.

        Argument attributes are specified in the header file, such as
        "[in][out][retval]" etc.  You can pass a specific string (eg "out")
        to find if this attribute was specified for the argument
        """
        return typ in self.inout

    def GetRawDeclaration(self):
        ret = "%s %s" % (self.raw_type, self.name)
        if self.arrayDecl:
            ret = ret + "[]"
        return ret


class Method:
    """A representation of a C++ method on a COM interface

    This class contains information about a specific method, as well as
    a list of all @Argument@s
    """

    # 										 options	 ret type callconv	 name
    # 								   ----------------- -------- -------- --------
    regex = re.compile(r"virtual (/\*.*?\*/ )?(.*?) (.*?) (.*?)\(\w?")

    def __init__(self, good_interface_names):
        self.good_interface_names = good_interface_names
        self.name = self.result = self.callconv = None
        self.args = []

    def BuildFromFile(self, file):
        """Parse and build my data from a file

        Reads the next line in the file, and matches it as a method
        description.  If not a valid method line, an error_not_found exception
        is raised.
        """
        line = file.readline()
        mo = self.regex.search(line)
        if not mo:
            raise error_not_found
        self.name = mo.group(4)
        self.result = mo.group(2)
        if self.result != "HRESULT":
            if self.result == "DWORD":  # DWORD is for old old stuff?
                print(
                    "Warning: Old style interface detected - compilation errors likely!"
                )
            else:
                print(
                    "Method %s - Only HRESULT return types are supported." % self.name
                )
            # 				raise error_not_supported,		if VERBOSE:
            print("	 Method %s %s(" % (self.result, self.name))
        while 1:
            arg = Argument(self.good_interface_names)
            try:
                arg.BuildFromFile(file)
                self.args.append(arg)
            except error_not_found:
                break


class Interface:
    """A representation of a C++ COM Interface

    This class contains information about a specific interface, as well as
    a list of all @Method@s
    """

    # 									  name				 base
    # 									 --------		   --------
    regex = re.compile("(interface|) ([^ ]*) : public (.*)$")

    def __init__(self, mo):
        self.methods = []
        self.name = mo.group(2)
        self.base = mo.group(3)
        if VERBOSE:
            print("Interface %s : public %s" % (self.name, self.base))

    def BuildMethods(self, file):
        """Build all sub-methods for this interface"""
        # skip the next 2 lines.
        file.readline()
        file.readline()
        while 1:
            try:
                method = Method([self.name])
                method.BuildFromFile(file)
                self.methods.append(method)
            except error_not_found:
                break


def find_interface(interfaceName, file):
    """Find and return an interface in a file

    Given an interface name and file, search for the specified interface.

    Upon return, the interface itself has been built,
    but not the methods.
    """
    interface = None
    line = file.readline()
    while line:
        mo = Interface.regex.search(line)
        if mo:
            name = mo.group(2)
            print(name)
            AllConverters[name] = (ArgFormatterInterface, 0, 1)
            if name == interfaceName:
                interface = Interface(mo)
                interface.BuildMethods(file)
        line = file.readline()
    if interface:
        return interface
    raise error_not_found


def parse_interface_info(interfaceName, file):
    """Find, parse and return an interface in a file

    Given an interface name and file, search for the specified interface.

    Upon return, the interface itself is fully built,
    """
    try:
        return find_interface(interfaceName, file)
    except re.error:
        traceback.print_exc()
        print("The interface could not be built, as the regular expression failed!")


def test():
    f = open("d:\\msdev\\include\\objidl.h")
    try:
        parse_interface_info("IPersistStream", f)
    finally:
        f.close()


def test_regex(r, text):
    res = r.search(text, 0)
    if res == -1:
        print("** Not found")
    else:
        print(
            "%d\n%s\n%s\n%s\n%s" % (res, r.group(1), r.group(2), r.group(3), r.group(4))
        )
