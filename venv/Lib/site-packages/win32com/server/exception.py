"""Exception Handling

 Exceptions

	 To better support COM exceptions, the framework allows for an instance to be
	 raised.  This instance may have a certain number of known attributes, which are
	 translated into COM exception details.
	
	 This means, for example, that Python could raise a COM exception that includes details
	 on a Help file and location, and a description for the user.
	
	 This module provides a class which provides the necessary attributes.

"""
import sys

import pythoncom


# Note that we derive from com_error, which derives from exceptions.Exception
# Also note that we dont support "self.args", as we dont support tuple-unpacking
class COMException(pythoncom.com_error):
    """An Exception object that is understood by the framework.

    If the framework is presented with an exception of type class,
    it looks for certain known attributes on this class to provide rich
    error information to the caller.

    It should be noted that the framework supports providing this error
    information via COM Exceptions, or via the ISupportErrorInfo interface.

    By using this class, you automatically provide rich error information to the
    server.
    """

    def __init__(
        self,
        description=None,
        scode=None,
        source=None,
        helpfile=None,
        helpContext=None,
        desc=None,
        hresult=None,
    ):
        """Initialize an exception
        **Params**

        description -- A string description for the exception.
        scode -- An integer scode to be returned to the server, if necessary.
        The pythoncom framework defaults this to be DISP_E_EXCEPTION if not specified otherwise.
        source -- A string which identifies the source of the error.
        helpfile -- A string which points to a help file which contains details on the error.
        helpContext -- An integer context in the help file.
        desc -- A short-cut for description.
        hresult -- A short-cut for scode.
        """

        # convert a WIN32 error into an HRESULT
        scode = scode or hresult
        if scode and scode != 1:  # We dont want S_FALSE mapped!
            if scode >= -32768 and scode < 32768:
                # this is HRESULT_FROM_WIN32()
                scode = -2147024896 | (scode & 0x0000FFFF)
        self.scode = scode

        self.description = description or desc
        if scode == 1 and not self.description:
            self.description = "S_FALSE"
        elif scode and not self.description:
            self.description = pythoncom.GetScodeString(scode)

        self.source = source
        self.helpfile = helpfile
        self.helpcontext = helpContext

        # todo - fill in the exception value
        pythoncom.com_error.__init__(self, scode, self.description, None, -1)

    def __repr__(self):
        return "<COM Exception - scode=%s, desc=%s>" % (self.scode, self.description)


# Old name for the COMException class.
# Do NOT use the name Exception, as it is now a built-in
# COMException is the new, official name.
Exception = COMException


def IsCOMException(t=None):
    if t is None:
        t = sys.exc_info()[0]
    try:
        return issubclass(t, pythoncom.com_error)
    except TypeError:  # 1.5 in -X mode?
        return t is pythoncon.com_error


def IsCOMServerException(t=None):
    if t is None:
        t = sys.exc_info()[0]
    try:
        return issubclass(t, COMException)
    except TypeError:  # String exception
        return 0
