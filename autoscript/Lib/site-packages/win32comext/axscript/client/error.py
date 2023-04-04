"""Exception and error handling.

 This contains the core exceptions that the implementations should raise
 as well as the IActiveScriptError interface code.
 
"""

import re
import sys
import traceback

import pythoncom
import win32com.server.exception
import win32com.server.util
import winerror
from win32com.axscript import axscript

debugging = 0


def FormatForAX(text):
    """Format a string suitable for an AX Host"""
    # Replace all " with ', so it works OK in HTML (ie, ASP)
    return ExpandTabs(AddCR(text))


def ExpandTabs(text):
    return re.sub("\t", "    ", text)


def AddCR(text):
    return re.sub("\n", "\r\n", text)


class IActiveScriptError:
    """An implementation of IActiveScriptError

    The ActiveX Scripting host calls this client whenever we report
    an exception to it.  This interface provides the exception details
    for the host to report to the user.
    """

    _com_interfaces_ = [axscript.IID_IActiveScriptError]
    _public_methods_ = ["GetSourceLineText", "GetSourcePosition", "GetExceptionInfo"]

    def _query_interface_(self, iid):
        print("IActiveScriptError QI - unknown IID", iid)
        return 0

    def _SetExceptionInfo(self, exc):
        self.exception = exc

    def GetSourceLineText(self):
        return self.exception.linetext

    def GetSourcePosition(self):
        ctx = self.exception.sourceContext
        # Zero based in the debugger (but our columns are too!)
        return (
            ctx,
            self.exception.lineno + self.exception.startLineNo - 1,
            self.exception.colno,
        )

    def GetExceptionInfo(self):
        return self.exception


class AXScriptException(win32com.server.exception.COMException):
    """A class used as a COM exception.

    Note this has attributes which conform to the standard attributes
    for COM exceptions, plus a few others specific to our IActiveScriptError
    object.
    """

    def __init__(self, site, codeBlock, exc_type, exc_value, exc_traceback):
        # set properties base class shares via base ctor...
        win32com.server.exception.COMException.__init__(
            self,
            description="Unknown Exception",
            scode=winerror.DISP_E_EXCEPTION,
            source="Python ActiveX Scripting Engine",
        )

        # And my other values...
        if codeBlock is None:
            self.sourceContext = 0
            self.startLineNo = 0
        else:
            self.sourceContext = codeBlock.sourceContextCookie
            self.startLineNo = codeBlock.startLineNumber
        self.linetext = ""

        self.__BuildFromException(site, exc_type, exc_value, exc_traceback)

    def __BuildFromException(self, site, type, value, tb):
        if debugging:
            import linecache

            linecache.clearcache()
        try:
            if issubclass(type, SyntaxError):
                self._BuildFromSyntaxError(site, value, tb)
            else:
                self._BuildFromOther(site, type, value, tb)
        except:  # Error extracting traceback info!!!
            traceback.print_exc()
            # re-raise.
            raise

    def _BuildFromSyntaxError(self, site, exc, tb):
        value = exc.args
        # All syntax errors should have a message as element 0
        try:
            msg = value[0]
        except:
            msg = "Unknown Error (%s)" % (value,)
        try:
            (filename, lineno, offset, line) = value[1]
            # Some of these may be None, which upsets us!
            if offset is None:
                offset = 0
            if line is None:
                line = ""
        except:
            msg = "Unknown"
            lineno = 0
            offset = 0
            line = "Unknown"
        self.description = FormatForAX(msg)
        self.lineno = lineno
        self.colno = offset - 1
        self.linetext = ExpandTabs(line.rstrip())

    def _BuildFromOther(self, site, exc_type, value, tb):
        self.colno = -1
        self.lineno = 0
        if debugging:  # Full traceback if debugging.
            list = traceback.format_exception(exc_type, value, tb)
            self.description = ExpandTabs("".join(list))
            return
        # Run down the traceback list, looking for the first "<Script..>"
        # Hide traceback above this.  In addition, keep going down
        # looking for a "_*_" attribute, and below hide these also.
        hide_names = [
            "r_import",
            "r_reload",
            "r_open",
        ]  # hide from these functions down in the traceback.
        depth = None
        tb_top = tb
        while tb_top:
            filename, lineno, name, line = self.ExtractTracebackInfo(tb_top, site)
            if filename[:7] == "<Script":
                break
            tb_top = tb_top.tb_next
        format_items = []
        if tb_top:  # found one.
            depth = 0
            tb_look = tb_top
            # Look down for our bottom
            while tb_look:
                filename, lineno, name, line = self.ExtractTracebackInfo(tb_look, site)
                if name in hide_names:
                    break
                # We can report a line-number, but not a filename.  Therefore,
                # we return the last line-number we find in one of our script
                # blocks.
                if filename.startswith("<Script"):
                    self.lineno = lineno
                    self.linetext = line
                format_items.append((filename, lineno, name, line))
                depth = depth + 1
                tb_look = tb_look.tb_next
        else:
            depth = None
            tb_top = tb

        bits = ["Traceback (most recent call last):\n"]
        bits.extend(traceback.format_list(format_items))
        if exc_type == pythoncom.com_error:
            desc = "%s (0x%x)" % (value.strerror, value.hresult)
            if (
                value.hresult == winerror.DISP_E_EXCEPTION
                and value.excepinfo
                and value.excepinfo[2]
            ):
                desc = value.excepinfo[2]
            bits.append("COM Error: " + desc)
        else:
            bits.extend(traceback.format_exception_only(exc_type, value))

        # XXX - this utf8 encoding seems bogus.  From well before py3k,
        # we had the comment:
        # > all items in the list are utf8 courtesy of Python magically
        # > converting unicode to utf8 before compilation.
        # but that is likely just confusion from early unicode days;
        # Python isn't doing it, pywin32 probably was, so 'mbcs' would
        # be the default encoding.  We should never hit this these days
        # anyway, but on py3k, we *never* will, and str objects there
        # don't have a decode method...
        if sys.version_info < (3,):
            for i in range(len(bits)):
                if type(bits[i]) is str:
                    # assert type(bits[i]) is str, type(bits[i])
                    bits[i] = bits[i].decode("utf8")

        self.description = ExpandTabs("".join(bits))
        # Clear tracebacks etc.
        tb = tb_top = tb_look = None

    def ExtractTracebackInfo(self, tb, site):
        import linecache

        f = tb.tb_frame
        lineno = tb.tb_lineno
        co = f.f_code
        filename = co.co_filename
        name = co.co_name
        line = linecache.getline(filename, lineno)
        if not line:
            try:
                codeBlock = site.scriptCodeBlocks[filename]
            except KeyError:
                codeBlock = None
            if codeBlock:
                # Note: 'line' will now be unicode.
                line = codeBlock.GetLineNo(lineno)
        if line:
            line = line.strip()
        else:
            line = None
        return filename, lineno, name, line

    def __repr__(self):
        return "AXScriptException Object with description:" + self.description


def ProcessAXScriptException(scriptingSite, debugManager, exceptionInstance):
    """General function to handle any exception in AX code

    This function creates an instance of our IActiveScriptError interface, and
    gives it to the host, along with out exception class.  The host will
    likely call back on the IActiveScriptError interface to get the source text
    and other information not normally in COM exceptions.
    """
    # 	traceback.print_exc()
    instance = IActiveScriptError()
    instance._SetExceptionInfo(exceptionInstance)
    gateway = win32com.server.util.wrap(instance, axscript.IID_IActiveScriptError)
    if debugManager:
        fCallOnError = debugManager.HandleRuntimeError()
        if not fCallOnError:
            return None

    try:
        result = scriptingSite.OnScriptError(gateway)
    except pythoncom.com_error as details:
        print("**OnScriptError failed:", details)
        print("Exception description:'%s'" % (repr(exceptionInstance.description)))
        print("Exception text:'%s'" % (repr(exceptionInstance.linetext)))
        result = winerror.S_FALSE

    if result == winerror.S_OK:
        # If the above  returns NOERROR, it is assumed the error has been
        # correctly registered and the value SCRIPT_E_REPORTED is returned.
        ret = win32com.server.exception.COMException(scode=axscript.SCRIPT_E_REPORTED)
        return ret
    else:
        # The error is taken to be unreported and is propagated up the call stack
        # via the IDispatch::Invoke's EXCEPINFO parameter (hr returned is DISP_E_EXCEPTION.
        return exceptionInstance
