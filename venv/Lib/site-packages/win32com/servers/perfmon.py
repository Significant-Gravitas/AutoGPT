"""A COM Server which exposes the NT Performance monitor in a very rudimentary way

Usage from VB:
	set ob = CreateObject("Python.PerfmonQuery")
	freeBytes = ob.Query("Memory", "Available Bytes")
"""
import pythoncom
import win32pdhutil
import winerror
from win32com.server import exception, register


class PerfMonQuery:
    _reg_verprogid_ = "Python.PerfmonQuery.1"
    _reg_progid_ = "Python.PerfmonQuery"
    _reg_desc_ = "Python Performance Monitor query object"
    _reg_clsid_ = "{64cef7a0-8ece-11d1-a65a-00aa00125a98}"
    _reg_class_spec_ = "win32com.servers.perfmon.PerfMonQuery"
    _public_methods_ = ["Query"]

    def Query(self, object, counter, instance=None, machine=None):
        try:
            return win32pdhutil.GetPerformanceAttributes(
                object, counter, instance, machine=machine
            )
        except win32pdhutil.error as exc:
            raise exception.Exception(desc=exc.strerror)
        except TypeError as desc:
            raise exception.Exception(desc=desc, scode=winerror.DISP_E_TYPEMISMATCH)


if __name__ == "__main__":
    print("Registering COM server...")
    register.UseCommandLine(PerfMonQuery)
