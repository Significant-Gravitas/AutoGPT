"""Python.Interpreter COM Server

  This module implements a very very simple COM server which
  exposes the Python interpreter.

  This is designed more as a demonstration than a full blown COM server.
  General functionality and Error handling are both limited.

  To use this object, ensure it is registered by running this module
  from Python.exe.  Then, from Visual Basic, use "CreateObject('Python.Interpreter')",
  and call its methods!
"""

import winerror
from win32com.server.exception import Exception


# Expose the Python interpreter.
class Interpreter:
    """The interpreter object exposed via COM"""

    _public_methods_ = ["Exec", "Eval"]
    # All registration stuff to support fully automatic register/unregister
    _reg_verprogid_ = "Python.Interpreter.2"
    _reg_progid_ = "Python.Interpreter"
    _reg_desc_ = "Python Interpreter"
    _reg_clsid_ = "{30BD3490-2632-11cf-AD5B-524153480001}"
    _reg_class_spec_ = "win32com.servers.interp.Interpreter"

    def __init__(self):
        self.dict = {}

    def Eval(self, exp):
        """Evaluate an expression."""
        if type(exp) != str:
            raise Exception(desc="Must be a string", scode=winerror.DISP_E_TYPEMISMATCH)

        return eval(str(exp), self.dict)

    def Exec(self, exp):
        """Execute a statement."""
        if type(exp) != str:
            raise Exception(desc="Must be a string", scode=winerror.DISP_E_TYPEMISMATCH)
        exec(str(exp), self.dict)


def Register():
    import win32com.server.register

    return win32com.server.register.UseCommandLine(Interpreter)


if __name__ == "__main__":
    print("Registering COM server...")
    Register()
