"""Support for ActiveX control hosting in Pythonwin.
"""
import win32ui
import win32uiole

from . import window

# XXX - we are still "classic style" classes in py2x, so we need can't yet
# use 'type()' everywhere - revisit soon, as py2x will move to new-style too...
try:
    from types import ClassType as new_type
except ImportError:
    new_type = type  # py3k


class Control(window.Wnd):
    """An ActiveX control base class.  A new class must be derived from both
    this class and the Events class.  See the demos for more details.
    """

    def __init__(self):
        self.__dict__["_dispobj_"] = None
        window.Wnd.__init__(self)

    def _GetControlCLSID(self):
        return self.CLSID

    def _GetDispatchClass(self):
        return self.default_interface

    def _GetEventMap(self):
        return self.default_source._dispid_to_func_

    def CreateControl(self, windowTitle, style, rect, parent, id, lic_string=None):
        clsid = str(self._GetControlCLSID())
        self.__dict__["_obj_"] = win32ui.CreateControl(
            clsid, windowTitle, style, rect, parent, id, None, False, lic_string
        )
        klass = self._GetDispatchClass()
        dispobj = klass(win32uiole.GetIDispatchForWindow(self._obj_))
        self.HookOleEvents()
        self.__dict__["_dispobj_"] = dispobj

    def HookOleEvents(self):
        dict = self._GetEventMap()
        for dispid, methodName in dict.items():
            if hasattr(self, methodName):
                self._obj_.HookOleEvent(getattr(self, methodName), dispid)

    def __getattr__(self, attr):
        # Delegate attributes to the windows and the Dispatch object for this class
        try:
            return window.Wnd.__getattr__(self, attr)
        except AttributeError:
            pass
        return getattr(self._dispobj_, attr)

    def __setattr__(self, attr, value):
        if hasattr(self.__dict__, attr):
            self.__dict__[attr] = value
            return
        try:
            if self._dispobj_:
                self._dispobj_.__setattr__(attr, value)
                return
        except AttributeError:
            pass
        self.__dict__[attr] = value


def MakeControlClass(controlClass, name=None):
    """Given a CoClass in a generated .py file, this function will return a Class
    object which can be used as an OCX control.

    This function is used when you do not want to handle any events from the OCX
    control.  If you need events, then you should derive a class from both the
    activex.Control class and the CoClass
    """
    if name is None:
        name = controlClass.__name__
    return new_type("OCX" + name, (Control, controlClass), {})


def MakeControlInstance(controlClass, name=None):
    """As for MakeControlClass(), but returns an instance of the class."""
    return MakeControlClass(controlClass, name)()
