# CallTips.py - An IDLE extension that provides "Call Tips" - ie, a floating window that
# displays parameter information as you open parens.

import inspect
import string
import sys
import traceback


class CallTips:
    menudefs = []

    keydefs = {
        "<<paren-open>>": ["<Key-parenleft>"],
        "<<paren-close>>": ["<Key-parenright>"],
        "<<check-calltip-cancel>>": ["<KeyRelease>"],
        "<<calltip-cancel>>": ["<ButtonPress>", "<Key-Escape>"],
    }

    windows_keydefs = {}

    unix_keydefs = {}

    def __init__(self, editwin):
        self.editwin = editwin
        self.text = editwin.text
        self.calltip = None
        if hasattr(self.text, "make_calltip_window"):
            self._make_calltip_window = self.text.make_calltip_window
        else:
            self._make_calltip_window = self._make_tk_calltip_window

    def close(self):
        self._make_calltip_window = None

    # Makes a Tk based calltip window.  Used by IDLE, but not Pythonwin.
    # See __init__ above for how this is used.
    def _make_tk_calltip_window(self):
        import CallTipWindow

        return CallTipWindow.CallTip(self.text)

    def _remove_calltip_window(self):
        if self.calltip:
            self.calltip.hidetip()
            self.calltip = None

    def paren_open_event(self, event):
        self._remove_calltip_window()
        arg_text = get_arg_text(self.get_object_at_cursor())
        if arg_text:
            self.calltip_start = self.text.index("insert")
            self.calltip = self._make_calltip_window()
            self.calltip.showtip(arg_text)
        return ""  # so the event is handled normally.

    def paren_close_event(self, event):
        # Now just hides, but later we should check if other
        # paren'd expressions remain open.
        self._remove_calltip_window()
        return ""  # so the event is handled normally.

    def check_calltip_cancel_event(self, event):
        if self.calltip:
            # If we have moved before the start of the calltip,
            # or off the calltip line, then cancel the tip.
            # (Later need to be smarter about multi-line, etc)
            if self.text.compare(
                "insert", "<=", self.calltip_start
            ) or self.text.compare("insert", ">", self.calltip_start + " lineend"):
                self._remove_calltip_window()
        return ""  # so the event is handled normally.

    def calltip_cancel_event(self, event):
        self._remove_calltip_window()
        return ""  # so the event is handled normally.

    def get_object_at_cursor(
        self,
        wordchars="._"
        + string.ascii_uppercase
        + string.ascii_lowercase
        + string.digits,
    ):
        # XXX - This needs to be moved to a better place
        # so the "." attribute lookup code can also use it.
        text = self.text
        chars = text.get("insert linestart", "insert")
        i = len(chars)
        while i and chars[i - 1] in wordchars:
            i = i - 1
        word = chars[i:]
        if word:
            # How is this for a hack!
            import __main__

            namespace = sys.modules.copy()
            namespace.update(__main__.__dict__)
            try:
                return eval(word, namespace)
            except:
                pass
        return None  # Can't find an object.


def _find_constructor(class_ob):
    # Given a class object, return a function object used for the
    # constructor (ie, __init__() ) or None if we can't find one.
    try:
        return class_ob.__init__
    except AttributeError:
        for base in class_ob.__bases__:
            rc = _find_constructor(base)
            if rc is not None:
                return rc
    return None


def get_arg_text(ob):
    # Get a string describing the arguments for the given object.
    argText = ""
    if ob is not None:
        if inspect.isclass(ob):
            # Look for the highest __init__ in the class chain.
            fob = _find_constructor(ob)
            if fob is None:
                fob = lambda: None
        else:
            fob = ob
        if inspect.isfunction(fob) or inspect.ismethod(fob):
            try:
                argText = str(inspect.signature(fob))
            except:
                print("Failed to format the args")
                traceback.print_exc()
        # See if we can use the docstring
        if hasattr(ob, "__doc__"):
            doc = ob.__doc__
            try:
                doc = doc.strip()
                pos = doc.find("\n")
            except AttributeError:
                ## New style classes may have __doc__ slot without actually
                ## having a string assigned to it
                pass
            else:
                if pos < 0 or pos > 70:
                    pos = 70
                if argText:
                    argText = argText + "\n"
                argText = argText + doc[:pos]

    return argText


#################################################
#
# Test code
#
if __name__ == "__main__":

    def t1():
        "()"

    def t2(a, b=None):
        "(a, b=None)"

    def t3(a, *args):
        "(a, *args)"

    def t4(*args):
        "(*args)"

    def t5(a, *args):
        "(a, *args)"

    def t6(a, b=None, *args, **kw):
        "(a, b=None, *args, **kw)"

    class TC:
        "(self, a=None, *b)"

        def __init__(self, a=None, *b):
            "(self, a=None, *b)"

        def t1(self):
            "(self)"

        def t2(self, a, b=None):
            "(self, a, b=None)"

        def t3(self, a, *args):
            "(self, a, *args)"

        def t4(self, *args):
            "(self, *args)"

        def t5(self, a, *args):
            "(self, a, *args)"

        def t6(self, a, b=None, *args, **kw):
            "(self, a, b=None, *args, **kw)"

    def test(tests):
        failed = []
        for t in tests:
            expected = t.__doc__ + "\n" + t.__doc__
            if get_arg_text(t) != expected:
                failed.append(t)
                print(
                    "%s - expected %s, but got %s"
                    % (t, repr(expected), repr(get_arg_text(t)))
                )
        print("%d of %d tests failed" % (len(failed), len(tests)))

    tc = TC()
    tests = t1, t2, t3, t4, t5, t6, TC, tc.t1, tc.t2, tc.t3, tc.t4, tc.t5, tc.t6

    test(tests)
