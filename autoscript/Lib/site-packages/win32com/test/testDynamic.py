# Test dynamic policy, and running object table.

import pythoncom
import winerror
from win32com.server.exception import Exception

error = "testDynamic error"

iid = pythoncom.MakeIID("{b48969a0-784b-11d0-ae71-d23f56000000}")


class VeryPermissive:
    def _dynamic_(self, name, lcid, wFlags, args):
        if wFlags & pythoncom.DISPATCH_METHOD:
            return getattr(self, name)(*args)

        if wFlags & pythoncom.DISPATCH_PROPERTYGET:
            try:
                # to avoid problems with byref param handling, tuple results are converted to lists.
                ret = self.__dict__[name]
                if type(ret) == type(()):
                    ret = list(ret)
                return ret
            except KeyError:  # Probably a method request.
                raise Exception(scode=winerror.DISP_E_MEMBERNOTFOUND)

        if wFlags & (
            pythoncom.DISPATCH_PROPERTYPUT | pythoncom.DISPATCH_PROPERTYPUTREF
        ):
            setattr(self, name, args[0])
            return

        raise Exception(scode=winerror.E_INVALIDARG, desc="invalid wFlags")

    def write(self, *args):
        if len(args) == 0:
            raise Exception(
                scode=winerror.DISP_E_BADPARAMCOUNT
            )  # Probably call as PROPGET.

        for arg in args[:-1]:
            print(str(arg), end=" ")
        print(str(args[-1]))


def Test():
    import win32com.server.policy
    import win32com.server.util

    #       import win32dbg;win32dbg.brk()
    ob = win32com.server.util.wrap(
        VeryPermissive(), usePolicy=win32com.server.policy.DynamicPolicy
    )
    try:
        handle = pythoncom.RegisterActiveObject(ob, iid, 0)
    except pythoncom.com_error as details:
        print("Warning - could not register the object in the ROT:", details)
        handle = None
    try:
        import win32com.client.dynamic

        client = win32com.client.dynamic.Dispatch(iid)
        client.ANewAttr = "Hello"
        if client.ANewAttr != "Hello":
            raise error("Could not set dynamic property")

        v = ["Hello", "From", "Python", 1.4]
        client.TestSequence = v
        if v != list(client.TestSequence):
            raise error(
                "Dynamic sequences not working! %r/%r"
                % (repr(v), repr(client.testSequence))
            )

        client.write("This", "output", "has", "come", "via", "testDynamic.py")
        # Check our new "_FlagAsMethod" works (kinda!)
        client._FlagAsMethod("NotReallyAMethod")
        if not callable(client.NotReallyAMethod):
            raise error("Method I flagged as callable isn't!")

        client = None
    finally:
        if handle is not None:
            pythoncom.RevokeActiveObject(handle)
    print("Test worked!")


if __name__ == "__main__":
    Test()
