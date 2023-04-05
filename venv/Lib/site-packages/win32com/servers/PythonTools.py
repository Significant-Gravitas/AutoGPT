import sys
import time


class Tools:
    _public_methods_ = ["reload", "adddir", "echo", "sleep"]

    def reload(self, module):
        if module in sys.modules:
            from importlib import reload

            reload(sys.modules[module])
            return "reload succeeded."
        return "no reload performed."

    def adddir(self, dir):
        if type(dir) == type(""):
            sys.path.append(dir)
        return str(sys.path)

    def echo(self, arg):
        return repr(arg)

    def sleep(self, t):
        time.sleep(t)


if __name__ == "__main__":
    from win32com.server.register import RegisterServer, UnregisterServer

    clsid = "{06ce7630-1d81-11d0-ae37-c2fa70000000}"
    progid = "Python.Tools"
    verprogid = "Python.Tools.1"
    if "--unregister" in sys.argv:
        print("Unregistering...")
        UnregisterServer(clsid, progid, verprogid)
        print("Unregistered OK")
    else:
        print("Registering COM server...")
        RegisterServer(
            clsid,
            "win32com.servers.PythonTools.Tools",
            "Python Tools",
            progid,
            verprogid,
        )
        print("Class registered.")
