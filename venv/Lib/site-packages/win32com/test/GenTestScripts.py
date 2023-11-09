#
# Generate scripts needed for serious testing!
#
import os
import sys

import pythoncom
import win32com
import win32com.client.makepy
import win32com.test

genList = [
    ("msword8", "{00020905-0000-0000-C000-000000000046}", 1033, 8, 0),
]

genDir = "Generated4Test"


def GetGenPath():
    import win32api

    return os.path.join(win32api.GetFullPathName(win32com.test.__path__[0]), genDir)


def GenerateFromRegistered(fname, *loadArgs):
    #       tlb = apply(pythoncom.LoadRegTypeLib, loadArgs)
    genPath = GetGenPath()
    try:
        os.stat(genPath)
    except os.error:
        os.mkdir(genPath)
    # Ensure an __init__ exists.
    open(os.path.join(genPath, "__init__.py"), "w").close()
    print(fname, ": generating -", end=" ")
    f = open(os.path.join(genPath, fname + ".py"), "w")
    win32com.client.makepy.GenerateFromTypeLibSpec(
        loadArgs, f, bQuiet=1, bGUIProgress=1
    )
    f.close()
    print("compiling -", end=" ")
    fullModName = "win32com.test.%s.%s" % (genDir, fname)
    exec("import " + fullModName)
    # Inject the generated module as a top level module.
    sys.modules[fname] = sys.modules[fullModName]
    print("done")


def GenerateAll():
    for args in genList:
        try:
            GenerateFromRegistered(*args)
        except KeyboardInterrupt:
            print("** Interrupted ***")
            break
        except pythoncom.com_error:
            print("** Could not generate test code for ", args[0])


def CleanAll():
    print("Cleaning generated test scripts...")
    try:  # Clear exceptions!
        1 / 0
    except:
        pass
    genPath = GetGenPath()
    for args in genList:
        try:
            name = args[0] + ".py"
            os.unlink(os.path.join(genPath, name))
        except os.error as details:
            if type(details) == type(()) and details[0] != 2:
                print("Could not deleted generated", name, details)
        try:
            name = args[0] + ".pyc"
            os.unlink(os.path.join(genPath, name))
        except os.error as details:
            if type(details) == type(()) and details[0] != 2:
                print("Could not deleted generated", name, details)
        try:
            os.unlink(os.path.join(genPath, "__init__.py"))
        except:
            pass
        try:
            os.unlink(os.path.join(genPath, "__init__.pyc"))
        except:
            pass
    try:
        os.rmdir(genPath)
    except os.error as details:
        print("Could not delete test directory -", details)


if __name__ == "__main__":
    GenerateAll()
    CleanAll()
