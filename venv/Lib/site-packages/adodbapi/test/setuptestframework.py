#!/usr/bin/python2
# Configure this in order to run the testcases.
"setuptestframework.py v 2.6.0.8"
import os
import shutil
import sys
import tempfile

try:
    OSErrors = (WindowsError, OSError)
except NameError:  # not running on Windows
    OSErrors = OSError


def maketemp():
    temphome = tempfile.gettempdir()
    tempdir = os.path.join(temphome, "adodbapi_test")
    try:
        os.mkdir(tempdir)
    except:
        pass
    return tempdir


def _cleanup_function(testfolder, mdb_name):
    try:
        os.unlink(os.path.join(testfolder, mdb_name))
    except:
        pass  # mdb database not present
    try:
        shutil.rmtree(testfolder)
        print("   cleaned up folder", testfolder)
    except:
        pass  # test package not present


def getcleanupfunction():
    return _cleanup_function


def find_ado_path():
    adoName = os.path.normpath(os.getcwd() + "/../../adodbapi.py")
    adoPackage = os.path.dirname(adoName)
    return adoPackage


# make a new package directory for the test copy of ado
def makeadopackage(testfolder):
    adoName = os.path.normpath(os.getcwd() + "/../adodbapi.py")
    adoPath = os.path.dirname(adoName)
    if os.path.exists(adoName):
        newpackage = os.path.join(testfolder, "adodbapi")
        try:
            os.mkdir(newpackage)
        except OSErrors:
            print(
                "*Note: temporary adodbapi package already exists: may be two versions running?"
            )
        for f in os.listdir(adoPath):
            if f.endswith(".py"):
                shutil.copy(os.path.join(adoPath, f), newpackage)
        if sys.version_info >= (3, 0):  # only when running Py3.n
            save = sys.stdout
            sys.stdout = None
            from lib2to3.main import main  # use 2to3 to make test package

            main("lib2to3.fixes", args=["-n", "-w", newpackage])
            sys.stdout = save
        return testfolder
    else:
        raise EnvironmentError("Connot find source of adodbapi to test.")


def makemdb(testfolder, mdb_name):
    # following setup code borrowed from pywin32 odbc test suite
    # kindly contributed by Frank Millman.
    import os

    _accessdatasource = os.path.join(testfolder, mdb_name)
    if os.path.isfile(_accessdatasource):
        print("using JET database=", _accessdatasource)
    else:
        try:
            from win32com.client import constants
            from win32com.client.gencache import EnsureDispatch

            win32 = True
        except ImportError:  # perhaps we are running IronPython
            win32 = False  # iron Python
            try:
                from System import Activator, Type
            except:
                pass

        # Create a brand-new database - what is the story with these?
        dbe = None
        for suffix in (".36", ".35", ".30"):
            try:
                if win32:
                    dbe = EnsureDispatch("DAO.DBEngine" + suffix)
                else:
                    type = Type.GetTypeFromProgID("DAO.DBEngine" + suffix)
                    dbe = Activator.CreateInstance(type)
                break
            except:
                pass
        if dbe:
            print("    ...Creating ACCESS db at " + _accessdatasource)
            if win32:
                workspace = dbe.Workspaces(0)
                newdb = workspace.CreateDatabase(
                    _accessdatasource, constants.dbLangGeneral, constants.dbVersion40
                )
            else:
                newdb = dbe.CreateDatabase(
                    _accessdatasource, ";LANGID=0x0409;CP=1252;COUNTRY=0"
                )
            newdb.Close()
        else:
            print("    ...copying test ACCESS db to " + _accessdatasource)
            mdbName = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "examples", "test.mdb")
            )
            import shutil

            shutil.copy(mdbName, _accessdatasource)

    return _accessdatasource


if __name__ == "__main__":
    print("Setting up a Jet database for server to use for remote testing...")
    temp = maketemp()
    makemdb(temp, "server_test.mdb")
