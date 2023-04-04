# vss.py -- Source Control using Microsoft VSS.

# Provides routines for checking files out of VSS.
#
# Uses an INI file very similar to how VB integrates with VSS - even
# as far as using the same name.

# The file must be named "Mssccprj.scc", and be in the format of
# an INI file.  This file may be in a parent directory, in which
# case the project name will be built from what is specified in the
# ini file, plus the path from the INI file to the file itself.
#
# The INI file should have a [Python] section, and a
# Project=Project Name
#  and optionally
# Database=??


import os
import sys
import traceback

import win32api
import win32ui

g_iniName = "Mssccprj.scc"  # Use the same INI name as VB!

g_sourceSafe = None


def FindVssProjectInfo(fullfname):
    """Looks up the file system for an INI file describing the project.

    Looking up the tree is for ni style packages.

    Returns (projectName, pathToFileName) where pathToFileName contains
    the path from the ini file to the actual file.
    """
    path, fnameonly = os.path.split(fullfname)
    origPath = path
    project = ""
    retPaths = [fnameonly]
    while not project:
        iniName = os.path.join(path, g_iniName)
        database = win32api.GetProfileVal("Python", "Database", "", iniName)
        project = win32api.GetProfileVal("Python", "Project", "", iniName)
        if project:
            break
        # No valid INI file in this directory - look up a level.
        path, addpath = os.path.split(path)
        if not addpath:  # Root?
            break
        retPaths.insert(0, addpath)
    if not project:
        win32ui.MessageBox(
            "%s\r\n\r\nThis directory is not configured for Python/VSS" % origPath
        )
        return
    return project, "/".join(retPaths), database


def CheckoutFile(fileName):
    global g_sourceSafe
    import pythoncom

    ok = 0
    # Assumes the fileName has a complete path,
    # and that the INI file can be found in that path
    # (or a parent path if a ni style package)
    try:
        import win32com.client
        import win32com.client.gencache

        mod = win32com.client.gencache.EnsureModule(
            "{783CD4E0-9D54-11CF-B8EE-00608CC9A71F}", 0, 5, 0
        )
        if mod is None:
            win32ui.MessageBox(
                "VSS does not appear to be installed.  The TypeInfo can not be created"
            )
            return ok

        rc = FindVssProjectInfo(fileName)
        if rc is None:
            return
        project, vssFname, database = rc
        if g_sourceSafe is None:
            g_sourceSafe = win32com.client.Dispatch("SourceSafe")
            # SS seems a bit wierd.  It defaults the arguments as empty strings, but
            # then complains when they are used - so we pass "Missing"
            if not database:
                database = pythoncom.Missing
            g_sourceSafe.Open(database, pythoncom.Missing, pythoncom.Missing)
        item = g_sourceSafe.VSSItem("$/%s/%s" % (project, vssFname))
        item.Checkout(None, fileName)
        ok = 1
    except pythoncom.com_error as exc:
        win32ui.MessageBox(exc.strerror, "Error checking out file")
    except:
        typ, val, tb = sys.exc_info()
        traceback.print_exc()
        win32ui.MessageBox("%s - %s" % (str(typ), str(val)), "Error checking out file")
        tb = None  # Cleanup a cycle
    return ok
