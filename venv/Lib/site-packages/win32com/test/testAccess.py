#
# This assumes that you have MSAccess and DAO installed.
#  You need to run makepy.py over "msaccess.tlb" and
#  "dao3032.dll", and ensure the generated files are on the
#  path.

# You can run this with no args, and a test database will be generated.
# You can optionally pass a dbname on the command line, in which case it will be dumped.

import os
import sys

import pythoncom
import win32api
from win32com.client import Dispatch, constants, gencache


def CreateTestAccessDatabase(dbname=None):
    # Creates a test access database - returns the filename.
    if dbname is None:
        dbname = os.path.join(win32api.GetTempPath(), "COMTestSuiteTempDatabase.mdb")

    access = Dispatch("Access.Application")
    dbEngine = access.DBEngine
    workspace = dbEngine.Workspaces(0)

    try:
        os.unlink(dbname)
    except os.error:
        print(
            "WARNING - Unable to delete old test database - expect a COM exception RSN!"
        )

    newdb = workspace.CreateDatabase(
        dbname, constants.dbLangGeneral, constants.dbEncrypt
    )

    # Create one test table.
    table = newdb.CreateTableDef("Test Table 1")
    table.Fields.Append(table.CreateField("First Name", constants.dbText))
    table.Fields.Append(table.CreateField("Last Name", constants.dbText))

    index = table.CreateIndex("UniqueIndex")
    index.Fields.Append(index.CreateField("First Name"))
    index.Fields.Append(index.CreateField("Last Name"))
    index.Unique = -1
    table.Indexes.Append(index)

    newdb.TableDefs.Append(table)

    # Create a second test table.
    table = newdb.CreateTableDef("Test Table 2")
    table.Fields.Append(table.CreateField("First Name", constants.dbText))
    table.Fields.Append(table.CreateField("Last Name", constants.dbText))

    newdb.TableDefs.Append(table)

    # Create a relationship between them
    relation = newdb.CreateRelation("TestRelationship")
    relation.Table = "Test Table 1"
    relation.ForeignTable = "Test Table 2"

    field = relation.CreateField("First Name")
    field.ForeignName = "First Name"
    relation.Fields.Append(field)

    field = relation.CreateField("Last Name")
    field.ForeignName = "Last Name"
    relation.Fields.Append(field)

    relation.Attributes = (
        constants.dbRelationDeleteCascade + constants.dbRelationUpdateCascade
    )

    newdb.Relations.Append(relation)

    # Finally we can add some data to the table.
    tab1 = newdb.OpenRecordset("Test Table 1")
    tab1.AddNew()
    tab1.Fields("First Name").Value = "Mark"
    tab1.Fields("Last Name").Value = "Hammond"
    tab1.Update()

    tab1.MoveFirst()
    # We do a simple bookmark test which tests our optimized VT_SAFEARRAY|VT_UI1 support.
    # The bookmark will be a buffer object - remember it for later.
    bk = tab1.Bookmark

    # Add a second record.
    tab1.AddNew()
    tab1.Fields("First Name").Value = "Second"
    tab1.Fields("Last Name").Value = "Person"
    tab1.Update()

    # Reset the bookmark to the one we saved.
    # But first check the test is actually doing something!
    tab1.MoveLast()
    if tab1.Fields("First Name").Value != "Second":
        raise RuntimeError("Unexpected record is last - makes bookmark test pointless!")

    tab1.Bookmark = bk
    if tab1.Bookmark != bk:
        raise RuntimeError("The bookmark data is not the same")

    if tab1.Fields("First Name").Value != "Mark":
        raise RuntimeError("The bookmark did not reset the record pointer correctly")

    return dbname


def DoDumpAccessInfo(dbname):
    from . import daodump

    a = forms = None
    try:
        sys.stderr.write("Creating Access Application...\n")
        a = Dispatch("Access.Application")
        print("Opening database %s" % dbname)
        a.OpenCurrentDatabase(dbname)
        db = a.CurrentDb()
        daodump.DumpDB(db, 1)
        forms = a.Forms
        print("There are %d forms open." % (len(forms)))
        # Uncommenting these lines means Access remains open.
        #               for form in forms:
        #                       print " %s" % form.Name
        reports = a.Reports
        print("There are %d reports open" % (len(reports)))
    finally:
        if not a is None:
            sys.stderr.write("Closing database\n")
            try:
                a.CloseCurrentDatabase()
            except pythoncom.com_error:
                pass


# Generate all the support we can.
def GenerateSupport():
    # dao
    gencache.EnsureModule("{00025E01-0000-0000-C000-000000000046}", 0, 4, 0)
    # Access
    #       gencache.EnsureModule("{4AFFC9A0-5F99-101B-AF4E-00AA003F0F07}", 0, 8, 0)
    gencache.EnsureDispatch("Access.Application")


def DumpAccessInfo(dbname):
    amod = gencache.GetModuleForProgID("Access.Application")
    dmod = gencache.GetModuleForProgID("DAO.DBEngine.35")
    if amod is None and dmod is None:
        DoDumpAccessInfo(dbname)
        # Now generate all the support we can.
        GenerateSupport()
    else:
        sys.stderr.write(
            "testAccess not doing dynamic test, as generated code already exists\n"
        )
    # Now a generated version.
    DoDumpAccessInfo(dbname)


def test(dbname=None):
    if dbname is None:
        # We need makepy support to create a database (just for the constants!)
        try:
            GenerateSupport()
        except pythoncom.com_error:
            print("*** Can not import the MSAccess type libraries - tests skipped")
            return
        dbname = CreateTestAccessDatabase()
        print("A test database at '%s' was created" % dbname)

    DumpAccessInfo(dbname)


if __name__ == "__main__":
    import sys

    from .util import CheckClean

    dbname = None
    if len(sys.argv) > 1:
        dbname = sys.argv[1]

    test(dbname)

    CheckClean()
