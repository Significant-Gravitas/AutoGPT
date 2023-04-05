print("This module depends on the dbapi20 compliance tests created by Stuart Bishop")
print("(see db-sig mailing list history for info)")
import platform
import sys
import unittest

import dbapi20
import setuptestframework

testfolder = setuptestframework.maketemp()
if "--package" in sys.argv:
    pth = setuptestframework.makeadopackage(testfolder)
    sys.argv.remove("--package")
else:
    pth = setuptestframework.find_ado_path()
if pth not in sys.path:
    sys.path.insert(1, pth)
# function to clean up the temporary folder -- calling program must run this function before exit.
cleanup = setuptestframework.getcleanupfunction()

import adodbapi
import adodbapi.is64bit as is64bit

db = adodbapi

if "--verbose" in sys.argv:
    db.adodbapi.verbose = 3

print(adodbapi.version)
print("Tested with dbapi20 %s" % dbapi20.__version__)

try:
    onWindows = bool(sys.getwindowsversion())  # seems to work on all versions of Python
except:
    onWindows = False

node = platform.node()

conn_kws = {}
host = "testsql.2txt.us,1430"  # if None, will use macro to fill in node name
instance = r"%s\SQLEXPRESS"
conn_kws["name"] = "adotest"

conn_kws["user"] = "adotestuser"  # None implies Windows security
conn_kws["password"] = "Sq1234567"
# macro definition for keyword "security" using macro "auto_security"
conn_kws["macro_auto_security"] = "security"

if host is None:
    conn_kws["macro_getnode"] = ["host", instance]
else:
    conn_kws["host"] = host

conn_kws[
    "provider"
] = "Provider=MSOLEDBSQL;DataTypeCompatibility=80;MARS Connection=True;"
connStr = "%(provider)s; %(security)s; Initial Catalog=%(name)s;Data Source=%(host)s"

if onWindows and node != "z-PC":
    pass  # default should make a local SQL Server connection
elif node == "xxx":  # try Postgres database
    _computername = "25.223.161.222"
    _databasename = "adotest"
    _username = "adotestuser"
    _password = "12345678"
    _driver = "PostgreSQL Unicode"
    _provider = ""
    connStr = "%sDriver={%s};Server=%s;Database=%s;uid=%s;pwd=%s;" % (
        _provider,
        _driver,
        _computername,
        _databasename,
        _username,
        _password,
    )
elif node == "yyy":  # ACCESS data base is known to fail some tests.
    if is64bit.Python():
        driver = "Microsoft.ACE.OLEDB.12.0"
    else:
        driver = "Microsoft.Jet.OLEDB.4.0"
    testmdb = setuptestframework.makemdb(testfolder)
    connStr = r"Provider=%s;Data Source=%s" % (driver, testmdb)
else:  # try a remote connection to an SQL server
    conn_kws["proxy_host"] = "25.44.77.176"
    import adodbapi.remote

    db = adodbapi.remote

print("Using Connection String like=%s" % connStr)
print("Keywords=%s" % repr(conn_kws))


class test_adodbapi(dbapi20.DatabaseAPI20Test):
    driver = db
    connect_args = (connStr,)
    connect_kw_args = conn_kws

    def __init__(self, arg):
        dbapi20.DatabaseAPI20Test.__init__(self, arg)

    def getTestMethodName(self):
        return self.id().split(".")[-1]

    def setUp(self):
        # Call superclass setUp In case this does something in the
        # future
        dbapi20.DatabaseAPI20Test.setUp(self)
        if self.getTestMethodName() == "test_callproc":
            con = self._connect()
            engine = con.dbms_name
            ## print('Using database Engine=%s' % engine) ##
            if engine != "MS Jet":
                sql = """
                    create procedure templower
                        @theData varchar(50)
                    as
                        select lower(@theData)
                """
            else:  # Jet
                sql = """
                    create procedure templower
                        (theData varchar(50))
                    as
                        select lower(theData);
                """
            cur = con.cursor()
            try:
                cur.execute(sql)
                con.commit()
            except:
                pass
            cur.close()
            con.close()
            self.lower_func = "templower"

    def tearDown(self):
        if self.getTestMethodName() == "test_callproc":
            con = self._connect()
            cur = con.cursor()
            try:
                cur.execute("drop procedure templower")
            except:
                pass
            con.commit()
        dbapi20.DatabaseAPI20Test.tearDown(self)

    def help_nextset_setUp(self, cur):
        "Should create a procedure called deleteme"
        'that returns two result sets, first the number of rows in booze then "name from booze"'
        sql = """
            create procedure deleteme as
            begin
                select count(*) from %sbooze
                select name from %sbooze
            end
        """ % (
            self.table_prefix,
            self.table_prefix,
        )
        cur.execute(sql)

    def help_nextset_tearDown(self, cur):
        "If cleaning up is needed after nextSetTest"
        try:
            cur.execute("drop procedure deleteme")
        except:
            pass

    def test_nextset(self):
        con = self._connect()
        try:
            cur = con.cursor()

            stmts = [self.ddl1] + self._populate()
            for sql in stmts:
                cur.execute(sql)

            self.help_nextset_setUp(cur)

            cur.callproc("deleteme")
            numberofrows = cur.fetchone()
            assert numberofrows[0] == 6
            assert cur.nextset()
            names = cur.fetchall()
            assert len(names) == len(self.samples)
            s = cur.nextset()
            assert s == None, "No more return sets, should return None"
        finally:
            try:
                self.help_nextset_tearDown(cur)
            finally:
                con.close()

    def test_setoutputsize(self):
        pass


if __name__ == "__main__":
    unittest.main()
    cleanup(testfolder, None)
