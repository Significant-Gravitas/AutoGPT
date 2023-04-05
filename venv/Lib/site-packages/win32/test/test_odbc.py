# odbc test suite kindly contributed by Frank Millman.
import os
import sys
import tempfile
import unittest

import odbc
import pythoncom
from pywin32_testutil import TestSkipped, str2bytes, str2memory
from win32com.client import constants

# We use the DAO ODBC driver
from win32com.client.gencache import EnsureDispatch


class TestStuff(unittest.TestCase):
    def setUp(self):
        self.tablename = "pywin32test_users"
        self.db_filename = None
        self.conn = self.cur = None
        try:
            # Test any database if a connection string is supplied...
            conn_str = os.environ["TEST_ODBC_CONNECTION_STRING"]
        except KeyError:
            # Create a local MSAccess DB for testing.
            self.db_filename = tempfile.NamedTemporaryFile().name + ".mdb"

            # Create a brand-new database - what is the story with these?
            for suffix in (".36", ".35", ".30"):
                try:
                    dbe = EnsureDispatch("DAO.DBEngine" + suffix)
                    break
                except pythoncom.com_error:
                    pass
            else:
                raise TestSkipped("Can't find a DB engine")

            workspace = dbe.Workspaces(0)

            newdb = workspace.CreateDatabase(
                self.db_filename, constants.dbLangGeneral, constants.dbEncrypt
            )

            newdb.Close()

            conn_str = "Driver={Microsoft Access Driver (*.mdb)};dbq=%s;Uid=;Pwd=;" % (
                self.db_filename,
            )
        ## print 'Connection string:', conn_str
        self.conn = odbc.odbc(conn_str)
        # And we expect a 'users' table for these tests.
        self.cur = self.conn.cursor()
        ## self.cur.setoutputsize(1000)
        try:
            self.cur.execute("""drop table %s""" % self.tablename)
        except (odbc.error, odbc.progError):
            pass

        ## This needs to be adjusted for sql server syntax for unicode fields
        ##  - memo -> TEXT
        ##  - varchar -> nvarchar
        self.assertEqual(
            self.cur.execute(
                """create table %s (
                    userid varchar(25),
                    username varchar(25),
                    bitfield bit,
                    intfield integer,
                    floatfield float,
                    datefield datetime,
                    rawfield varbinary(100),
                    longtextfield memo,
                    longbinaryfield image
            )"""
                % self.tablename
            ),
            -1,
        )

    def tearDown(self):
        if self.cur is not None:
            try:
                self.cur.execute("""drop table %s""" % self.tablename)
            except (odbc.error, odbc.progError) as why:
                print("Failed to delete test table %s" % self.tablename, why)

            self.cur.close()
            self.cur = None
        if self.conn is not None:
            self.conn.close()
            self.conn = None
        if self.db_filename is not None:
            try:
                os.unlink(self.db_filename)
            except OSError:
                pass

    def test_insert_select(self, userid="Frank", username="Frank Millman"):
        self.assertEqual(
            self.cur.execute(
                "insert into %s (userid, username) \
            values (?,?)"
                % self.tablename,
                [userid, username],
            ),
            1,
        )
        self.assertEqual(
            self.cur.execute(
                "select * from %s \
            where userid = ?"
                % self.tablename,
                [userid.lower()],
            ),
            0,
        )
        self.assertEqual(
            self.cur.execute(
                "select * from %s \
            where username = ?"
                % self.tablename,
                [username.lower()],
            ),
            0,
        )

    def test_insert_select_unicode(self, userid="Frank", username="Frank Millman"):
        self.assertEqual(
            self.cur.execute(
                "insert into %s (userid, username)\
            values (?,?)"
                % self.tablename,
                [userid, username],
            ),
            1,
        )
        self.assertEqual(
            self.cur.execute(
                "select * from %s \
            where userid = ?"
                % self.tablename,
                [userid.lower()],
            ),
            0,
        )
        self.assertEqual(
            self.cur.execute(
                "select * from %s \
            where username = ?"
                % self.tablename,
                [username.lower()],
            ),
            0,
        )

    def test_insert_select_unicode_ext(self):
        userid = "t-\xe0\xf2"
        username = "test-\xe0\xf2 name"
        self.test_insert_select_unicode(userid, username)

    def _test_val(self, fieldName, value):
        for x in range(100):
            self.cur.execute("delete from %s where userid='Frank'" % self.tablename)
            self.assertEqual(
                self.cur.execute(
                    "insert into %s (userid, %s) values (?,?)"
                    % (self.tablename, fieldName),
                    ["Frank", value],
                ),
                1,
            )
            self.cur.execute(
                "select %s from %s where userid = ?" % (fieldName, self.tablename),
                ["Frank"],
            )
            rows = self.cur.fetchmany()
            self.assertEqual(1, len(rows))
            row = rows[0]
            self.assertEqual(row[0], value)

    def testBit(self):
        self._test_val("bitfield", 1)
        self._test_val("bitfield", 0)

    def testInt(self):
        self._test_val("intfield", 1)
        self._test_val("intfield", 0)
        try:
            big = sys.maxsize
        except AttributeError:
            big = sys.maxint
        self._test_val("intfield", big)

    def testFloat(self):
        self._test_val("floatfield", 1.01)
        self._test_val("floatfield", 0)

    def testVarchar(
        self,
    ):
        self._test_val("username", "foo")

    def testLongVarchar(self):
        """Test a long text field in excess of internal cursor data size (65536)"""
        self._test_val("longtextfield", "abc" * 70000)

    def testLongBinary(self):
        """Test a long raw field in excess of internal cursor data size (65536)"""
        self._test_val("longbinaryfield", str2memory("\0\1\2" * 70000))

    def testRaw(self):
        ## Test binary data
        self._test_val("rawfield", str2memory("\1\2\3\4\0\5\6\7\8"))

    def test_widechar(self):
        """Test a unicode character that would be mangled if bound as plain character.
        For example, previously the below was returned as ascii 'a'
        """
        self._test_val("username", "\u0101")

    def testDates(self):
        import datetime

        for v in ((1900, 12, 25, 23, 39, 59),):
            d = datetime.datetime(*v)
            self._test_val("datefield", d)

    def test_set_nonzero_length(self):
        self.assertEqual(
            self.cur.execute(
                "insert into %s (userid,username) " "values (?,?)" % self.tablename,
                ["Frank", "Frank Millman"],
            ),
            1,
        )
        self.assertEqual(
            self.cur.execute("update %s set username = ?" % self.tablename, ["Frank"]),
            1,
        )
        self.assertEqual(self.cur.execute("select * from %s" % self.tablename), 0)
        self.assertEqual(len(self.cur.fetchone()[1]), 5)

    def test_set_zero_length(self):
        self.assertEqual(
            self.cur.execute(
                "insert into %s (userid,username) " "values (?,?)" % self.tablename,
                [str2bytes("Frank"), ""],
            ),
            1,
        )
        self.assertEqual(self.cur.execute("select * from %s" % self.tablename), 0)
        self.assertEqual(len(self.cur.fetchone()[1]), 0)

    def test_set_zero_length_unicode(self):
        self.assertEqual(
            self.cur.execute(
                "insert into %s (userid,username) " "values (?,?)" % self.tablename,
                ["Frank", ""],
            ),
            1,
        )
        self.assertEqual(self.cur.execute("select * from %s" % self.tablename), 0)
        self.assertEqual(len(self.cur.fetchone()[1]), 0)


if __name__ == "__main__":
    unittest.main()
