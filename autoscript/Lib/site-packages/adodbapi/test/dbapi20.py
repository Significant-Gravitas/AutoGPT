#!/usr/bin/env python
""" Python DB API 2.0 driver compliance unit test suite. 
    
    This software is Public Domain and may be used without restrictions.

 "Now we have booze and barflies entering the discussion, plus rumours of
  DBAs on drugs... and I won't tell you what flashes through my mind each
  time I read the subject line with 'Anal Compliance' in it.  All around
  this is turning out to be a thoroughly unwholesome unit test."

    -- Ian Bicking
"""

__version__ = "$Revision: 1.15.0 $"[11:-2]
__author__ = "Stuart Bishop <stuart@stuartbishop.net>"

import sys
import time
import unittest

if sys.version[0] >= "3":  # python 3.x
    _BaseException = Exception

    def _failUnless(self, expr, msg=None):
        self.assertTrue(expr, msg)

else:  # python 2.x
    from exceptions import Exception as _BaseException

    def _failUnless(self, expr, msg=None):
        self.failUnless(expr, msg)  ## deprecated since Python 2.6


# set this to "True" to follow API 2.0 to the letter
TEST_FOR_NON_IDEMPOTENT_CLOSE = False

# Revision 1.15  2019/11/22 00:50:00  kf7xm
# Make Turn off IDEMPOTENT_CLOSE a proper skipTest

# Revision 1.14  2013/05/20 11:02:05  kf7xm
# Add a literal string to the format insertion test to catch trivial re-format algorithms

# Revision 1.13  2013/05/08 14:31:50  kf7xm
# Quick switch to Turn off IDEMPOTENT_CLOSE test. Also: Silence teardown failure


# Revision 1.12  2009/02/06 03:35:11  kf7xm
# Tested okay with Python 3.0, includes last minute patches from Mark H.
#
# Revision 1.1.1.1.2.1  2008/09/20 19:54:59  rupole
# Include latest changes from main branch
# Updates for py3k
#
# Revision 1.11  2005/01/02 02:41:01  zenzen
# Update author email address
#
# Revision 1.10  2003/10/09 03:14:14  zenzen
# Add test for DB API 2.0 optional extension, where database exceptions
# are exposed as attributes on the Connection object.
#
# Revision 1.9  2003/08/13 01:16:36  zenzen
# Minor tweak from Stefan Fleiter
#
# Revision 1.8  2003/04/10 00:13:25  zenzen
# Changes, as per suggestions by M.-A. Lemburg
# - Add a table prefix, to ensure namespace collisions can always be avoided
#
# Revision 1.7  2003/02/26 23:33:37  zenzen
# Break out DDL into helper functions, as per request by David Rushby
#
# Revision 1.6  2003/02/21 03:04:33  zenzen
# Stuff from Henrik Ekelund:
#     added test_None
#     added test_nextset & hooks
#
# Revision 1.5  2003/02/17 22:08:43  zenzen
# Implement suggestions and code from Henrik Eklund - test that cursor.arraysize
# defaults to 1 & generic cursor.callproc test added
#
# Revision 1.4  2003/02/15 00:16:33  zenzen
# Changes, as per suggestions and bug reports by M.-A. Lemburg,
# Matthew T. Kromer, Federico Di Gregorio and Daniel Dittmar
# - Class renamed
# - Now a subclass of TestCase, to avoid requiring the driver stub
#   to use multiple inheritance
# - Reversed the polarity of buggy test in test_description
# - Test exception heirarchy correctly
# - self.populate is now self._populate(), so if a driver stub
#   overrides self.ddl1 this change propogates
# - VARCHAR columns now have a width, which will hopefully make the
#   DDL even more portible (this will be reversed if it causes more problems)
# - cursor.rowcount being checked after various execute and fetchXXX methods
# - Check for fetchall and fetchmany returning empty lists after results
#   are exhausted (already checking for empty lists if select retrieved
#   nothing
# - Fix bugs in test_setoutputsize_basic and test_setinputsizes
#
def str2bytes(sval):
    if sys.version_info < (3, 0) and isinstance(sval, str):
        sval = sval.decode("latin1")
    return sval.encode("latin1")  # python 3 make unicode into bytes


class DatabaseAPI20Test(unittest.TestCase):
    """Test a database self.driver for DB API 2.0 compatibility.
    This implementation tests Gadfly, but the TestCase
    is structured so that other self.drivers can subclass this
    test case to ensure compiliance with the DB-API. It is
    expected that this TestCase may be expanded in the future
    if ambiguities or edge conditions are discovered.

    The 'Optional Extensions' are not yet being tested.

    self.drivers should subclass this test, overriding setUp, tearDown,
    self.driver, connect_args and connect_kw_args. Class specification
    should be as follows:

    import dbapi20
    class mytest(dbapi20.DatabaseAPI20Test):
       [...]

    Don't 'import DatabaseAPI20Test from dbapi20', or you will
    confuse the unit tester - just 'import dbapi20'.
    """

    # The self.driver module. This should be the module where the 'connect'
    # method is to be found
    driver = None
    connect_args = ()  # List of arguments to pass to connect
    connect_kw_args = {}  # Keyword arguments for connect
    table_prefix = "dbapi20test_"  # If you need to specify a prefix for tables

    ddl1 = "create table %sbooze (name varchar(20))" % table_prefix
    ddl2 = "create table %sbarflys (name varchar(20), drink varchar(30))" % table_prefix
    xddl1 = "drop table %sbooze" % table_prefix
    xddl2 = "drop table %sbarflys" % table_prefix

    lowerfunc = "lower"  # Name of stored procedure to convert string->lowercase

    # Some drivers may need to override these helpers, for example adding
    # a 'commit' after the execute.
    def executeDDL1(self, cursor):
        cursor.execute(self.ddl1)

    def executeDDL2(self, cursor):
        cursor.execute(self.ddl2)

    def setUp(self):
        """self.drivers should override this method to perform required setup
        if any is necessary, such as creating the database.
        """
        pass

    def tearDown(self):
        """self.drivers should override this method to perform required cleanup
        if any is necessary, such as deleting the test database.
        The default drops the tables that may be created.
        """
        try:
            con = self._connect()
            try:
                cur = con.cursor()
                for ddl in (self.xddl1, self.xddl2):
                    try:
                        cur.execute(ddl)
                        con.commit()
                    except self.driver.Error:
                        # Assume table didn't exist. Other tests will check if
                        # execute is busted.
                        pass
            finally:
                con.close()
        except _BaseException:
            pass

    def _connect(self):
        try:
            r = self.driver.connect(*self.connect_args, **self.connect_kw_args)
        except AttributeError:
            self.fail("No connect method found in self.driver module")
        return r

    def test_connect(self):
        con = self._connect()
        con.close()

    def test_apilevel(self):
        try:
            # Must exist
            apilevel = self.driver.apilevel
            # Must equal 2.0
            self.assertEqual(apilevel, "2.0")
        except AttributeError:
            self.fail("Driver doesn't define apilevel")

    def test_threadsafety(self):
        try:
            # Must exist
            threadsafety = self.driver.threadsafety
            # Must be a valid value
            _failUnless(self, threadsafety in (0, 1, 2, 3))
        except AttributeError:
            self.fail("Driver doesn't define threadsafety")

    def test_paramstyle(self):
        try:
            # Must exist
            paramstyle = self.driver.paramstyle
            # Must be a valid value
            _failUnless(
                self, paramstyle in ("qmark", "numeric", "named", "format", "pyformat")
            )
        except AttributeError:
            self.fail("Driver doesn't define paramstyle")

    def test_Exceptions(self):
        # Make sure required exceptions exist, and are in the
        # defined heirarchy.
        if sys.version[0] == "3":  # under Python 3 StardardError no longer exists
            self.assertTrue(issubclass(self.driver.Warning, Exception))
            self.assertTrue(issubclass(self.driver.Error, Exception))
        else:
            self.failUnless(issubclass(self.driver.Warning, Exception))
            self.failUnless(issubclass(self.driver.Error, Exception))

        _failUnless(self, issubclass(self.driver.InterfaceError, self.driver.Error))
        _failUnless(self, issubclass(self.driver.DatabaseError, self.driver.Error))
        _failUnless(self, issubclass(self.driver.OperationalError, self.driver.Error))
        _failUnless(self, issubclass(self.driver.IntegrityError, self.driver.Error))
        _failUnless(self, issubclass(self.driver.InternalError, self.driver.Error))
        _failUnless(self, issubclass(self.driver.ProgrammingError, self.driver.Error))
        _failUnless(self, issubclass(self.driver.NotSupportedError, self.driver.Error))

    def test_ExceptionsAsConnectionAttributes(self):
        # OPTIONAL EXTENSION
        # Test for the optional DB API 2.0 extension, where the exceptions
        # are exposed as attributes on the Connection object
        # I figure this optional extension will be implemented by any
        # driver author who is using this test suite, so it is enabled
        # by default.
        con = self._connect()
        drv = self.driver
        _failUnless(self, con.Warning is drv.Warning)
        _failUnless(self, con.Error is drv.Error)
        _failUnless(self, con.InterfaceError is drv.InterfaceError)
        _failUnless(self, con.DatabaseError is drv.DatabaseError)
        _failUnless(self, con.OperationalError is drv.OperationalError)
        _failUnless(self, con.IntegrityError is drv.IntegrityError)
        _failUnless(self, con.InternalError is drv.InternalError)
        _failUnless(self, con.ProgrammingError is drv.ProgrammingError)
        _failUnless(self, con.NotSupportedError is drv.NotSupportedError)

    def test_commit(self):
        con = self._connect()
        try:
            # Commit must work, even if it doesn't do anything
            con.commit()
        finally:
            con.close()

    def test_rollback(self):
        con = self._connect()
        # If rollback is defined, it should either work or throw
        # the documented exception
        if hasattr(con, "rollback"):
            try:
                con.rollback()
            except self.driver.NotSupportedError:
                pass

    def test_cursor(self):
        con = self._connect()
        try:
            cur = con.cursor()
        finally:
            con.close()

    def test_cursor_isolation(self):
        con = self._connect()
        try:
            # Make sure cursors created from the same connection have
            # the documented transaction isolation level
            cur1 = con.cursor()
            cur2 = con.cursor()
            self.executeDDL1(cur1)
            cur1.execute(
                "insert into %sbooze values ('Victoria Bitter')" % (self.table_prefix)
            )
            cur2.execute("select name from %sbooze" % self.table_prefix)
            booze = cur2.fetchall()
            self.assertEqual(len(booze), 1)
            self.assertEqual(len(booze[0]), 1)
            self.assertEqual(booze[0][0], "Victoria Bitter")
        finally:
            con.close()

    def test_description(self):
        con = self._connect()
        try:
            cur = con.cursor()
            self.executeDDL1(cur)
            self.assertEqual(
                cur.description,
                None,
                "cursor.description should be none after executing a "
                "statement that can return no rows (such as DDL)",
            )
            cur.execute("select name from %sbooze" % self.table_prefix)
            self.assertEqual(
                len(cur.description), 1, "cursor.description describes too many columns"
            )
            self.assertEqual(
                len(cur.description[0]),
                7,
                "cursor.description[x] tuples must have 7 elements",
            )
            self.assertEqual(
                cur.description[0][0].lower(),
                "name",
                "cursor.description[x][0] must return column name",
            )
            self.assertEqual(
                cur.description[0][1],
                self.driver.STRING,
                "cursor.description[x][1] must return column type. Got %r"
                % cur.description[0][1],
            )

            # Make sure self.description gets reset
            self.executeDDL2(cur)
            self.assertEqual(
                cur.description,
                None,
                "cursor.description not being set to None when executing "
                "no-result statements (eg. DDL)",
            )
        finally:
            con.close()

    def test_rowcount(self):
        con = self._connect()
        try:
            cur = con.cursor()
            self.executeDDL1(cur)
            _failUnless(
                self,
                cur.rowcount in (-1, 0),  # Bug #543885
                "cursor.rowcount should be -1 or 0 after executing no-result "
                "statements",
            )
            cur.execute(
                "insert into %sbooze values ('Victoria Bitter')" % (self.table_prefix)
            )
            _failUnless(
                self,
                cur.rowcount in (-1, 1),
                "cursor.rowcount should == number or rows inserted, or "
                "set to -1 after executing an insert statement",
            )
            cur.execute("select name from %sbooze" % self.table_prefix)
            _failUnless(
                self,
                cur.rowcount in (-1, 1),
                "cursor.rowcount should == number of rows returned, or "
                "set to -1 after executing a select statement",
            )
            self.executeDDL2(cur)
            self.assertEqual(
                cur.rowcount,
                -1,
                "cursor.rowcount not being reset to -1 after executing "
                "no-result statements",
            )
        finally:
            con.close()

    lower_func = "lower"

    def test_callproc(self):
        con = self._connect()
        try:
            cur = con.cursor()
            if self.lower_func and hasattr(cur, "callproc"):
                r = cur.callproc(self.lower_func, ("FOO",))
                self.assertEqual(len(r), 1)
                self.assertEqual(r[0], "FOO")
                r = cur.fetchall()
                self.assertEqual(len(r), 1, "callproc produced no result set")
                self.assertEqual(len(r[0]), 1, "callproc produced invalid result set")
                self.assertEqual(r[0][0], "foo", "callproc produced invalid results")
        finally:
            con.close()

    def test_close(self):
        con = self._connect()
        try:
            cur = con.cursor()
        finally:
            con.close()

        # cursor.execute should raise an Error if called after connection
        # closed
        self.assertRaises(self.driver.Error, self.executeDDL1, cur)

        # connection.commit should raise an Error if called after connection'
        # closed.'
        self.assertRaises(self.driver.Error, con.commit)

        # connection.close should raise an Error if called more than once
        #!!! reasonable persons differ about the usefulness of this test and this feature !!!
        if TEST_FOR_NON_IDEMPOTENT_CLOSE:
            self.assertRaises(self.driver.Error, con.close)
        else:
            self.skipTest(
                "Non-idempotent close is considered a bad thing by some people."
            )

    def test_execute(self):
        con = self._connect()
        try:
            cur = con.cursor()
            self._paraminsert(cur)
        finally:
            con.close()

    def _paraminsert(self, cur):
        self.executeDDL2(cur)
        cur.execute(
            "insert into %sbarflys values ('Victoria Bitter', 'thi%%s :may ca%%(u)se? troub:1e')"
            % (self.table_prefix)
        )
        _failUnless(self, cur.rowcount in (-1, 1))

        if self.driver.paramstyle == "qmark":
            cur.execute(
                "insert into %sbarflys values (?, 'thi%%s :may ca%%(u)se? troub:1e')"
                % self.table_prefix,
                ("Cooper's",),
            )
        elif self.driver.paramstyle == "numeric":
            cur.execute(
                "insert into %sbarflys values (:1, 'thi%%s :may ca%%(u)se? troub:1e')"
                % self.table_prefix,
                ("Cooper's",),
            )
        elif self.driver.paramstyle == "named":
            cur.execute(
                "insert into %sbarflys values (:beer, 'thi%%s :may ca%%(u)se? troub:1e')"
                % self.table_prefix,
                {"beer": "Cooper's"},
            )
        elif self.driver.paramstyle == "format":
            cur.execute(
                "insert into %sbarflys values (%%s, 'thi%%s :may ca%%(u)se? troub:1e')"
                % self.table_prefix,
                ("Cooper's",),
            )
        elif self.driver.paramstyle == "pyformat":
            cur.execute(
                "insert into %sbarflys values (%%(beer)s, 'thi%%s :may ca%%(u)se? troub:1e')"
                % self.table_prefix,
                {"beer": "Cooper's"},
            )
        else:
            self.fail("Invalid paramstyle")
        _failUnless(self, cur.rowcount in (-1, 1))

        cur.execute("select name, drink from %sbarflys" % self.table_prefix)
        res = cur.fetchall()
        self.assertEqual(len(res), 2, "cursor.fetchall returned too few rows")
        beers = [res[0][0], res[1][0]]
        beers.sort()
        self.assertEqual(
            beers[0],
            "Cooper's",
            "cursor.fetchall retrieved incorrect data, or data inserted " "incorrectly",
        )
        self.assertEqual(
            beers[1],
            "Victoria Bitter",
            "cursor.fetchall retrieved incorrect data, or data inserted " "incorrectly",
        )
        trouble = "thi%s :may ca%(u)se? troub:1e"
        self.assertEqual(
            res[0][1],
            trouble,
            "cursor.fetchall retrieved incorrect data, or data inserted "
            "incorrectly. Got=%s, Expected=%s" % (repr(res[0][1]), repr(trouble)),
        )
        self.assertEqual(
            res[1][1],
            trouble,
            "cursor.fetchall retrieved incorrect data, or data inserted "
            "incorrectly. Got=%s, Expected=%s" % (repr(res[1][1]), repr(trouble)),
        )

    def test_executemany(self):
        con = self._connect()
        try:
            cur = con.cursor()
            self.executeDDL1(cur)
            largs = [("Cooper's",), ("Boag's",)]
            margs = [{"beer": "Cooper's"}, {"beer": "Boag's"}]
            if self.driver.paramstyle == "qmark":
                cur.executemany(
                    "insert into %sbooze values (?)" % self.table_prefix, largs
                )
            elif self.driver.paramstyle == "numeric":
                cur.executemany(
                    "insert into %sbooze values (:1)" % self.table_prefix, largs
                )
            elif self.driver.paramstyle == "named":
                cur.executemany(
                    "insert into %sbooze values (:beer)" % self.table_prefix, margs
                )
            elif self.driver.paramstyle == "format":
                cur.executemany(
                    "insert into %sbooze values (%%s)" % self.table_prefix, largs
                )
            elif self.driver.paramstyle == "pyformat":
                cur.executemany(
                    "insert into %sbooze values (%%(beer)s)" % (self.table_prefix),
                    margs,
                )
            else:
                self.fail("Unknown paramstyle")
            _failUnless(
                self,
                cur.rowcount in (-1, 2),
                "insert using cursor.executemany set cursor.rowcount to "
                "incorrect value %r" % cur.rowcount,
            )
            cur.execute("select name from %sbooze" % self.table_prefix)
            res = cur.fetchall()
            self.assertEqual(
                len(res), 2, "cursor.fetchall retrieved incorrect number of rows"
            )
            beers = [res[0][0], res[1][0]]
            beers.sort()
            self.assertEqual(
                beers[0], "Boag's", 'incorrect data "%s" retrieved' % beers[0]
            )
            self.assertEqual(beers[1], "Cooper's", "incorrect data retrieved")
        finally:
            con.close()

    def test_fetchone(self):
        con = self._connect()
        try:
            cur = con.cursor()

            # cursor.fetchone should raise an Error if called before
            # executing a select-type query
            self.assertRaises(self.driver.Error, cur.fetchone)

            # cursor.fetchone should raise an Error if called after
            # executing a query that cannnot return rows
            self.executeDDL1(cur)
            self.assertRaises(self.driver.Error, cur.fetchone)

            cur.execute("select name from %sbooze" % self.table_prefix)
            self.assertEqual(
                cur.fetchone(),
                None,
                "cursor.fetchone should return None if a query retrieves " "no rows",
            )
            _failUnless(self, cur.rowcount in (-1, 0))

            # cursor.fetchone should raise an Error if called after
            # executing a query that cannnot return rows
            cur.execute(
                "insert into %sbooze values ('Victoria Bitter')" % (self.table_prefix)
            )
            self.assertRaises(self.driver.Error, cur.fetchone)

            cur.execute("select name from %sbooze" % self.table_prefix)
            r = cur.fetchone()
            self.assertEqual(
                len(r), 1, "cursor.fetchone should have retrieved a single row"
            )
            self.assertEqual(
                r[0], "Victoria Bitter", "cursor.fetchone retrieved incorrect data"
            )
            self.assertEqual(
                cur.fetchone(),
                None,
                "cursor.fetchone should return None if no more rows available",
            )
            _failUnless(self, cur.rowcount in (-1, 1))
        finally:
            con.close()

    samples = [
        "Carlton Cold",
        "Carlton Draft",
        "Mountain Goat",
        "Redback",
        "Victoria Bitter",
        "XXXX",
    ]

    def _populate(self):
        """Return a list of sql commands to setup the DB for the fetch
        tests.
        """
        populate = [
            "insert into %sbooze values ('%s')" % (self.table_prefix, s)
            for s in self.samples
        ]
        return populate

    def test_fetchmany(self):
        con = self._connect()
        try:
            cur = con.cursor()

            # cursor.fetchmany should raise an Error if called without
            # issuing a query
            self.assertRaises(self.driver.Error, cur.fetchmany, 4)

            self.executeDDL1(cur)
            for sql in self._populate():
                cur.execute(sql)

            cur.execute("select name from %sbooze" % self.table_prefix)
            r = cur.fetchmany()
            self.assertEqual(
                len(r),
                1,
                "cursor.fetchmany retrieved incorrect number of rows, "
                "default of arraysize is one.",
            )
            cur.arraysize = 10
            r = cur.fetchmany(3)  # Should get 3 rows
            self.assertEqual(
                len(r), 3, "cursor.fetchmany retrieved incorrect number of rows"
            )
            r = cur.fetchmany(4)  # Should get 2 more
            self.assertEqual(
                len(r), 2, "cursor.fetchmany retrieved incorrect number of rows"
            )
            r = cur.fetchmany(4)  # Should be an empty sequence
            self.assertEqual(
                len(r),
                0,
                "cursor.fetchmany should return an empty sequence after "
                "results are exhausted",
            )
            _failUnless(self, cur.rowcount in (-1, 6))

            # Same as above, using cursor.arraysize
            cur.arraysize = 4
            cur.execute("select name from %sbooze" % self.table_prefix)
            r = cur.fetchmany()  # Should get 4 rows
            self.assertEqual(
                len(r), 4, "cursor.arraysize not being honoured by fetchmany"
            )
            r = cur.fetchmany()  # Should get 2 more
            self.assertEqual(len(r), 2)
            r = cur.fetchmany()  # Should be an empty sequence
            self.assertEqual(len(r), 0)
            _failUnless(self, cur.rowcount in (-1, 6))

            cur.arraysize = 6
            cur.execute("select name from %sbooze" % self.table_prefix)
            rows = cur.fetchmany()  # Should get all rows
            _failUnless(self, cur.rowcount in (-1, 6))
            self.assertEqual(len(rows), 6)
            self.assertEqual(len(rows), 6)
            rows = [r[0] for r in rows]
            rows.sort()

            # Make sure we get the right data back out
            for i in range(0, 6):
                self.assertEqual(
                    rows[i],
                    self.samples[i],
                    "incorrect data retrieved by cursor.fetchmany",
                )

            rows = cur.fetchmany()  # Should return an empty list
            self.assertEqual(
                len(rows),
                0,
                "cursor.fetchmany should return an empty sequence if "
                "called after the whole result set has been fetched",
            )
            _failUnless(self, cur.rowcount in (-1, 6))

            self.executeDDL2(cur)
            cur.execute("select name from %sbarflys" % self.table_prefix)
            r = cur.fetchmany()  # Should get empty sequence
            self.assertEqual(
                len(r),
                0,
                "cursor.fetchmany should return an empty sequence if "
                "query retrieved no rows",
            )
            _failUnless(self, cur.rowcount in (-1, 0))

        finally:
            con.close()

    def test_fetchall(self):
        con = self._connect()
        try:
            cur = con.cursor()
            # cursor.fetchall should raise an Error if called
            # without executing a query that may return rows (such
            # as a select)
            self.assertRaises(self.driver.Error, cur.fetchall)

            self.executeDDL1(cur)
            for sql in self._populate():
                cur.execute(sql)

            # cursor.fetchall should raise an Error if called
            # after executing a a statement that cannot return rows
            self.assertRaises(self.driver.Error, cur.fetchall)

            cur.execute("select name from %sbooze" % self.table_prefix)
            rows = cur.fetchall()
            _failUnless(self, cur.rowcount in (-1, len(self.samples)))
            self.assertEqual(
                len(rows),
                len(self.samples),
                "cursor.fetchall did not retrieve all rows",
            )
            rows = [r[0] for r in rows]
            rows.sort()
            for i in range(0, len(self.samples)):
                self.assertEqual(
                    rows[i], self.samples[i], "cursor.fetchall retrieved incorrect rows"
                )
            rows = cur.fetchall()
            self.assertEqual(
                len(rows),
                0,
                "cursor.fetchall should return an empty list if called "
                "after the whole result set has been fetched",
            )
            _failUnless(self, cur.rowcount in (-1, len(self.samples)))

            self.executeDDL2(cur)
            cur.execute("select name from %sbarflys" % self.table_prefix)
            rows = cur.fetchall()
            _failUnless(self, cur.rowcount in (-1, 0))
            self.assertEqual(
                len(rows),
                0,
                "cursor.fetchall should return an empty list if "
                "a select query returns no rows",
            )

        finally:
            con.close()

    def test_mixedfetch(self):
        con = self._connect()
        try:
            cur = con.cursor()
            self.executeDDL1(cur)
            for sql in self._populate():
                cur.execute(sql)

            cur.execute("select name from %sbooze" % self.table_prefix)
            rows1 = cur.fetchone()
            rows23 = cur.fetchmany(2)
            rows4 = cur.fetchone()
            rows56 = cur.fetchall()
            _failUnless(self, cur.rowcount in (-1, 6))
            self.assertEqual(
                len(rows23), 2, "fetchmany returned incorrect number of rows"
            )
            self.assertEqual(
                len(rows56), 2, "fetchall returned incorrect number of rows"
            )

            rows = [rows1[0]]
            rows.extend([rows23[0][0], rows23[1][0]])
            rows.append(rows4[0])
            rows.extend([rows56[0][0], rows56[1][0]])
            rows.sort()
            for i in range(0, len(self.samples)):
                self.assertEqual(
                    rows[i], self.samples[i], "incorrect data retrieved or inserted"
                )
        finally:
            con.close()

    def help_nextset_setUp(self, cur):
        """Should create a procedure called deleteme
        that returns two result sets, first the
        number of rows in booze then "name from booze"
        """
        raise NotImplementedError("Helper not implemented")
        # sql="""
        #    create procedure deleteme as
        #    begin
        #        select count(*) from booze
        #        select name from booze
        #    end
        # """
        # cur.execute(sql)

    def help_nextset_tearDown(self, cur):
        "If cleaning up is needed after nextSetTest"
        raise NotImplementedError("Helper not implemented")
        # cur.execute("drop procedure deleteme")

    def test_nextset(self):
        con = self._connect()
        try:
            cur = con.cursor()
            if not hasattr(cur, "nextset"):
                return

            try:
                self.executeDDL1(cur)
                sql = self._populate()
                for sql in self._populate():
                    cur.execute(sql)

                self.help_nextset_setUp(cur)

                cur.callproc("deleteme")
                numberofrows = cur.fetchone()
                assert numberofrows[0] == len(self.samples)
                assert cur.nextset()
                names = cur.fetchall()
                assert len(names) == len(self.samples)
                s = cur.nextset()
                assert s == None, "No more return sets, should return None"
            finally:
                self.help_nextset_tearDown(cur)

        finally:
            con.close()

    def test_nextset(self):
        raise NotImplementedError("Drivers need to override this test")

    def test_arraysize(self):
        # Not much here - rest of the tests for this are in test_fetchmany
        con = self._connect()
        try:
            cur = con.cursor()
            _failUnless(
                self, hasattr(cur, "arraysize"), "cursor.arraysize must be defined"
            )
        finally:
            con.close()

    def test_setinputsizes(self):
        con = self._connect()
        try:
            cur = con.cursor()
            cur.setinputsizes((25,))
            self._paraminsert(cur)  # Make sure cursor still works
        finally:
            con.close()

    def test_setoutputsize_basic(self):
        # Basic test is to make sure setoutputsize doesn't blow up
        con = self._connect()
        try:
            cur = con.cursor()
            cur.setoutputsize(1000)
            cur.setoutputsize(2000, 0)
            self._paraminsert(cur)  # Make sure the cursor still works
        finally:
            con.close()

    def test_setoutputsize(self):
        # Real test for setoutputsize is driver dependant
        raise NotImplementedError("Driver needed to override this test")

    def test_None(self):
        con = self._connect()
        try:
            cur = con.cursor()
            self.executeDDL1(cur)
            cur.execute("insert into %sbooze values (NULL)" % self.table_prefix)
            cur.execute("select name from %sbooze" % self.table_prefix)
            r = cur.fetchall()
            self.assertEqual(len(r), 1)
            self.assertEqual(len(r[0]), 1)
            self.assertEqual(r[0][0], None, "NULL value not returned as None")
        finally:
            con.close()

    def test_Date(self):
        d1 = self.driver.Date(2002, 12, 25)
        d2 = self.driver.DateFromTicks(time.mktime((2002, 12, 25, 0, 0, 0, 0, 0, 0)))
        # Can we assume this? API doesn't specify, but it seems implied
        # self.assertEqual(str(d1),str(d2))

    def test_Time(self):
        t1 = self.driver.Time(13, 45, 30)
        t2 = self.driver.TimeFromTicks(time.mktime((2001, 1, 1, 13, 45, 30, 0, 0, 0)))
        # Can we assume this? API doesn't specify, but it seems implied
        # self.assertEqual(str(t1),str(t2))

    def test_Timestamp(self):
        t1 = self.driver.Timestamp(2002, 12, 25, 13, 45, 30)
        t2 = self.driver.TimestampFromTicks(
            time.mktime((2002, 12, 25, 13, 45, 30, 0, 0, 0))
        )
        # Can we assume this? API doesn't specify, but it seems implied
        # self.assertEqual(str(t1),str(t2))

    def test_Binary(self):
        b = self.driver.Binary(str2bytes("Something"))
        b = self.driver.Binary(str2bytes(""))

    def test_STRING(self):
        _failUnless(
            self, hasattr(self.driver, "STRING"), "module.STRING must be defined"
        )

    def test_BINARY(self):
        _failUnless(
            self, hasattr(self.driver, "BINARY"), "module.BINARY must be defined."
        )

    def test_NUMBER(self):
        _failUnless(
            self, hasattr(self.driver, "NUMBER"), "module.NUMBER must be defined."
        )

    def test_DATETIME(self):
        _failUnless(
            self, hasattr(self.driver, "DATETIME"), "module.DATETIME must be defined."
        )

    def test_ROWID(self):
        _failUnless(
            self, hasattr(self.driver, "ROWID"), "module.ROWID must be defined."
        )
