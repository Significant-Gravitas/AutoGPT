import os
import time

import pythoncom
from win32com.client import Dispatch, DispatchWithEvents, constants

finished = 0  # Flag for the wait loop from (3) to test


class ADOEvents:  # event handler class
    def OnWillConnect(self, str, user, pw, opt, sts, cn):
        # Must have this event, as if it is not handled, ADO assumes the
        # operation is cancelled, and raises an error (Operation cancelled
        # by the user)
        pass

    def OnConnectComplete(self, error, status, connection):
        # Assume no errors, until we have the basic stuff
        # working. Now, "connection" should be an open
        # connection to my data source
        # Do the "something" from (2). For now, just
        # print the connection data source
        print("connection is", connection)
        print("Connected to", connection.Properties("Data Source"))
        # OK, our work is done. Let the main loop know
        global finished
        finished = 1

    def OnCommitTransComplete(self, pError, adStatus, pConnection):
        pass

    def OnInfoMessage(self, pError, adStatus, pConnection):
        pass

    def OnDisconnect(self, adStatus, pConnection):
        pass

    def OnBeginTransComplete(self, TransactionLevel, pError, adStatus, pConnection):
        pass

    def OnRollbackTransComplete(self, pError, adStatus, pConnection):
        pass

    def OnExecuteComplete(
        self, RecordsAffected, pError, adStatus, pCommand, pRecordset, pConnection
    ):
        pass

    def OnWillExecute(
        self,
        Source,
        CursorType,
        LockType,
        Options,
        adStatus,
        pCommand,
        pRecordset,
        pConnection,
    ):
        pass


def TestConnection(dbname):
    # Create the ADO connection object, and link the event
    # handlers into it
    c = DispatchWithEvents("ADODB.Connection", ADOEvents)

    # Initiate the asynchronous open
    dsn = "Driver={Microsoft Access Driver (*.mdb)};Dbq=%s" % dbname
    user = "system"
    pw = "manager"
    c.Open(dsn, user, pw, constants.adAsyncConnect)

    # Sit in a loop, until our event handler (above) sets the
    # "finished" flag or we time out.
    end_time = time.clock() + 10
    while time.clock() < end_time:
        # Pump messages so that COM gets a look in
        pythoncom.PumpWaitingMessages()
    if not finished:
        print("XXX - Failed to connect!")


def Test():
    from . import testAccess

    try:
        testAccess.GenerateSupport()
    except pythoncom.com_error:
        print("*** Can not import the MSAccess type libraries - tests skipped")
        return
    dbname = testAccess.CreateTestAccessDatabase()
    try:
        TestConnection(dbname)
    finally:
        os.unlink(dbname)


if __name__ == "__main__":
    Test()
