"""Excel IRTDServer implementation.

This module is a functional example of how to implement the IRTDServer interface
in python, using the pywin32 extensions. Further details, about this interface
and it can be found at:
     http://msdn.microsoft.com/library/default.asp?url=/library/en-us/dnexcl2k2/html/odc_xlrtdfaq.asp
"""

# Copyright (c) 2003-2004 by Chris Nilsson <chris@slort.org>
#
# By obtaining, using, and/or copying this software and/or its
# associated documentation, you agree that you have read, understood,
# and will comply with the following terms and conditions:
#
# Permission to use, copy, modify, and distribute this software and
# its associated documentation for any purpose and without fee is
# hereby granted, provided that the above copyright notice appears in
# all copies, and that both that copyright notice and this permission
# notice appear in supporting documentation, and that the name of
# Christopher Nilsson (the author) not be used in advertising or publicity
# pertaining to distribution of the software without specific, written
# prior permission.
#
# THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD
# TO THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANT-
# ABILITY AND FITNESS.  IN NO EVENT SHALL THE AUTHOR
# BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THIS SOFTWARE.

import datetime  # For the example classes...
import threading

import pythoncom
import win32com.client
from win32com import universal
from win32com.client import gencache
from win32com.server.exception import COMException

# Typelib info for version 10 - aka Excel XP.
# This is the minimum version of excel that we can work with as this is when
# Microsoft introduced these interfaces.
EXCEL_TLB_GUID = "{00020813-0000-0000-C000-000000000046}"
EXCEL_TLB_LCID = 0
EXCEL_TLB_MAJOR = 1
EXCEL_TLB_MINOR = 4

# Import the excel typelib to make sure we've got early-binding going on.
# The "ByRef" parameters we use later won't work without this.
gencache.EnsureModule(EXCEL_TLB_GUID, EXCEL_TLB_LCID, EXCEL_TLB_MAJOR, EXCEL_TLB_MINOR)

# Tell pywin to import these extra interfaces.
# --
# QUESTION: Why? The interfaces seem to descend from IDispatch, so
# I'd have thought, for example, calling callback.UpdateNotify() (on the
# IRTDUpdateEvent callback excel gives us) would work without molestation.
# But the callback needs to be cast to a "real" IRTDUpdateEvent type. Hmm...
# This is where my small knowledge of the pywin framework / COM gets hazy.
# --
# Again, we feed in the Excel typelib as the source of these interfaces.
universal.RegisterInterfaces(
    EXCEL_TLB_GUID,
    EXCEL_TLB_LCID,
    EXCEL_TLB_MAJOR,
    EXCEL_TLB_MINOR,
    ["IRtdServer", "IRTDUpdateEvent"],
)


class ExcelRTDServer(object):
    """Base RTDServer class.

    Provides most of the features needed to implement the IRtdServer interface.
    Manages topic adding, removal, and packing up the values for excel.

    Shouldn't be instanciated directly.

    Instead, descendant classes should override the CreateTopic() method.
    Topic objects only need to provide a GetValue() function to play nice here.
    The values given need to be atomic (eg. string, int, float... etc).

    Also note: nothing has been done within this class to ensure that we get
    time to check our topics for updates. I've left that up to the subclass
    since the ways, and needs, of refreshing your topics will vary greatly. For
    example, the sample implementation uses a timer thread to wake itself up.
    Whichever way you choose to do it, your class needs to be able to wake up
    occaisionally, since excel will never call your class without being asked to
    first.

    Excel will communicate with our object in this order:
      1. Excel instanciates our object and calls ServerStart, providing us with
         an IRTDUpdateEvent callback object.
      2. Excel calls ConnectData when it wants to subscribe to a new "topic".
      3. When we have new data to provide, we call the UpdateNotify method of the
         callback object we were given.
      4. Excel calls our RefreshData method, and receives a 2d SafeArray (row-major)
         containing the Topic ids in the 1st dim, and the topic values in the
         2nd dim.
      5. When not needed anymore, Excel will call our DisconnectData to
         unsubscribe from a topic.
      6. When there are no more topics left, Excel will call our ServerTerminate
         method to kill us.

    Throughout, at undetermined periods, Excel will call our Heartbeat
    method to see if we're still alive. It must return a non-zero value, or
    we'll be killed.

    NOTE: By default, excel will at most call RefreshData once every 2 seconds.
          This is a setting that needs to be changed excel-side. To change this,
          you can set the throttle interval like this in the excel VBA object model:
            Application.RTD.ThrottleInterval = 1000 ' milliseconds
    """

    _com_interfaces_ = ["IRtdServer"]
    _public_methods_ = [
        "ConnectData",
        "DisconnectData",
        "Heartbeat",
        "RefreshData",
        "ServerStart",
        "ServerTerminate",
    ]
    _reg_clsctx_ = pythoncom.CLSCTX_INPROC_SERVER
    # _reg_clsid_ = "# subclass must provide this class attribute"
    # _reg_desc_ = "# subclass should provide this description"
    # _reg_progid_ = "# subclass must provide this class attribute"

    ALIVE = 1
    NOT_ALIVE = 0

    def __init__(self):
        """Constructor"""
        super(ExcelRTDServer, self).__init__()
        self.IsAlive = self.ALIVE
        self.__callback = None
        self.topics = {}

    def SignalExcel(self):
        """Use the callback we were given to tell excel new data is available."""
        if self.__callback is None:
            raise COMException(desc="Callback excel provided is Null")
        self.__callback.UpdateNotify()

    def ConnectData(self, TopicID, Strings, GetNewValues):
        """Creates a new topic out of the Strings excel gives us."""
        try:
            self.topics[TopicID] = self.CreateTopic(Strings)
        except Exception as why:
            raise COMException(desc=str(why))
        GetNewValues = True
        result = self.topics[TopicID]
        if result is None:
            result = "# %s: Waiting for update" % self.__class__.__name__
        else:
            result = result.GetValue()

        # fire out internal event...
        self.OnConnectData(TopicID)

        # GetNewValues as per interface is ByRef, so we need to pass it back too.
        return result, GetNewValues

    def DisconnectData(self, TopicID):
        """Deletes the given topic."""
        self.OnDisconnectData(TopicID)

        if TopicID in self.topics:
            self.topics[TopicID] = None
            del self.topics[TopicID]

    def Heartbeat(self):
        """Called by excel to see if we're still here."""
        return self.IsAlive

    def RefreshData(self, TopicCount):
        """Packs up the topic values. Called by excel when it's ready for an update.

        Needs to:
          * Return the current number of topics, via the "ByRef" TopicCount
          * Return a 2d SafeArray of the topic data.
            - 1st dim: topic numbers
            - 2nd dim: topic values

        We could do some caching, instead of repacking everytime...
        But this works for demonstration purposes."""
        TopicCount = len(self.topics)
        self.OnRefreshData()

        # Grow the lists, so we don't need a heap of calls to append()
        results = [[None] * TopicCount, [None] * TopicCount]

        # Excel expects a 2-dimensional array. The first dim contains the
        # topic numbers, and the second contains the values for the topics.
        # In true VBA style (yuck), we need to pack the array in row-major format,
        # which looks like:
        #   ( (topic_num1, topic_num2, ..., topic_numN), \
        #     (topic_val1, topic_val2, ..., topic_valN) )
        for idx, topicdata in enumerate(self.topics.items()):
            topicNum, topic = topicdata
            results[0][idx] = topicNum
            results[1][idx] = topic.GetValue()

        # TopicCount is meant to be passed to us ByRef, so return it as well, as per
        # the way pywin32 handles ByRef arguments.
        return tuple(results), TopicCount

    def ServerStart(self, CallbackObject):
        """Excel has just created us... We take its callback for later, and set up shop."""
        self.IsAlive = self.ALIVE

        if CallbackObject is None:
            raise COMException(desc="Excel did not provide a callback")

        # Need to "cast" the raw PyIDispatch object to the IRTDUpdateEvent interface
        IRTDUpdateEventKlass = win32com.client.CLSIDToClass.GetClass(
            "{A43788C1-D91B-11D3-8F39-00C04F3651B8}"
        )
        self.__callback = IRTDUpdateEventKlass(CallbackObject)

        self.OnServerStart()

        return self.IsAlive

    def ServerTerminate(self):
        """Called when excel no longer wants us."""
        self.IsAlive = self.NOT_ALIVE  # On next heartbeat, excel will free us
        self.OnServerTerminate()

    def CreateTopic(self, TopicStrings=None):
        """Topic factory method. Subclass must override.

        Topic objects need to provide:
          * GetValue() method which returns an atomic value.

        Will raise NotImplemented if not overridden.
        """
        raise NotImplemented("Subclass must implement")

    # Overridable class events...
    def OnConnectData(self, TopicID):
        """Called when a new topic has been created, at excel's request."""
        pass

    def OnDisconnectData(self, TopicID):
        """Called when a topic is about to be deleted, at excel's request."""
        pass

    def OnRefreshData(self):
        """Called when excel has requested all current topic data."""
        pass

    def OnServerStart(self):
        """Called when excel has instanciated us."""
        pass

    def OnServerTerminate(self):
        """Called when excel is about to destroy us."""
        pass


class RTDTopic(object):
    """Base RTD Topic.
    Only method required by our RTDServer implementation is GetValue().
    The others are more for convenience."""

    def __init__(self, TopicStrings):
        super(RTDTopic, self).__init__()
        self.TopicStrings = TopicStrings
        self.__currentValue = None
        self.__dirty = False

    def Update(self, sender):
        """Called by the RTD Server.
        Gives us a chance to check if our topic data needs to be
        changed (eg. check a file, quiz a database, etc)."""
        raise NotImplemented("subclass must implement")

    def Reset(self):
        """Call when this topic isn't considered "dirty" anymore."""
        self.__dirty = False

    def GetValue(self):
        return self.__currentValue

    def SetValue(self, value):
        self.__dirty = True
        self.__currentValue = value

    def HasChanged(self):
        return self.__dirty


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

######################################
# Example classes
######################################


class TimeServer(ExcelRTDServer):
    """Example Time RTD server.

    Sends time updates back to excel.

    example of use, in an excel sheet:
      =RTD("Python.RTD.TimeServer","","seconds","5")

    This will cause a timestamp string to fill the cell, and update its value
    every 5 seconds (or as close as possible depending on how busy excel is).

    The empty string parameter denotes the com server is running on the local
    machine. Otherwise, put in the hostname to look on. For more info
    on this, lookup the Excel help for its "RTD" worksheet function.

    Obviously, you'd want to wrap this kind of thing in a friendlier VBA
    function.

    Also, remember that the RTD function accepts a maximum of 28 arguments!
    If you want to pass more, you may need to concatenate arguments into one
    string, and have your topic parse them appropriately.
    """

    # win32com.server setup attributes...
    # Never copy the _reg_clsid_ value in your own classes!
    _reg_clsid_ = "{EA7F2CF1-11A2-45E4-B2D5-68E240DB8CB1}"
    _reg_progid_ = "Python.RTD.TimeServer"
    _reg_desc_ = "Python class implementing Excel IRTDServer -- feeds time"

    # other class attributes...
    INTERVAL = 0.5  # secs. Threaded timer will wake us up at this interval.

    def __init__(self):
        super(TimeServer, self).__init__()

        # Simply timer thread to ensure we get to update our topics, and
        # tell excel about any changes. This is a pretty basic and dirty way to
        # do this. Ideally, there should be some sort of waitable (eg. either win32
        # event, socket data event...) and be kicked off by that event triggering.
        # As soon as we set up shop here, we _must_ return control back to excel.
        # (ie. we can't block and do our own thing...)
        self.ticker = threading.Timer(self.INTERVAL, self.Update)

    def OnServerStart(self):
        self.ticker.start()

    def OnServerTerminate(self):
        if not self.ticker.finished.isSet():
            self.ticker.cancel()  # Cancel our wake-up thread. Excel has killed us.

    def Update(self):
        # Get our wake-up thread ready...
        self.ticker = threading.Timer(self.INTERVAL, self.Update)
        try:
            # Check if any of our topics have new info to pass on
            if len(self.topics):
                refresh = False
                for topic in self.topics.values():
                    topic.Update(self)
                    if topic.HasChanged():
                        refresh = True
                    topic.Reset()

                if refresh:
                    self.SignalExcel()
        finally:
            self.ticker.start()  # Make sure we get to run again

    def CreateTopic(self, TopicStrings=None):
        """Topic factory. Builds a TimeTopic object out of the given TopicStrings."""
        return TimeTopic(TopicStrings)


class TimeTopic(RTDTopic):
    """Example topic for example RTD server.

    Will accept some simple commands to alter how long to delay value updates.

    Commands:
      * seconds, delay_in_seconds
      * minutes, delay_in_minutes
      * hours, delay_in_hours
    """

    def __init__(self, TopicStrings):
        super(TimeTopic, self).__init__(TopicStrings)
        try:
            self.cmd, self.delay = self.TopicStrings
        except Exception as E:
            # We could simply return a "# ERROR" type string as the
            # topic value, but explosions like this should be able to get handled by
            # the VBA-side "On Error" stuff.
            raise ValueError("Invalid topic strings: %s" % str(TopicStrings))

        # self.cmd = str(self.cmd)
        self.delay = float(self.delay)

        # setup our initial value
        self.checkpoint = self.timestamp()
        self.SetValue(str(self.checkpoint))

    def timestamp(self):
        return datetime.datetime.now()

    def Update(self, sender):
        now = self.timestamp()
        delta = now - self.checkpoint
        refresh = False
        if self.cmd == "seconds":
            if delta.seconds >= self.delay:
                refresh = True
        elif self.cmd == "minutes":
            if delta.minutes >= self.delay:
                refresh = True
        elif self.cmd == "hours":
            if delta.hours >= self.delay:
                refresh = True
        else:
            self.SetValue("#Unknown command: " + self.cmd)

        if refresh:
            self.SetValue(str(now))
            self.checkpoint = now


if __name__ == "__main__":
    import win32com.server.register

    # Register/Unregister TimeServer example
    # eg. at the command line: excelrtd.py --register
    # Then type in an excel cell something like:
    # =RTD("Python.RTD.TimeServer","","seconds","5")
    win32com.server.register.UseCommandLine(TimeServer)
