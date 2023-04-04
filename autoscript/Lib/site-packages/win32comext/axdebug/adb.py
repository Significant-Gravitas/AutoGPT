"""The glue between the Python debugger interface and the Active Debugger interface
"""
import _thread
import bdb
import os
import sys
import traceback

import pythoncom
import win32api
import win32com.client.connect
from win32com.axdebug.util import _wrap, _wrap_remove, trace
from win32com.server.util import unwrap

from . import axdebug, gateways, stackframe


def fnull(*args):
    pass


try:
    os.environ["DEBUG_AXDEBUG"]
    debugging = 1
except KeyError:
    debugging = 0

traceenter = fnull  # trace enter of functions
tracev = fnull  # verbose trace

if debugging:
    traceenter = trace  # trace enter of functions
    tracev = trace  # verbose trace


class OutputReflector:
    def __init__(self, file, writefunc):
        self.writefunc = writefunc
        self.file = file

    def __getattr__(self, name):
        return getattr(self.file, name)

    def write(self, message):
        self.writefunc(message)
        self.file.write(message)


def _dumpf(frame):
    if frame is None:
        return "<None>"
    else:
        addn = "(with trace!)"
        if frame.f_trace is None:
            addn = " **No Trace Set **"
        return "Frame at %d, file %s, line: %d%s" % (
            id(frame),
            frame.f_code.co_filename,
            frame.f_lineno,
            addn,
        )


g_adb = None


def OnSetBreakPoint(codeContext, breakPointState, lineNo):
    try:
        fileName = codeContext.codeContainer.GetFileName()
        # inject the code into linecache.
        import linecache

        linecache.cache[fileName] = 0, 0, codeContext.codeContainer.GetText(), fileName
        g_adb._OnSetBreakPoint(fileName, codeContext, breakPointState, lineNo + 1)
    except:
        traceback.print_exc()


class Adb(bdb.Bdb, gateways.RemoteDebugApplicationEvents):
    def __init__(self):
        self.debugApplication = None
        self.debuggingThread = None
        self.debuggingThreadStateHandle = None
        self.stackSnifferCookie = self.stackSniffer = None
        self.codeContainerProvider = None
        self.debuggingThread = None
        self.breakFlags = None
        self.breakReason = None
        self.appDebugger = None
        self.appEventConnection = None
        self.logicalbotframe = None  # Anything at this level or below does not exist!
        self.currentframe = None  # The frame we are currently in.
        self.recursiveData = []  # Data saved for each reentery on this thread.
        bdb.Bdb.__init__(self)
        self._threadprotectlock = _thread.allocate_lock()
        self.reset()

    def canonic(self, fname):
        if fname[0] == "<":
            return fname
        return bdb.Bdb.canonic(self, fname)

    def reset(self):
        traceenter("adb.reset")
        bdb.Bdb.reset(self)

    def __xxxxx__set_break(self, filename, lineno, cond=None):
        # As per standard one, except no linecache checking!
        if filename not in self.breaks:
            self.breaks[filename] = []
        list = self.breaks[filename]
        if lineno in list:
            return "There is already a breakpoint there!"
        list.append(lineno)
        if cond is not None:
            self.cbreaks[filename, lineno] = cond

    def stop_here(self, frame):
        traceenter("stop_here", _dumpf(frame), _dumpf(self.stopframe))
        # As per bdb.stop_here, except for logicalbotframe
        ##              if self.stopframe is None:
        ##                      return 1
        if frame is self.stopframe:
            return 1

        tracev("stop_here said 'No'!")
        return 0

    def break_here(self, frame):
        traceenter("break_here", self.breakFlags, _dumpf(frame))
        self.breakReason = None
        if self.breakFlags == axdebug.APPBREAKFLAG_DEBUGGER_HALT:
            self.breakReason = axdebug.BREAKREASON_DEBUGGER_HALT
        elif self.breakFlags == axdebug.APPBREAKFLAG_DEBUGGER_BLOCK:
            self.breakReason = axdebug.BREAKREASON_DEBUGGER_BLOCK
        elif self.breakFlags == axdebug.APPBREAKFLAG_STEP:
            self.breakReason = axdebug.BREAKREASON_STEP
        else:
            print("Calling base 'break_here' with", self.breaks)
            if bdb.Bdb.break_here(self, frame):
                self.breakReason = axdebug.BREAKREASON_BREAKPOINT
        return self.breakReason is not None

    def break_anywhere(self, frame):
        traceenter("break_anywhere", _dumpf(frame))
        if self.breakFlags == axdebug.APPBREAKFLAG_DEBUGGER_HALT:
            self.breakReason = axdebug.BREAKREASON_DEBUGGER_HALT
            return 1
        rc = bdb.Bdb.break_anywhere(self, frame)
        tracev("break_anywhere", _dumpf(frame), "returning", rc)
        return rc

    def dispatch_return(self, frame, arg):
        traceenter("dispatch_return", _dumpf(frame), arg)
        if self.logicalbotframe is frame:
            # We dont want to debug parent frames.
            tracev("dispatch_return resetting sys.trace")
            sys.settrace(None)
            return
        #                       self.bSetTrace = 0
        self.currentframe = frame.f_back
        return bdb.Bdb.dispatch_return(self, frame, arg)

    def dispatch_line(self, frame):
        traceenter("dispatch_line", _dumpf(frame), _dumpf(self.botframe))
        #               trace("logbotframe is", _dumpf(self.logicalbotframe), "botframe is", self.botframe)
        if frame is self.logicalbotframe:
            trace("dispatch_line", _dumpf(frame), "for bottom frame returing tracer")
            # The next code executed in the frame above may be a builtin (eg, apply())
            # in which sys.trace needs to be set.
            sys.settrace(self.trace_dispatch)
            # And return the tracer incase we are about to execute Python code,
            # in which case sys tracer is ignored!
            return self.trace_dispatch

        if self.codeContainerProvider.FromFileName(frame.f_code.co_filename) is None:
            trace(
                "dispatch_line has no document for", _dumpf(frame), "- skipping trace!"
            )
            return None
        self.currentframe = (
            frame  # So the stack sniffer knows our most recent, debuggable code.
        )
        return bdb.Bdb.dispatch_line(self, frame)

    def dispatch_call(self, frame, arg):
        traceenter("dispatch_call", _dumpf(frame))
        frame.f_locals["__axstack_address__"] = axdebug.GetStackAddress()
        if frame is self.botframe:
            trace("dispatch_call is self.botframe - returning tracer")
            return self.trace_dispatch
        # Not our bottom frame.  If we have a document for it,
        # then trace it, otherwise run at full speed.
        if self.codeContainerProvider.FromFileName(frame.f_code.co_filename) is None:
            trace(
                "dispatch_call has no document for", _dumpf(frame), "- skipping trace!"
            )
            ##                      sys.settrace(None)
            return None
        return self.trace_dispatch

    #               rc =  bdb.Bdb.dispatch_call(self, frame, arg)
    #               trace("dispatch_call", _dumpf(frame),"returned",rc)
    #               return rc

    def trace_dispatch(self, frame, event, arg):
        traceenter("trace_dispatch", _dumpf(frame), event, arg)
        if self.debugApplication is None:
            trace("trace_dispatch has no application!")
            return  # None
        return bdb.Bdb.trace_dispatch(self, frame, event, arg)

    #
    # The user functions do bugger all!
    #
    #       def user_call(self, frame, argument_list):
    #               traceenter("user_call",_dumpf(frame))

    def user_line(self, frame):
        traceenter("user_line", _dumpf(frame))
        # Traces at line zero
        if frame.f_lineno != 0:
            breakReason = self.breakReason
            if breakReason is None:
                breakReason = axdebug.BREAKREASON_STEP
            self._HandleBreakPoint(frame, None, breakReason)

    def user_return(self, frame, return_value):
        #               traceenter("user_return",_dumpf(frame),return_value)
        bdb.Bdb.user_return(self, frame, return_value)

    def user_exception(self, frame, exc_info):
        #               traceenter("user_exception")
        bdb.Bdb.user_exception(self, frame, exc_info)

    def _HandleBreakPoint(self, frame, tb, reason):
        traceenter(
            "Calling HandleBreakPoint with reason", reason, "at frame", _dumpf(frame)
        )
        traceenter(" Current frame is", _dumpf(self.currentframe))
        try:
            resumeAction = self.debugApplication.HandleBreakPoint(reason)
            tracev("HandleBreakPoint returned with ", resumeAction)
        except pythoncom.com_error as details:
            # Eeek - the debugger is dead, or something serious is happening.
            # Assume we should continue
            resumeAction = axdebug.BREAKRESUMEACTION_CONTINUE
            trace("HandleBreakPoint FAILED with", details)

        self.stack = []
        self.curindex = 0
        if resumeAction == axdebug.BREAKRESUMEACTION_ABORT:
            self.set_quit()
        elif resumeAction == axdebug.BREAKRESUMEACTION_CONTINUE:
            tracev("resume action is continue")
            self.set_continue()
        elif resumeAction == axdebug.BREAKRESUMEACTION_STEP_INTO:
            tracev("resume action is step")
            self.set_step()
        elif resumeAction == axdebug.BREAKRESUMEACTION_STEP_OVER:
            tracev("resume action is next")
            self.set_next(frame)
        elif resumeAction == axdebug.BREAKRESUMEACTION_STEP_OUT:
            tracev("resume action is stop out")
            self.set_return(frame)
        else:
            raise ValueError("unknown resume action flags")
        self.breakReason = None

    def set_trace(self):
        self.breakReason = axdebug.BREAKREASON_LANGUAGE_INITIATED
        bdb.Bdb.set_trace(self)

    def CloseApp(self):
        traceenter("ClosingApp")
        self.reset()
        self.logicalbotframe = None
        if self.stackSnifferCookie is not None:
            try:
                self.debugApplication.RemoveStackFrameSniffer(self.stackSnifferCookie)

            except pythoncom.com_error:
                trace(
                    "*** Could not RemoveStackFrameSniffer %d"
                    % (self.stackSnifferCookie)
                )
        if self.stackSniffer:
            _wrap_remove(self.stackSniffer)
        self.stackSnifferCookie = self.stackSniffer = None

        if self.appEventConnection is not None:
            self.appEventConnection.Disconnect()
            self.appEventConnection = None
        self.debugApplication = None
        self.appDebugger = None
        if self.codeContainerProvider is not None:
            self.codeContainerProvider.Close()
            self.codeContainerProvider = None

    def AttachApp(self, debugApplication, codeContainerProvider):
        #               traceenter("AttachApp", debugApplication, codeContainerProvider)
        self.codeContainerProvider = codeContainerProvider
        self.debugApplication = debugApplication
        self.stackSniffer = _wrap(
            stackframe.DebugStackFrameSniffer(self), axdebug.IID_IDebugStackFrameSniffer
        )
        self.stackSnifferCookie = debugApplication.AddStackFrameSniffer(
            self.stackSniffer
        )
        #               trace("StackFrameSniffer added (%d)" % self.stackSnifferCookie)

        # Connect to the application events.
        self.appEventConnection = win32com.client.connect.SimpleConnection(
            self.debugApplication, self, axdebug.IID_IRemoteDebugApplicationEvents
        )

    def ResetAXDebugging(self):
        traceenter("ResetAXDebugging", self, "with refcount", len(self.recursiveData))
        if win32api.GetCurrentThreadId() != self.debuggingThread:
            trace("ResetAXDebugging called on other thread")
            return

        if len(self.recursiveData) == 0:
            #                       print "ResetAXDebugging called for final time."
            self.logicalbotframe = None
            self.debuggingThread = None
            self.currentframe = None
            self.debuggingThreadStateHandle = None
            return

        (
            self.logbotframe,
            self.stopframe,
            self.currentframe,
            self.debuggingThreadStateHandle,
        ) = self.recursiveData[0]
        self.recursiveData = self.recursiveData[1:]

    def SetupAXDebugging(self, baseFrame=None, userFrame=None):
        """Get ready for potential debugging.  Must be called on the thread
        that is being debugged.
        """
        # userFrame is for non AXScript debugging.  This is the first frame of the
        # users code.
        if userFrame is None:
            userFrame = baseFrame
        else:
            # We have missed the "dispatch_call" function, so set this up now!
            userFrame.f_locals["__axstack_address__"] = axdebug.GetStackAddress()

        traceenter("SetupAXDebugging", self)
        self._threadprotectlock.acquire()
        try:
            thisThread = win32api.GetCurrentThreadId()
            if self.debuggingThread is None:
                self.debuggingThread = thisThread
            else:
                if self.debuggingThread != thisThread:
                    trace("SetupAXDebugging called on other thread - ignored!")
                    return
                # push our context.
                self.recursiveData.insert(
                    0,
                    (
                        self.logicalbotframe,
                        self.stopframe,
                        self.currentframe,
                        self.debuggingThreadStateHandle,
                    ),
                )
        finally:
            self._threadprotectlock.release()

        trace("SetupAXDebugging has base frame as", _dumpf(baseFrame))
        self.botframe = baseFrame
        self.stopframe = userFrame
        self.logicalbotframe = baseFrame
        self.currentframe = None
        self.debuggingThreadStateHandle = axdebug.GetThreadStateHandle()

        self._BreakFlagsChanged()

    # RemoteDebugApplicationEvents
    def OnConnectDebugger(self, appDebugger):
        traceenter("OnConnectDebugger", appDebugger)
        self.appDebugger = appDebugger
        # Reflect output to appDebugger
        writefunc = lambda s: appDebugger.onDebugOutput(s)
        sys.stdout = OutputReflector(sys.stdout, writefunc)
        sys.stderr = OutputReflector(sys.stderr, writefunc)

    def OnDisconnectDebugger(self):
        traceenter("OnDisconnectDebugger")
        # Stop reflecting output
        if isinstance(sys.stdout, OutputReflector):
            sys.stdout = sys.stdout.file
        if isinstance(sys.stderr, OutputReflector):
            sys.stderr = sys.stderr.file
        self.appDebugger = None
        self.set_quit()

    def OnSetName(self, name):
        traceenter("OnSetName", name)

    def OnDebugOutput(self, string):
        traceenter("OnDebugOutput", string)

    def OnClose(self):
        traceenter("OnClose")

    def OnEnterBreakPoint(self, rdat):
        traceenter("OnEnterBreakPoint", rdat)

    def OnLeaveBreakPoint(self, rdat):
        traceenter("OnLeaveBreakPoint", rdat)

    def OnCreateThread(self, rdat):
        traceenter("OnCreateThread", rdat)

    def OnDestroyThread(self, rdat):
        traceenter("OnDestroyThread", rdat)

    def OnBreakFlagChange(self, abf, rdat):
        traceenter("Debugger OnBreakFlagChange", abf, rdat)
        self.breakFlags = abf
        self._BreakFlagsChanged()

    def _BreakFlagsChanged(self):
        traceenter(
            "_BreakFlagsChanged to %s with our thread = %s, and debugging thread = %s"
            % (self.breakFlags, self.debuggingThread, win32api.GetCurrentThreadId())
        )
        trace("_BreakFlagsChanged has breaks", self.breaks)
        # If a request comes on our debugging thread, then do it now!
        #               if self.debuggingThread!=win32api.GetCurrentThreadId():
        #                       return

        if len(self.breaks) or self.breakFlags:
            if self.logicalbotframe:
                trace("BreakFlagsChange with bot frame", _dumpf(self.logicalbotframe))
                # We have frames not to be debugged (eg, Scripting engine frames
                # (sys.settrace will be set when out logicalbotframe is hit -
                #  this may not be the right thing to do, as it may not cause the
                #  immediate break we desire.)
                self.logicalbotframe.f_trace = self.trace_dispatch
            else:
                trace("BreakFlagsChanged, but no bottom frame")
                if self.stopframe is not None:
                    self.stopframe.f_trace = self.trace_dispatch
            # If we have the thread-state for the thread being debugged, then
            # we dynamically set its trace function - it is possible that the thread
            # being debugged is in a blocked call (eg, a message box) and we
            # want to hit the debugger the instant we return
        if (
            self.debuggingThreadStateHandle is not None
            and self.breakFlags
            and self.debuggingThread != win32api.GetCurrentThreadId()
        ):
            axdebug.SetThreadStateTrace(
                self.debuggingThreadStateHandle, self.trace_dispatch
            )

    def _OnSetBreakPoint(self, key, codeContext, bps, lineNo):
        traceenter("_OnSetBreakPoint", self, key, codeContext, bps, lineNo)
        if bps == axdebug.BREAKPOINT_ENABLED:
            problem = self.set_break(key, lineNo)
            if problem:
                print("*** set_break failed -", problem)
            trace("_OnSetBreakPoint just set BP and has breaks", self.breaks)
        else:
            self.clear_break(key, lineNo)
        self._BreakFlagsChanged()
        trace("_OnSetBreakPoint leaving with breaks", self.breaks)


def Debugger():
    global g_adb
    if g_adb is None:
        g_adb = Adb()
    return g_adb
