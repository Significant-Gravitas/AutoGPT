"""
Performance Data Helper (PDH) Query Classes

Wrapper classes for end-users and high-level access to the PDH query
mechanisms.  PDH is a win32-specific mechanism for accessing the
performance data made available by the system.  The Python for Windows
PDH module does not implement the "Registry" interface, implementing
the more straightforward Query-based mechanism.

The basic idea of a PDH Query is an object which can query the system
about the status of any number of "counters."  The counters are paths
to a particular piece of performance data.  For instance, the path 
'\\Memory\\Available Bytes' describes just about exactly what it says
it does, the amount of free memory on the default computer expressed 
in Bytes.  These paths can be considerably more complex than this, 
but part of the point of this wrapper module is to hide that
complexity from the end-user/programmer.

EXAMPLE: A more complex Path
	'\\\\RAISTLIN\\PhysicalDisk(_Total)\\Avg. Disk Bytes/Read'
	Raistlin --> Computer Name
	PhysicalDisk --> Object Name
	_Total --> The particular Instance (in this case, all instances, i.e. all drives)
	Avg. Disk Bytes/Read --> The piece of data being monitored.

EXAMPLE: Collecting Data with a Query
	As an example, the following code implements a logger which allows the
	user to choose what counters they would like to log, and logs those
	counters for 30 seconds, at two-second intervals.
	
	query = Query()
	query.addcounterbybrowsing()
	query.collectdatafor(30,2)
	
	The data is now stored in a list of lists as:
	query.curresults
	
	The counters(paths) which were used to collect the data are:
	query.curpaths
	
	You can use the win32pdh.ParseCounterPath(path) utility function
	to turn the paths into more easily read values for your task, or
	write the data to a file, or do whatever you want with it.

OTHER NOTABLE METHODS:
	query.collectdatawhile(period) # start a logging thread for collecting data
	query.collectdatawhile_stop() # signal the logging thread to stop logging
	query.collectdata() # run the query only once
	query.addperfcounter(object, counter, machine=None) # add a standard performance counter
	query.addinstcounter(object, counter,machine=None,objtype = 'Process',volatile=1,format = win32pdh.PDH_FMT_LONG) # add a possibly volatile counter

### Known bugs and limitations ###
Due to a problem with threading under the PythonWin interpreter, there
will be no data logged if the PythonWin window is not the foreground
application.  Workaround: scripts using threading should be run in the
python.exe interpreter.

The volatile-counter handlers are possibly buggy, they haven't been
tested to any extent.  The wrapper Query makes it safe to pass invalid
paths (a -1 will be returned, or the Query will be totally ignored,
depending on the missing element), so you should be able to work around
the error by including all possible paths and filtering out the -1's.

There is no way I know of to stop a thread which is currently sleeping,
so you have to wait until the thread in collectdatawhile is activated
again.  This might become a problem in situations where the collection
period is multiple minutes (or hours, or whatever).

Should make the win32pdh.ParseCounter function available to the Query
classes as a method or something similar, so that it can be accessed
by programmes that have just picked up an instance from somewhere.

Should explicitly mention where QueryErrors can be raised, and create a
full test set to see if there are any uncaught win32api.error's still
hanging around.

When using the python.exe interpreter, the addcounterbybrowsing-
generated browser window is often hidden behind other windows.  No known
workaround other than Alt-tabing to reach the browser window.

### Other References ###
The win32pdhutil module (which should be in the %pythonroot%/win32/lib 
directory) provides quick-and-dirty utilities for one-off access to
variables from the PDH.  Almost everything in that module can be done
with a Query object, but it provides task-oriented functions for a
number of common one-off tasks.

If you can access the MS Developers Network Library, you can find
information about the PDH API as MS describes it.  For a background article,
try:
http://msdn.microsoft.com/library/en-us/dnperfmo/html/msdn_pdhlib.asp

The reference guide for the PDH API was last spotted at:
http://msdn.microsoft.com/library/en-us/perfmon/base/using_the_pdh_interface.asp


In general the Python version of the API is just a wrapper around the
Query-based version of this API (as far as I can see), so you can learn what
you need to from there.  From what I understand, the MSDN Online 
resources are available for the price of signing up for them.  I can't
guarantee how long that's supposed to last. (Or anything for that
matter).
http://premium.microsoft.com/isapi/devonly/prodinfo/msdnprod/msdnlib.idc?theURL=/msdn/library/sdkdoc/perfdata_4982.htm

The eventual plan is for my (Mike Fletcher's) Starship account to include
a section on NT Administration, and the Query is the first project
in this plan.  There should be an article describing the creation of
a simple logger there, but the example above is 90% of the work of
that project, so don't sweat it if you don't find anything there.
(currently the account hasn't been set up).
http://starship.skyport.net/crew/mcfletch/

If you need to contact me immediately, (why I can't imagine), you can
email me at mcfletch@golden.net, or just post your question to the
Python newsgroup with a catchy subject line.
news:comp.lang.python

### Other Stuff ###
The Query classes are by Mike Fletcher, with the working code
being corruptions of Mark Hammonds win32pdhutil module.

Use at your own risk, no warranties, no guarantees, no assurances,
if you use it, you accept the risk of using it, etceteras.

"""
# Feb 12, 98 - MH added "rawaddcounter" so caller can get exception details.

import _thread
import copy
import time

import win32api
import win32pdh


class BaseQuery:
    """
    Provides wrapped access to the Performance Data Helper query
    objects, generally you should use the child class Query
    unless you have need of doing weird things :)

    This class supports two major working paradigms.  In the first,
    you open the query, and run it as many times as you need, closing
    the query when you're done with it.  This is suitable for static
    queries (ones where processes being monitored don't disappear).

    In the second, you allow the query to be opened each time and
    closed afterward.  This causes the base query object to be
    destroyed after each call.  Suitable for dynamic queries (ones
    which watch processes which might be closed while watching.)
    """

    def __init__(self, paths=None):
        """
        The PDH Query object is initialised with a single, optional
        list argument, that must be properly formatted PDH Counter
        paths.  Generally this list will only be provided by the class
        when it is being unpickled (removed from storage).  Normal
        use is to call the class with no arguments and use the various
        addcounter functions (particularly, for end user's, the use of
        addcounterbybrowsing is the most common approach)  You might
        want to provide the list directly if you want to hard-code the
        elements with which your query deals (and thereby avoid the
        overhead of unpickling the class).
        """
        self.counters = []
        if paths:
            self.paths = paths
        else:
            self.paths = []
        self._base = None
        self.active = 0
        self.curpaths = []

    def addcounterbybrowsing(
        self, flags=win32pdh.PERF_DETAIL_WIZARD, windowtitle="Python Browser"
    ):
        """
        Adds possibly multiple paths to the paths attribute of the query,
        does this by calling the standard counter browsing dialogue.  Within
        this dialogue, find the counter you want to log, and click: Add,
        repeat for every path you want to log, then click on close.  The
        paths are appended to the non-volatile paths list for this class,
        subclasses may create a function which parses the paths and decides
        (via heuristics) whether to add the path to the volatile or non-volatile
        path list.
        e.g.:
                query.addcounter()
        """
        win32pdh.BrowseCounters(None, 0, self.paths.append, flags, windowtitle)

    def rawaddcounter(self, object, counter, instance=None, inum=-1, machine=None):
        """
        Adds a single counter path, without catching any exceptions.

        See addcounter for details.
        """
        path = win32pdh.MakeCounterPath(
            (machine, object, instance, None, inum, counter)
        )
        self.paths.append(path)

    def addcounter(self, object, counter, instance=None, inum=-1, machine=None):
        """
        Adds a single counter path to the paths attribute.  Normally
        this will be called by a child class' speciality functions,
        rather than being called directly by the user. (Though it isn't
        hard to call manually, since almost everything is given a default)
        This method is only functional when the query is closed (or hasn't
        yet been opened).  This is to prevent conflict in multi-threaded
        query applications).
        e.g.:
                query.addcounter('Memory','Available Bytes')
        """
        if not self.active:
            try:
                self.rawaddcounter(object, counter, instance, inum, machine)
                return 0
            except win32api.error:
                return -1
        else:
            return -1

    def open(self):
        """
        Build the base query object for this wrapper,
        then add all of the counters required for the query.
        Raise a QueryError if we can't complete the functions.
        If we are already open, then do nothing.
        """
        if not self.active:  # to prevent having multiple open queries
            # curpaths are made accessible here because of the possibility of volatile paths
            # which may be dynamically altered by subclasses.
            self.curpaths = copy.copy(self.paths)
            try:
                base = win32pdh.OpenQuery()
                for path in self.paths:
                    try:
                        self.counters.append(win32pdh.AddCounter(base, path))
                    except win32api.error:  # we passed a bad path
                        self.counters.append(0)
                        pass
                self._base = base
                self.active = 1
                return 0  # open succeeded
            except:  # if we encounter any errors, kill the Query
                try:
                    self.killbase(base)
                except NameError:  # failed in creating query
                    pass
                self.active = 0
                self.curpaths = []
                raise QueryError(self)
        return 1  # already open

    def killbase(self, base=None):
        """
        ### This is not a public method
        Mission critical function to kill the win32pdh objects held
        by this object.  User's should generally use the close method
        instead of this method, in case a sub-class has overridden
        close to provide some special functionality.
        """
        # Kill Pythonic references to the objects in this object's namespace
        self._base = None
        counters = self.counters
        self.counters = []
        # we don't kill the curpaths for convenience, this allows the
        # user to close a query and still access the last paths
        self.active = 0
        # Now call the delete functions on all of the objects
        try:
            map(win32pdh.RemoveCounter, counters)
        except:
            pass
        try:
            win32pdh.CloseQuery(base)
        except:
            pass
        del counters
        del base

    def close(self):
        """
        Makes certain that the underlying query object has been closed,
        and that all counters have been removed from it.  This is
        important for reference counting.
        You should only need to call close if you have previously called
        open.  The collectdata methods all can handle opening and
        closing the query.  Calling close multiple times is acceptable.
        """
        try:
            self.killbase(self._base)
        except AttributeError:
            self.killbase()

    __del__ = close

    def collectdata(self, format=win32pdh.PDH_FMT_LONG):
        """
        Returns the formatted current values for the Query
        """
        if self._base:  # we are currently open, don't change this
            return self.collectdataslave(format)
        else:  # need to open and then close the _base, should be used by one-offs and elements tracking application instances
            self.open()  # will raise QueryError if couldn't open the query
            temp = self.collectdataslave(format)
            self.close()  # will always close
            return temp

    def collectdataslave(self, format=win32pdh.PDH_FMT_LONG):
        """
        ### Not a public method
        Called only when the Query is known to be open, runs over
        the whole set of counters, appending results to the temp,
        returns the values as a list.
        """
        try:
            win32pdh.CollectQueryData(self._base)
            temp = []
            for counter in self.counters:
                ok = 0
                try:
                    if counter:
                        temp.append(
                            win32pdh.GetFormattedCounterValue(counter, format)[1]
                        )
                        ok = 1
                except win32api.error:
                    pass
                if not ok:
                    temp.append(-1)  # a better way to signal failure???
            return temp
        except (
            win32api.error
        ):  # will happen if, for instance, no counters are part of the query and we attempt to collect data for it.
            return [-1] * len(self.counters)

    # pickle functions
    def __getinitargs__(self):
        """
        ### Not a public method
        """
        return (self.paths,)


class Query(BaseQuery):
    """
    Performance Data Helper(PDH) Query object:

    Provides a wrapper around the native PDH query object which
    allows for query reuse, query storage, and general maintenance
    functions (adding counter paths in various ways being the most
    obvious ones).
    """

    def __init__(self, *args, **namedargs):
        """
        The PDH Query object is initialised with a single, optional
        list argument, that must be properly formatted PDH Counter
        paths.  Generally this list will only be provided by the class
        when it is being unpickled (removed from storage).  Normal
        use is to call the class with no arguments and use the various
        addcounter functions (particularly, for end user's, the use of
        addcounterbybrowsing is the most common approach)  You might
        want to provide the list directly if you want to hard-code the
        elements with which your query deals (and thereby avoid the
        overhead of unpickling the class).
        """
        self.volatilecounters = []
        BaseQuery.__init__(*(self,) + args, **namedargs)

    def addperfcounter(self, object, counter, machine=None):
        """
        A "Performance Counter" is a stable, known, common counter,
        such as Memory, or Processor.  The use of addperfcounter by
        end-users is deprecated, since the use of
        addcounterbybrowsing is considerably more flexible and general.
        It is provided here to allow the easy development of scripts
        which need to access variables so common we know them by name
        (such as Memory|Available Bytes), and to provide symmetry with
        the add inst counter method.
        usage:
                query.addperfcounter('Memory', 'Available Bytes')
        It is just as easy to access addcounter directly, the following
        has an identicle effect.
                query.addcounter('Memory', 'Available Bytes')
        """
        BaseQuery.addcounter(self, object=object, counter=counter, machine=machine)

    def addinstcounter(
        self,
        object,
        counter,
        machine=None,
        objtype="Process",
        volatile=1,
        format=win32pdh.PDH_FMT_LONG,
    ):
        """
        The purpose of using an instcounter is to track particular
        instances of a counter object (e.g. a single processor, a single
        running copy of a process).  For instance, to track all python.exe
        instances, you would need merely to ask:
                query.addinstcounter('python','Virtual Bytes')
        You can find the names of the objects and their available counters
        by doing an addcounterbybrowsing() call on a query object (or by
        looking in performance monitor's add dialog.)

        Beyond merely rearranging the call arguments to make more sense,
        if the volatile flag is true, the instcounters also recalculate
        the paths of the available instances on every call to open the
        query.
        """
        if volatile:
            self.volatilecounters.append((object, counter, machine, objtype, format))
        else:
            self.paths[len(self.paths) :] = self.getinstpaths(
                object, counter, machine, objtype, format
            )

    def getinstpaths(
        self,
        object,
        counter,
        machine=None,
        objtype="Process",
        format=win32pdh.PDH_FMT_LONG,
    ):
        """
        ### Not an end-user function
        Calculate the paths for an instance object. Should alter
        to allow processing for lists of object-counter pairs.
        """
        items, instances = win32pdh.EnumObjectItems(None, None, objtype, -1)
        # find out how many instances of this element we have...
        instances.sort()
        try:
            cur = instances.index(object)
        except ValueError:
            return []  # no instances of this object
        temp = [object]
        try:
            while instances[cur + 1] == object:
                temp.append(object)
                cur = cur + 1
        except IndexError:  # if we went over the end
            pass
        paths = []
        for ind in range(len(temp)):
            # can this raise an error?
            paths.append(
                win32pdh.MakeCounterPath(
                    (machine, "Process", object, None, ind, counter)
                )
            )
        return paths  # should also return the number of elements for naming purposes

    def open(self, *args, **namedargs):
        """
        Explicitly open a query:
        When you are needing to make multiple calls to the same query,
        it is most efficient to open the query, run all of the calls,
        then close the query, instead of having the collectdata method
        automatically open and close the query each time it runs.
        There are currently no arguments to open.
        """
        # do all the normal opening stuff, self._base is now the query object
        BaseQuery.open(*(self,) + args, **namedargs)
        # should rewrite getinstpaths to take a single tuple
        paths = []
        for tup in self.volatilecounters:
            paths[len(paths) :] = self.getinstpaths(*tup)
        for path in paths:
            try:
                self.counters.append(win32pdh.AddCounter(self._base, path))
                self.curpaths.append(
                    path
                )  # if we fail on the line above, this path won't be in the table or the counters
            except win32api.error:
                pass  # again, what to do with a malformed path???

    def collectdatafor(self, totalperiod, period=1):
        """
        Non-threaded collection of performance data:
        This method allows you to specify the total period for which you would
        like to run the Query, and the time interval between individual
        runs.  The collected data is stored in query.curresults at the
        _end_ of the run.  The pathnames for the query are stored in
        query.curpaths.
        e.g.:
                query.collectdatafor(30,2)
        Will collect data for 30seconds at 2 second intervals
        """
        tempresults = []
        try:
            self.open()
            for ind in range(totalperiod / period):
                tempresults.append(self.collectdata())
                time.sleep(period)
            self.curresults = tempresults
        finally:
            self.close()

    def collectdatawhile(self, period=1):
        """
        Threaded collection of performance data:
        This method sets up a simple semaphor system for signalling
        when you would like to start and stop a threaded data collection
        method.  The collection runs every period seconds until the
        semaphor attribute is set to a non-true value (which normally
        should be done by calling query.collectdatawhile_stop() .)
        e.g.:
                query.collectdatawhile(2)
                # starts the query running, returns control to the caller immediately
                # is collecting data every two seconds.
                # do whatever you want to do while the thread runs, then call:
                query.collectdatawhile_stop()
                # when you want to deal with the data.  It is generally a good idea
                # to sleep for period seconds yourself, since the query will not copy
                # the required data until the next iteration:
                time.sleep(2)
                # now you can access the data from the attributes of the query
                query.curresults
                query.curpaths
        """
        self.collectdatawhile_active = 1
        _thread.start_new_thread(self.collectdatawhile_slave, (period,))

    def collectdatawhile_stop(self):
        """
        Signals the collectdatawhile slave thread to stop collecting data
        on the next logging iteration.
        """
        self.collectdatawhile_active = 0

    def collectdatawhile_slave(self, period):
        """
        ### Not a public function
        Does the threaded work of collecting the data and storing it
        in an attribute of the class.
        """
        tempresults = []
        try:
            self.open()  # also sets active, so can't be changed.
            while self.collectdatawhile_active:
                tempresults.append(self.collectdata())
                time.sleep(period)
            self.curresults = tempresults
        finally:
            self.close()

    # pickle functions
    def __getinitargs__(self):
        return (self.paths,)

    def __getstate__(self):
        return self.volatilecounters

    def __setstate__(self, volatilecounters):
        self.volatilecounters = volatilecounters


class QueryError:
    def __init__(self, query):
        self.query = query

    def __repr__(self):
        return "<Query Error in %s>" % repr(self.query)

    __str__ = __repr__
