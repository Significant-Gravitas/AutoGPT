# These classes implement a doctest runner plugin for nose, a "known failure"
# error class, and a customized TestProgram for NumPy.

# Because this module imports nose directly, it should not
# be used except by nosetester.py to avoid a general NumPy
# dependency on nose.
import os
import sys
import doctest
import inspect

import numpy
import nose
from nose.plugins import doctests as npd
from nose.plugins.errorclass import ErrorClass, ErrorClassPlugin
from nose.plugins.base import Plugin
from nose.util import src
from .nosetester import get_package_name
from .utils import KnownFailureException, KnownFailureTest


# Some of the classes in this module begin with 'Numpy' to clearly distinguish
# them from the plethora of very similar names from nose/unittest/doctest

#-----------------------------------------------------------------------------
# Modified version of the one in the stdlib, that fixes a python bug (doctests
# not found in extension modules, https://bugs.python.org/issue3158)
class NumpyDocTestFinder(doctest.DocTestFinder):

    def _from_module(self, module, object):
        """
        Return true if the given object is defined in the given
        module.
        """
        if module is None:
            return True
        elif inspect.isfunction(object):
            return module.__dict__ is object.__globals__
        elif inspect.isbuiltin(object):
            return module.__name__ == object.__module__
        elif inspect.isclass(object):
            return module.__name__ == object.__module__
        elif inspect.ismethod(object):
            # This one may be a bug in cython that fails to correctly set the
            # __module__ attribute of methods, but since the same error is easy
            # to make by extension code writers, having this safety in place
            # isn't such a bad idea
            return module.__name__ == object.__self__.__class__.__module__
        elif inspect.getmodule(object) is not None:
            return module is inspect.getmodule(object)
        elif hasattr(object, '__module__'):
            return module.__name__ == object.__module__
        elif isinstance(object, property):
            return True  # [XX] no way not be sure.
        else:
            raise ValueError("object must be a class or function")

    def _find(self, tests, obj, name, module, source_lines, globs, seen):
        """
        Find tests for the given object and any contained objects, and
        add them to `tests`.
        """

        doctest.DocTestFinder._find(self, tests, obj, name, module,
                                    source_lines, globs, seen)

        # Below we re-run pieces of the above method with manual modifications,
        # because the original code is buggy and fails to correctly identify
        # doctests in extension modules.

        # Local shorthands
        from inspect import (
            isroutine, isclass, ismodule, isfunction, ismethod
            )

        # Look for tests in a module's contained objects.
        if ismodule(obj) and self._recurse:
            for valname, val in obj.__dict__.items():
                valname1 = f'{name}.{valname}'
                if ( (isroutine(val) or isclass(val))
                     and self._from_module(module, val)):

                    self._find(tests, val, valname1, module, source_lines,
                               globs, seen)

        # Look for tests in a class's contained objects.
        if isclass(obj) and self._recurse:
            for valname, val in obj.__dict__.items():
                # Special handling for staticmethod/classmethod.
                if isinstance(val, staticmethod):
                    val = getattr(obj, valname)
                if isinstance(val, classmethod):
                    val = getattr(obj, valname).__func__

                # Recurse to methods, properties, and nested classes.
                if ((isfunction(val) or isclass(val) or
                     ismethod(val) or isinstance(val, property)) and
                      self._from_module(module, val)):
                    valname = f'{name}.{valname}'
                    self._find(tests, val, valname, module, source_lines,
                               globs, seen)


# second-chance checker; if the default comparison doesn't
# pass, then see if the expected output string contains flags that
# tell us to ignore the output
class NumpyOutputChecker(doctest.OutputChecker):
    def check_output(self, want, got, optionflags):
        ret = doctest.OutputChecker.check_output(self, want, got,
                                                 optionflags)
        if not ret:
            if "#random" in want:
                return True

            # it would be useful to normalize endianness so that
            # bigendian machines don't fail all the tests (and there are
            # actually some bigendian examples in the doctests). Let's try
            # making them all little endian
            got = got.replace("'>", "'<")
            want = want.replace("'>", "'<")

            # try to normalize out 32 and 64 bit default int sizes
            for sz in [4, 8]:
                got = got.replace("'<i%d'" % sz, "int")
                want = want.replace("'<i%d'" % sz, "int")

            ret = doctest.OutputChecker.check_output(self, want,
                    got, optionflags)

        return ret


# Subclass nose.plugins.doctests.DocTestCase to work around a bug in
# its constructor that blocks non-default arguments from being passed
# down into doctest.DocTestCase
class NumpyDocTestCase(npd.DocTestCase):
    def __init__(self, test, optionflags=0, setUp=None, tearDown=None,
                 checker=None, obj=None, result_var='_'):
        self._result_var = result_var
        self._nose_obj = obj
        doctest.DocTestCase.__init__(self, test,
                                     optionflags=optionflags,
                                     setUp=setUp, tearDown=tearDown,
                                     checker=checker)


print_state = numpy.get_printoptions()

class NumpyDoctest(npd.Doctest):
    name = 'numpydoctest'   # call nosetests with --with-numpydoctest
    score = 1000  # load late, after doctest builtin

    # always use whitespace and ellipsis options for doctests
    doctest_optflags = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS

    # files that should be ignored for doctests
    doctest_ignore = ['generate_numpy_api.py',
                      'setup.py']

    # Custom classes; class variables to allow subclassing
    doctest_case_class = NumpyDocTestCase
    out_check_class = NumpyOutputChecker
    test_finder_class = NumpyDocTestFinder

    # Don't use the standard doctest option handler; hard-code the option values
    def options(self, parser, env=os.environ):
        Plugin.options(self, parser, env)
        # Test doctests in 'test' files / directories. Standard plugin default
        # is False
        self.doctest_tests = True
        # Variable name; if defined, doctest results stored in this variable in
        # the top-level namespace.  None is the standard default
        self.doctest_result_var = None

    def configure(self, options, config):
        # parent method sets enabled flag from command line --with-numpydoctest
        Plugin.configure(self, options, config)
        self.finder = self.test_finder_class()
        self.parser = doctest.DocTestParser()
        if self.enabled:
            # Pull standard doctest out of plugin list; there's no reason to run
            # both.  In practice the Unplugger plugin above would cover us when
            # run from a standard numpy.test() call; this is just in case
            # someone wants to run our plugin outside the numpy.test() machinery
            config.plugins.plugins = [p for p in config.plugins.plugins
                                      if p.name != 'doctest']

    def set_test_context(self, test):
        """ Configure `test` object to set test context

        We set the numpy / scipy standard doctest namespace

        Parameters
        ----------
        test : test object
            with ``globs`` dictionary defining namespace

        Returns
        -------
        None

        Notes
        -----
        `test` object modified in place
        """
        # set the namespace for tests
        pkg_name = get_package_name(os.path.dirname(test.filename))

        # Each doctest should execute in an environment equivalent to
        # starting Python and executing "import numpy as np", and,
        # for SciPy packages, an additional import of the local
        # package (so that scipy.linalg.basic.py's doctests have an
        # implicit "from scipy import linalg" as well).
        #
        # Note: __file__ allows the doctest in NoseTester to run
        # without producing an error
        test.globs = {'__builtins__':__builtins__,
                      '__file__':'__main__',
                      '__name__':'__main__',
                      'np':numpy}
        # add appropriate scipy import for SciPy tests
        if 'scipy' in pkg_name:
            p = pkg_name.split('.')
            p2 = p[-1]
            test.globs[p2] = __import__(pkg_name, test.globs, {}, [p2])

    # Override test loading to customize test context (with set_test_context
    # method), set standard docstring options, and install our own test output
    # checker
    def loadTestsFromModule(self, module):
        if not self.matches(module.__name__):
            npd.log.debug("Doctest doesn't want module %s", module)
            return
        try:
            tests = self.finder.find(module)
        except AttributeError:
            # nose allows module.__test__ = False; doctest does not and
            # throws AttributeError
            return
        if not tests:
            return
        tests.sort()
        module_file = src(module.__file__)
        for test in tests:
            if not test.examples:
                continue
            if not test.filename:
                test.filename = module_file
            # Set test namespace; test altered in place
            self.set_test_context(test)
            yield self.doctest_case_class(test,
                                          optionflags=self.doctest_optflags,
                                          checker=self.out_check_class(),
                                          result_var=self.doctest_result_var)

    # Add an afterContext method to nose.plugins.doctests.Doctest in order
    # to restore print options to the original state after each doctest
    def afterContext(self):
        numpy.set_printoptions(**print_state)

    # Ignore NumPy-specific build files that shouldn't be searched for tests
    def wantFile(self, file):
        bn = os.path.basename(file)
        if bn in self.doctest_ignore:
            return False
        return npd.Doctest.wantFile(self, file)


class Unplugger:
    """ Nose plugin to remove named plugin late in loading

    By default it removes the "doctest" plugin.
    """
    name = 'unplugger'
    enabled = True  # always enabled
    score = 4000  # load late in order to be after builtins

    def __init__(self, to_unplug='doctest'):
        self.to_unplug = to_unplug

    def options(self, parser, env):
        pass

    def configure(self, options, config):
        # Pull named plugin out of plugins list
        config.plugins.plugins = [p for p in config.plugins.plugins
                                  if p.name != self.to_unplug]


class KnownFailurePlugin(ErrorClassPlugin):
    '''Plugin that installs a KNOWNFAIL error class for the
    KnownFailureClass exception.  When KnownFailure is raised,
    the exception will be logged in the knownfail attribute of the
    result, 'K' or 'KNOWNFAIL' (verbose) will be output, and the
    exception will not be counted as an error or failure.'''
    enabled = True
    knownfail = ErrorClass(KnownFailureException,
                           label='KNOWNFAIL',
                           isfailure=False)

    def options(self, parser, env=os.environ):
        env_opt = 'NOSE_WITHOUT_KNOWNFAIL'
        parser.add_option('--no-knownfail', action='store_true',
                          dest='noKnownFail', default=env.get(env_opt, False),
                          help='Disable special handling of KnownFailure '
                               'exceptions')

    def configure(self, options, conf):
        if not self.can_configure:
            return
        self.conf = conf
        disable = getattr(options, 'noKnownFail', False)
        if disable:
            self.enabled = False

KnownFailure = KnownFailurePlugin   # backwards compat


class FPUModeCheckPlugin(Plugin):
    """
    Plugin that checks the FPU mode before and after each test,
    raising failures if the test changed the mode.
    """

    def prepareTestCase(self, test):
        from numpy.core._multiarray_tests import get_fpu_mode

        def run(result):
            old_mode = get_fpu_mode()
            test.test(result)
            new_mode = get_fpu_mode()

            if old_mode != new_mode:
                try:
                    raise AssertionError(
                        "FPU mode changed from {0:#x} to {1:#x} during the "
                        "test".format(old_mode, new_mode))
                except AssertionError:
                    result.addFailure(test, sys.exc_info())

        return run


# Class allows us to save the results of the tests in runTests - see runTests
# method docstring for details
class NumpyTestProgram(nose.core.TestProgram):
    def runTests(self):
        """Run Tests. Returns true on success, false on failure, and
        sets self.success to the same value.

        Because nose currently discards the test result object, but we need
        to return it to the user, override TestProgram.runTests to retain
        the result
        """
        if self.testRunner is None:
            self.testRunner = nose.core.TextTestRunner(stream=self.config.stream,
                                                       verbosity=self.config.verbosity,
                                                       config=self.config)
        plug_runner = self.config.plugins.prepareTestRunner(self.testRunner)
        if plug_runner is not None:
            self.testRunner = plug_runner
        self.result = self.testRunner.run(self.test)
        self.success = self.result.wasSuccessful()
        return self.success
