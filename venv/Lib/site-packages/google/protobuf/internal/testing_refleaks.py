# Protocol Buffers - Google's data interchange format
# Copyright 2008 Google Inc.  All rights reserved.
# https://developers.google.com/protocol-buffers/
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the
# distribution.
#     * Neither the name of Google Inc. nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""A subclass of unittest.TestCase which checks for reference leaks.

To use:
- Use testing_refleak.BaseTestCase instead of unittest.TestCase
- Configure and compile Python with --with-pydebug

If sys.gettotalrefcount() is not available (because Python was built without
the Py_DEBUG option), then this module is a no-op and tests will run normally.
"""

import copyreg
import gc
import sys
import unittest


class LocalTestResult(unittest.TestResult):
  """A TestResult which forwards events to a parent object, except for Skips."""

  def __init__(self, parent_result):
    unittest.TestResult.__init__(self)
    self.parent_result = parent_result

  def addError(self, test, error):
    self.parent_result.addError(test, error)

  def addFailure(self, test, error):
    self.parent_result.addFailure(test, error)

  def addSkip(self, test, reason):
    pass


class ReferenceLeakCheckerMixin(object):
  """A mixin class for TestCase, which checks reference counts."""

  NB_RUNS = 3

  def run(self, result=None):
    testMethod = getattr(self, self._testMethodName)
    expecting_failure_method = getattr(testMethod, "__unittest_expecting_failure__", False)
    expecting_failure_class = getattr(self, "__unittest_expecting_failure__", False)
    if expecting_failure_class or expecting_failure_method:
      return

    # python_message.py registers all Message classes to some pickle global
    # registry, which makes the classes immortal.
    # We save a copy of this registry, and reset it before we could references.
    self._saved_pickle_registry = copyreg.dispatch_table.copy()

    # Run the test twice, to warm up the instance attributes.
    super(ReferenceLeakCheckerMixin, self).run(result=result)
    super(ReferenceLeakCheckerMixin, self).run(result=result)

    oldrefcount = 0
    local_result = LocalTestResult(result)
    num_flakes = 0

    refcount_deltas = []
    while len(refcount_deltas) < self.NB_RUNS:
      oldrefcount = self._getRefcounts()
      super(ReferenceLeakCheckerMixin, self).run(result=local_result)
      newrefcount = self._getRefcounts()
      # If the GC was able to collect some objects after the call to run() that
      # it could not collect before the call, then the counts won't match.
      if newrefcount < oldrefcount and num_flakes < 2:
        # This result is (probably) a flake -- garbage collectors aren't very
        # predictable, but a lower ending refcount is the opposite of the
        # failure we are testing for. If the result is repeatable, then we will
        # eventually report it, but not after trying to eliminate it.
        num_flakes += 1
        continue
      num_flakes = 0
      refcount_deltas.append(newrefcount - oldrefcount)
    print(refcount_deltas, self)

    try:
      self.assertEqual(refcount_deltas, [0] * self.NB_RUNS)
    except Exception:  # pylint: disable=broad-except
      result.addError(self, sys.exc_info())

  def _getRefcounts(self):
    copyreg.dispatch_table.clear()
    copyreg.dispatch_table.update(self._saved_pickle_registry)
    # It is sometimes necessary to gc.collect() multiple times, to ensure
    # that all objects can be collected.
    gc.collect()
    gc.collect()
    gc.collect()
    return sys.gettotalrefcount()


if hasattr(sys, 'gettotalrefcount'):

  def TestCase(test_class):
    new_bases = (ReferenceLeakCheckerMixin,) + test_class.__bases__
    new_class = type(test_class)(
        test_class.__name__, new_bases, dict(test_class.__dict__))
    return new_class
  SkipReferenceLeakChecker = unittest.skip

else:
  # When PyDEBUG is not enabled, run the tests normally.

  def TestCase(test_class):
    return test_class

  def SkipReferenceLeakChecker(reason):
    del reason  # Don't skip, so don't need a reason.
    def Same(func):
      return func
    return Same
