#! /usr/bin/env python
#
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

"""Adds support for parameterized tests to Python's unittest TestCase class.

A parameterized test is a method in a test case that is invoked with different
argument tuples.

A simple example:

  class AdditionExample(_parameterized.TestCase):
    @_parameterized.parameters(
       (1, 2, 3),
       (4, 5, 9),
       (1, 1, 3))
    def testAddition(self, op1, op2, result):
      self.assertEqual(result, op1 + op2)


Each invocation is a separate test case and properly isolated just
like a normal test method, with its own setUp/tearDown cycle. In the
example above, there are three separate testcases, one of which will
fail due to an assertion error (1 + 1 != 3).

Parameters for individual test cases can be tuples (with positional parameters)
or dictionaries (with named parameters):

  class AdditionExample(_parameterized.TestCase):
    @_parameterized.parameters(
       {'op1': 1, 'op2': 2, 'result': 3},
       {'op1': 4, 'op2': 5, 'result': 9},
    )
    def testAddition(self, op1, op2, result):
      self.assertEqual(result, op1 + op2)

If a parameterized test fails, the error message will show the
original test name (which is modified internally) and the arguments
for the specific invocation, which are part of the string returned by
the shortDescription() method on test cases.

The id method of the test, used internally by the unittest framework,
is also modified to show the arguments. To make sure that test names
stay the same across several invocations, object representations like

  >>> class Foo(object):
  ...  pass
  >>> repr(Foo())
  '<__main__.Foo object at 0x23d8610>'

are turned into '<__main__.Foo>'. For even more descriptive names,
especially in test logs, you can use the named_parameters decorator. In
this case, only tuples are supported, and the first parameters has to
be a string (or an object that returns an apt name when converted via
str()):

  class NamedExample(_parameterized.TestCase):
    @_parameterized.named_parameters(
       ('Normal', 'aa', 'aaa', True),
       ('EmptyPrefix', '', 'abc', True),
       ('BothEmpty', '', '', True))
    def testStartsWith(self, prefix, string, result):
      self.assertEqual(result, strings.startswith(prefix))

Named tests also have the benefit that they can be run individually
from the command line:

  $ testmodule.py NamedExample.testStartsWithNormal
  .
  --------------------------------------------------------------------
  Ran 1 test in 0.000s

  OK

Parameterized Classes
=====================
If invocation arguments are shared across test methods in a single
TestCase class, instead of decorating all test methods
individually, the class itself can be decorated:

  @_parameterized.parameters(
    (1, 2, 3)
    (4, 5, 9))
  class ArithmeticTest(_parameterized.TestCase):
    def testAdd(self, arg1, arg2, result):
      self.assertEqual(arg1 + arg2, result)

    def testSubtract(self, arg2, arg2, result):
      self.assertEqual(result - arg1, arg2)

Inputs from Iterables
=====================
If parameters should be shared across several test cases, or are dynamically
created from other sources, a single non-tuple iterable can be passed into
the decorator. This iterable will be used to obtain the test cases:

  class AdditionExample(_parameterized.TestCase):
    @_parameterized.parameters(
      c.op1, c.op2, c.result for c in testcases
    )
    def testAddition(self, op1, op2, result):
      self.assertEqual(result, op1 + op2)


Single-Argument Test Methods
============================
If a test method takes only one argument, the single argument does not need to
be wrapped into a tuple:

  class NegativeNumberExample(_parameterized.TestCase):
    @_parameterized.parameters(
       -1, -3, -4, -5
    )
    def testIsNegative(self, arg):
      self.assertTrue(IsNegative(arg))
"""

__author__ = 'tmarek@google.com (Torsten Marek)'

import functools
import re
import types
import unittest
import uuid

try:
  # Since python 3
  import collections.abc as collections_abc
except ImportError:
  # Won't work after python 3.8
  import collections as collections_abc

ADDR_RE = re.compile(r'\<([a-zA-Z0-9_\-\.]+) object at 0x[a-fA-F0-9]+\>')
_SEPARATOR = uuid.uuid1().hex
_FIRST_ARG = object()
_ARGUMENT_REPR = object()


def _CleanRepr(obj):
  return ADDR_RE.sub(r'<\1>', repr(obj))


# Helper function formerly from the unittest module, removed from it in
# Python 2.7.
def _StrClass(cls):
  return '%s.%s' % (cls.__module__, cls.__name__)


def _NonStringIterable(obj):
  return (isinstance(obj, collections_abc.Iterable) and
          not isinstance(obj, str))


def _FormatParameterList(testcase_params):
  if isinstance(testcase_params, collections_abc.Mapping):
    return ', '.join('%s=%s' % (argname, _CleanRepr(value))
                     for argname, value in testcase_params.items())
  elif _NonStringIterable(testcase_params):
    return ', '.join(map(_CleanRepr, testcase_params))
  else:
    return _FormatParameterList((testcase_params,))


class _ParameterizedTestIter(object):
  """Callable and iterable class for producing new test cases."""

  def __init__(self, test_method, testcases, naming_type):
    """Returns concrete test functions for a test and a list of parameters.

    The naming_type is used to determine the name of the concrete
    functions as reported by the unittest framework. If naming_type is
    _FIRST_ARG, the testcases must be tuples, and the first element must
    have a string representation that is a valid Python identifier.

    Args:
      test_method: The decorated test method.
      testcases: (list of tuple/dict) A list of parameter
                 tuples/dicts for individual test invocations.
      naming_type: The test naming type, either _NAMED or _ARGUMENT_REPR.
    """
    self._test_method = test_method
    self.testcases = testcases
    self._naming_type = naming_type

  def __call__(self, *args, **kwargs):
    raise RuntimeError('You appear to be running a parameterized test case '
                       'without having inherited from parameterized.'
                       'TestCase. This is bad because none of '
                       'your test cases are actually being run.')

  def __iter__(self):
    test_method = self._test_method
    naming_type = self._naming_type

    def MakeBoundParamTest(testcase_params):
      @functools.wraps(test_method)
      def BoundParamTest(self):
        if isinstance(testcase_params, collections_abc.Mapping):
          test_method(self, **testcase_params)
        elif _NonStringIterable(testcase_params):
          test_method(self, *testcase_params)
        else:
          test_method(self, testcase_params)

      if naming_type is _FIRST_ARG:
        # Signal the metaclass that the name of the test function is unique
        # and descriptive.
        BoundParamTest.__x_use_name__ = True
        BoundParamTest.__name__ += str(testcase_params[0])
        testcase_params = testcase_params[1:]
      elif naming_type is _ARGUMENT_REPR:
        # __x_extra_id__ is used to pass naming information to the __new__
        # method of TestGeneratorMetaclass.
        # The metaclass will make sure to create a unique, but nondescriptive
        # name for this test.
        BoundParamTest.__x_extra_id__ = '(%s)' % (
            _FormatParameterList(testcase_params),)
      else:
        raise RuntimeError('%s is not a valid naming type.' % (naming_type,))

      BoundParamTest.__doc__ = '%s(%s)' % (
          BoundParamTest.__name__, _FormatParameterList(testcase_params))
      if test_method.__doc__:
        BoundParamTest.__doc__ += '\n%s' % (test_method.__doc__,)
      return BoundParamTest
    return (MakeBoundParamTest(c) for c in self.testcases)


def _IsSingletonList(testcases):
  """True iff testcases contains only a single non-tuple element."""
  return len(testcases) == 1 and not isinstance(testcases[0], tuple)


def _ModifyClass(class_object, testcases, naming_type):
  assert not getattr(class_object, '_id_suffix', None), (
      'Cannot add parameters to %s,'
      ' which already has parameterized methods.' % (class_object,))
  class_object._id_suffix = id_suffix = {}
  # We change the size of __dict__ while we iterate over it,
  # which Python 3.x will complain about, so use copy().
  for name, obj in class_object.__dict__.copy().items():
    if (name.startswith(unittest.TestLoader.testMethodPrefix)
        and isinstance(obj, types.FunctionType)):
      delattr(class_object, name)
      methods = {}
      _UpdateClassDictForParamTestCase(
          methods, id_suffix, name,
          _ParameterizedTestIter(obj, testcases, naming_type))
      for name, meth in methods.items():
        setattr(class_object, name, meth)


def _ParameterDecorator(naming_type, testcases):
  """Implementation of the parameterization decorators.

  Args:
    naming_type: The naming type.
    testcases: Testcase parameters.

  Returns:
    A function for modifying the decorated object.
  """
  def _Apply(obj):
    if isinstance(obj, type):
      _ModifyClass(
          obj,
          list(testcases) if not isinstance(testcases, collections_abc.Sequence)
          else testcases,
          naming_type)
      return obj
    else:
      return _ParameterizedTestIter(obj, testcases, naming_type)

  if _IsSingletonList(testcases):
    assert _NonStringIterable(testcases[0]), (
        'Single parameter argument must be a non-string iterable')
    testcases = testcases[0]

  return _Apply


def parameters(*testcases):  # pylint: disable=invalid-name
  """A decorator for creating parameterized tests.

  See the module docstring for a usage example.
  Args:
    *testcases: Parameters for the decorated method, either a single
                iterable, or a list of tuples/dicts/objects (for tests
                with only one argument).

  Returns:
     A test generator to be handled by TestGeneratorMetaclass.
  """
  return _ParameterDecorator(_ARGUMENT_REPR, testcases)


def named_parameters(*testcases):  # pylint: disable=invalid-name
  """A decorator for creating parameterized tests.

  See the module docstring for a usage example. The first element of
  each parameter tuple should be a string and will be appended to the
  name of the test method.

  Args:
    *testcases: Parameters for the decorated method, either a single
                iterable, or a list of tuples.

  Returns:
     A test generator to be handled by TestGeneratorMetaclass.
  """
  return _ParameterDecorator(_FIRST_ARG, testcases)


class TestGeneratorMetaclass(type):
  """Metaclass for test cases with test generators.

  A test generator is an iterable in a testcase that produces callables. These
  callables must be single-argument methods. These methods are injected into
  the class namespace and the original iterable is removed. If the name of the
  iterable conforms to the test pattern, the injected methods will be picked
  up as tests by the unittest framework.

  In general, it is supposed to be used in conjunction with the
  parameters decorator.
  """

  def __new__(mcs, class_name, bases, dct):
    dct['_id_suffix'] = id_suffix = {}
    for name, obj in dct.copy().items():
      if (name.startswith(unittest.TestLoader.testMethodPrefix) and
          _NonStringIterable(obj)):
        iterator = iter(obj)
        dct.pop(name)
        _UpdateClassDictForParamTestCase(dct, id_suffix, name, iterator)

    return type.__new__(mcs, class_name, bases, dct)


def _UpdateClassDictForParamTestCase(dct, id_suffix, name, iterator):
  """Adds individual test cases to a dictionary.

  Args:
    dct: The target dictionary.
    id_suffix: The dictionary for mapping names to test IDs.
    name: The original name of the test case.
    iterator: The iterator generating the individual test cases.
  """
  for idx, func in enumerate(iterator):
    assert callable(func), 'Test generators must yield callables, got %r' % (
        func,)
    if getattr(func, '__x_use_name__', False):
      new_name = func.__name__
    else:
      new_name = '%s%s%d' % (name, _SEPARATOR, idx)
    assert new_name not in dct, (
        'Name of parameterized test case "%s" not unique' % (new_name,))
    dct[new_name] = func
    id_suffix[new_name] = getattr(func, '__x_extra_id__', '')


class TestCase(unittest.TestCase, metaclass=TestGeneratorMetaclass):
  """Base class for test cases using the parameters decorator."""

  def _OriginalName(self):
    return self._testMethodName.split(_SEPARATOR)[0]

  def __str__(self):
    return '%s (%s)' % (self._OriginalName(), _StrClass(self.__class__))

  def id(self):  # pylint: disable=invalid-name
    """Returns the descriptive ID of the test.

    This is used internally by the unittesting framework to get a name
    for the test to be used in reports.

    Returns:
      The test id.
    """
    return '%s.%s%s' % (_StrClass(self.__class__),
                        self._OriginalName(),
                        self._id_suffix.get(self._testMethodName, ''))


def CoopTestCase(other_base_class):
  """Returns a new base class with a cooperative metaclass base.

  This enables the TestCase to be used in combination
  with other base classes that have custom metaclasses, such as
  mox.MoxTestBase.

  Only works with metaclasses that do not override type.__new__.

  Example:

    import google3
    import mox

    from google.protobuf.internal import _parameterized

    class ExampleTest(parameterized.CoopTestCase(mox.MoxTestBase)):
      ...

  Args:
    other_base_class: (class) A test case base class.

  Returns:
    A new class object.
  """
  metaclass = type(
      'CoopMetaclass',
      (other_base_class.__metaclass__,
       TestGeneratorMetaclass), {})
  return metaclass(
      'CoopTestCase',
      (other_base_class, TestCase), {})
