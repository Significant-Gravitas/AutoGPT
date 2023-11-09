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

"""Contains metaclasses used to create protocol service and service stub
classes from ServiceDescriptor objects at runtime.

The GeneratedServiceType and GeneratedServiceStubType metaclasses are used to
inject all useful functionality into the classes output by the protocol
compiler at compile-time.
"""

__author__ = 'petar@google.com (Petar Petrov)'


class GeneratedServiceType(type):

  """Metaclass for service classes created at runtime from ServiceDescriptors.

  Implementations for all methods described in the Service class are added here
  by this class. We also create properties to allow getting/setting all fields
  in the protocol message.

  The protocol compiler currently uses this metaclass to create protocol service
  classes at runtime. Clients can also manually create their own classes at
  runtime, as in this example::

    mydescriptor = ServiceDescriptor(.....)
    class MyProtoService(service.Service):
      __metaclass__ = GeneratedServiceType
      DESCRIPTOR = mydescriptor
    myservice_instance = MyProtoService()
    # ...
  """

  _DESCRIPTOR_KEY = 'DESCRIPTOR'

  def __init__(cls, name, bases, dictionary):
    """Creates a message service class.

    Args:
      name: Name of the class (ignored, but required by the metaclass
        protocol).
      bases: Base classes of the class being constructed.
      dictionary: The class dictionary of the class being constructed.
        dictionary[_DESCRIPTOR_KEY] must contain a ServiceDescriptor object
        describing this protocol service type.
    """
    # Don't do anything if this class doesn't have a descriptor. This happens
    # when a service class is subclassed.
    if GeneratedServiceType._DESCRIPTOR_KEY not in dictionary:
      return

    descriptor = dictionary[GeneratedServiceType._DESCRIPTOR_KEY]
    service_builder = _ServiceBuilder(descriptor)
    service_builder.BuildService(cls)
    cls.DESCRIPTOR = descriptor


class GeneratedServiceStubType(GeneratedServiceType):

  """Metaclass for service stubs created at runtime from ServiceDescriptors.

  This class has similar responsibilities as GeneratedServiceType, except that
  it creates the service stub classes.
  """

  _DESCRIPTOR_KEY = 'DESCRIPTOR'

  def __init__(cls, name, bases, dictionary):
    """Creates a message service stub class.

    Args:
      name: Name of the class (ignored, here).
      bases: Base classes of the class being constructed.
      dictionary: The class dictionary of the class being constructed.
        dictionary[_DESCRIPTOR_KEY] must contain a ServiceDescriptor object
        describing this protocol service type.
    """
    super(GeneratedServiceStubType, cls).__init__(name, bases, dictionary)
    # Don't do anything if this class doesn't have a descriptor. This happens
    # when a service stub is subclassed.
    if GeneratedServiceStubType._DESCRIPTOR_KEY not in dictionary:
      return

    descriptor = dictionary[GeneratedServiceStubType._DESCRIPTOR_KEY]
    service_stub_builder = _ServiceStubBuilder(descriptor)
    service_stub_builder.BuildServiceStub(cls)


class _ServiceBuilder(object):

  """This class constructs a protocol service class using a service descriptor.

  Given a service descriptor, this class constructs a class that represents
  the specified service descriptor. One service builder instance constructs
  exactly one service class. That means all instances of that class share the
  same builder.
  """

  def __init__(self, service_descriptor):
    """Initializes an instance of the service class builder.

    Args:
      service_descriptor: ServiceDescriptor to use when constructing the
        service class.
    """
    self.descriptor = service_descriptor

  def BuildService(builder, cls):
    """Constructs the service class.

    Args:
      cls: The class that will be constructed.
    """

    # CallMethod needs to operate with an instance of the Service class. This
    # internal wrapper function exists only to be able to pass the service
    # instance to the method that does the real CallMethod work.
    # Making sure to use exact argument names from the abstract interface in
    # service.py to match the type signature
    def _WrapCallMethod(self, method_descriptor, rpc_controller, request, done):
      return builder._CallMethod(self, method_descriptor, rpc_controller,
                                 request, done)

    def _WrapGetRequestClass(self, method_descriptor):
      return builder._GetRequestClass(method_descriptor)

    def _WrapGetResponseClass(self, method_descriptor):
      return builder._GetResponseClass(method_descriptor)

    builder.cls = cls
    cls.CallMethod = _WrapCallMethod
    cls.GetDescriptor = staticmethod(lambda: builder.descriptor)
    cls.GetDescriptor.__doc__ = 'Returns the service descriptor.'
    cls.GetRequestClass = _WrapGetRequestClass
    cls.GetResponseClass = _WrapGetResponseClass
    for method in builder.descriptor.methods:
      setattr(cls, method.name, builder._GenerateNonImplementedMethod(method))

  def _CallMethod(self, srvc, method_descriptor,
                  rpc_controller, request, callback):
    """Calls the method described by a given method descriptor.

    Args:
      srvc: Instance of the service for which this method is called.
      method_descriptor: Descriptor that represent the method to call.
      rpc_controller: RPC controller to use for this method's execution.
      request: Request protocol message.
      callback: A callback to invoke after the method has completed.
    """
    if method_descriptor.containing_service != self.descriptor:
      raise RuntimeError(
          'CallMethod() given method descriptor for wrong service type.')
    method = getattr(srvc, method_descriptor.name)
    return method(rpc_controller, request, callback)

  def _GetRequestClass(self, method_descriptor):
    """Returns the class of the request protocol message.

    Args:
      method_descriptor: Descriptor of the method for which to return the
        request protocol message class.

    Returns:
      A class that represents the input protocol message of the specified
      method.
    """
    if method_descriptor.containing_service != self.descriptor:
      raise RuntimeError(
          'GetRequestClass() given method descriptor for wrong service type.')
    return method_descriptor.input_type._concrete_class

  def _GetResponseClass(self, method_descriptor):
    """Returns the class of the response protocol message.

    Args:
      method_descriptor: Descriptor of the method for which to return the
        response protocol message class.

    Returns:
      A class that represents the output protocol message of the specified
      method.
    """
    if method_descriptor.containing_service != self.descriptor:
      raise RuntimeError(
          'GetResponseClass() given method descriptor for wrong service type.')
    return method_descriptor.output_type._concrete_class

  def _GenerateNonImplementedMethod(self, method):
    """Generates and returns a method that can be set for a service methods.

    Args:
      method: Descriptor of the service method for which a method is to be
        generated.

    Returns:
      A method that can be added to the service class.
    """
    return lambda inst, rpc_controller, request, callback: (
        self._NonImplementedMethod(method.name, rpc_controller, callback))

  def _NonImplementedMethod(self, method_name, rpc_controller, callback):
    """The body of all methods in the generated service class.

    Args:
      method_name: Name of the method being executed.
      rpc_controller: RPC controller used to execute this method.
      callback: A callback which will be invoked when the method finishes.
    """
    rpc_controller.SetFailed('Method %s not implemented.' % method_name)
    callback(None)


class _ServiceStubBuilder(object):

  """Constructs a protocol service stub class using a service descriptor.

  Given a service descriptor, this class constructs a suitable stub class.
  A stub is just a type-safe wrapper around an RpcChannel which emulates a
  local implementation of the service.

  One service stub builder instance constructs exactly one class. It means all
  instances of that class share the same service stub builder.
  """

  def __init__(self, service_descriptor):
    """Initializes an instance of the service stub class builder.

    Args:
      service_descriptor: ServiceDescriptor to use when constructing the
        stub class.
    """
    self.descriptor = service_descriptor

  def BuildServiceStub(self, cls):
    """Constructs the stub class.

    Args:
      cls: The class that will be constructed.
    """

    def _ServiceStubInit(stub, rpc_channel):
      stub.rpc_channel = rpc_channel
    self.cls = cls
    cls.__init__ = _ServiceStubInit
    for method in self.descriptor.methods:
      setattr(cls, method.name, self._GenerateStubMethod(method))

  def _GenerateStubMethod(self, method):
    return (lambda inst, rpc_controller, request, callback=None:
        self._StubMethod(inst, method, rpc_controller, request, callback))

  def _StubMethod(self, stub, method_descriptor,
                  rpc_controller, request, callback):
    """The body of all service methods in the generated stub class.

    Args:
      stub: Stub instance.
      method_descriptor: Descriptor of the invoked method.
      rpc_controller: Rpc controller to execute the method.
      request: Request protocol message.
      callback: A callback to execute when the method finishes.
    Returns:
      Response message (in case of blocking call).
    """
    return stub.rpc_channel.CallMethod(
        method_descriptor, rpc_controller, request,
        method_descriptor.output_type._concrete_class, callback)
