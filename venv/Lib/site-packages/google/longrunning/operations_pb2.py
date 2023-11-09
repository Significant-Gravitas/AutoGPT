# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Safe implementation of long-running operations with and without gRPC.

Multiplexes between versions of long-running operations with and without gRPC.
The former is preferred, but not possible in all environments (such as Google
AppEngine Standard).
"""

try:
    from google.longrunning.operations_grpc_pb2 import *
    from google.longrunning.operations_grpc_pb2 import _OPERATION
    from google.longrunning.operations_grpc_pb2 import _GETOPERATIONREQUEST
    from google.longrunning.operations_grpc_pb2 import _LISTOPERATIONSREQUEST
    from google.longrunning.operations_grpc_pb2 import _LISTOPERATIONSRESPONSE
    from google.longrunning.operations_grpc_pb2 import _CANCELOPERATIONREQUEST
    from google.longrunning.operations_grpc_pb2 import _DELETEOPERATIONREQUEST
    from google.longrunning.operations_grpc_pb2 import _OPERATIONINFO
    from google.longrunning.operations_grpc_pb2 import _OPERATIONS
except ImportError:
    from google.longrunning.operations_proto_pb2 import *
    from google.longrunning.operations_proto_pb2 import _OPERATION
    from google.longrunning.operations_proto_pb2 import _OPERATION
    from google.longrunning.operations_proto_pb2 import _GETOPERATIONREQUEST
    from google.longrunning.operations_proto_pb2 import _LISTOPERATIONSREQUEST
    from google.longrunning.operations_proto_pb2 import _LISTOPERATIONSRESPONSE
    from google.longrunning.operations_proto_pb2 import _CANCELOPERATIONREQUEST
    from google.longrunning.operations_proto_pb2 import _DELETEOPERATIONREQUEST
    from google.longrunning.operations_proto_pb2 import _OPERATIONINFO
    from google.longrunning.operations_proto_pb2 import _OPERATIONS
