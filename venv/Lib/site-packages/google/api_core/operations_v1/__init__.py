# Copyright 2017 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Package for interacting with the google.longrunning.operations meta-API."""

from google.api_core.operations_v1.abstract_operations_client import AbstractOperationsClient
from google.api_core.operations_v1.operations_async_client import OperationsAsyncClient
from google.api_core.operations_v1.operations_client import OperationsClient
from google.api_core.operations_v1.transports.rest import OperationsRestTransport

__all__ = [
    "AbstractOperationsClient",
    "OperationsAsyncClient",
    "OperationsClient",
    "OperationsRestTransport"
]
