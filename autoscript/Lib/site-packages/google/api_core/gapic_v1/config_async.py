# Copyright 2020 Google LLC
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
"""AsyncIO helpers for loading gapic configuration data.

The Google API generator creates supplementary configuration for each RPC
method to tell the client library how to deal with retries and timeouts.
"""

from google.api_core import retry_async
from google.api_core.gapic_v1 import config
from google.api_core.gapic_v1.config import MethodConfig  # noqa: F401


def parse_method_configs(interface_config):
    """Creates default retry and timeout objects for each method in a gapic
    interface config with AsyncIO semantics.

    Args:
        interface_config (Mapping): The interface config section of the full
            gapic library config. For example, If the full configuration has
            an interface named ``google.example.v1.ExampleService`` you would
            pass in just that interface's configuration, for example
            ``gapic_config['interfaces']['google.example.v1.ExampleService']``.

    Returns:
        Mapping[str, MethodConfig]: A mapping of RPC method names to their
            configuration.
    """
    return config.parse_method_configs(
        interface_config, retry_impl=retry_async.AsyncRetry
    )
