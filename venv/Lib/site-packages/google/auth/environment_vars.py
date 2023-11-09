# Copyright 2016 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Environment variables used by :mod:`google.auth`."""


PROJECT = "GOOGLE_CLOUD_PROJECT"
"""Environment variable defining default project.

This used by :func:`google.auth.default` to explicitly set a project ID. This
environment variable is also used by the Google Cloud Python Library.
"""

LEGACY_PROJECT = "GCLOUD_PROJECT"
"""Previously used environment variable defining the default project.

This environment variable is used instead of the current one in some
situations (such as Google App Engine).
"""

GOOGLE_CLOUD_QUOTA_PROJECT = "GOOGLE_CLOUD_QUOTA_PROJECT"
"""Environment variable defining the project to be used for
quota and billing."""

CREDENTIALS = "GOOGLE_APPLICATION_CREDENTIALS"
"""Environment variable defining the location of Google application default
credentials."""

# The environment variable name which can replace ~/.config if set.
CLOUD_SDK_CONFIG_DIR = "CLOUDSDK_CONFIG"
"""Environment variable defines the location of Google Cloud SDK's config
files."""

# These two variables allow for customization of the addresses used when
# contacting the GCE metadata service.
GCE_METADATA_HOST = "GCE_METADATA_HOST"
"""Environment variable providing an alternate hostname or host:port to be
used for GCE metadata requests.

This environment variable was originally named GCE_METADATA_ROOT. The system will
check this environemnt variable first; should there be no value present,
the system will fall back to the old variable.
"""

GCE_METADATA_ROOT = "GCE_METADATA_ROOT"
"""Old environment variable for GCE_METADATA_HOST."""

GCE_METADATA_IP = "GCE_METADATA_IP"
"""Environment variable providing an alternate ip:port to be used for ip-only
GCE metadata requests."""

GOOGLE_API_USE_CLIENT_CERTIFICATE = "GOOGLE_API_USE_CLIENT_CERTIFICATE"
"""Environment variable controlling whether to use client certificate or not.

The default value is false. Users have to explicitly set this value to true
in order to use client certificate to establish a mutual TLS channel."""

LEGACY_APPENGINE_RUNTIME = "APPENGINE_RUNTIME"
"""Gen1 environment variable defining the App Engine Runtime.

Used to distinguish between GAE gen1 and GAE gen2+.
"""

# AWS environment variables used with AWS workload identity pools to retrieve
# AWS security credentials and the AWS region needed to create a serialized
# signed requests to the AWS STS GetCalledIdentity API that can be exchanged
# for a Google access tokens via the GCP STS endpoint.
# When not available the AWS metadata server is used to retrieve these values.
AWS_ACCESS_KEY_ID = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY"
AWS_SESSION_TOKEN = "AWS_SESSION_TOKEN"
AWS_REGION = "AWS_REGION"
AWS_DEFAULT_REGION = "AWS_DEFAULT_REGION"
