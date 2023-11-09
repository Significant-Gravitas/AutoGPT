# Copyright 2014 Google Inc. All Rights Reserved.
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

"""Utilities for making samples.

Consolidates a lot of code commonly repeated in sample applications.
"""
from __future__ import absolute_import

__author__ = "jcgregorio@google.com (Joe Gregorio)"
__all__ = ["init"]


import argparse
import os

from googleapiclient import discovery
from googleapiclient.http import build_http


def init(
    argv, name, version, doc, filename, scope=None, parents=[], discovery_filename=None
):
    """A common initialization routine for samples.

    Many of the sample applications do the same initialization, which has now
    been consolidated into this function. This function uses common idioms found
    in almost all the samples, i.e. for an API with name 'apiname', the
    credentials are stored in a file named apiname.dat, and the
    client_secrets.json file is stored in the same directory as the application
    main file.

    Args:
      argv: list of string, the command-line parameters of the application.
      name: string, name of the API.
      version: string, version of the API.
      doc: string, description of the application. Usually set to __doc__.
      file: string, filename of the application. Usually set to __file__.
      parents: list of argparse.ArgumentParser, additional command-line flags.
      scope: string, The OAuth scope used.
      discovery_filename: string, name of local discovery file (JSON). Use when discovery doc not available via URL.

    Returns:
      A tuple of (service, flags), where service is the service object and flags
      is the parsed command-line flags.
    """
    try:
        from oauth2client import client, file, tools
    except ImportError:
        raise ImportError(
            "googleapiclient.sample_tools requires oauth2client. Please install oauth2client and try again."
        )

    if scope is None:
        scope = "https://www.googleapis.com/auth/" + name

    # Parser command-line arguments.
    parent_parsers = [tools.argparser]
    parent_parsers.extend(parents)
    parser = argparse.ArgumentParser(
        description=doc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=parent_parsers,
    )
    flags = parser.parse_args(argv[1:])

    # Name of a file containing the OAuth 2.0 information for this
    # application, including client_id and client_secret, which are found
    # on the API Access tab on the Google APIs
    # Console <http://code.google.com/apis/console>.
    client_secrets = os.path.join(os.path.dirname(filename), "client_secrets.json")

    # Set up a Flow object to be used if we need to authenticate.
    flow = client.flow_from_clientsecrets(
        client_secrets, scope=scope, message=tools.message_if_missing(client_secrets)
    )

    # Prepare credentials, and authorize HTTP object with them.
    # If the credentials don't exist or are invalid run through the native client
    # flow. The Storage object will ensure that if successful the good
    # credentials will get written back to a file.
    storage = file.Storage(name + ".dat")
    credentials = storage.get()
    if credentials is None or credentials.invalid:
        credentials = tools.run_flow(flow, storage, flags)
    http = credentials.authorize(http=build_http())

    if discovery_filename is None:
        # Construct a service object via the discovery service.
        service = discovery.build(name, version, http=http)
    else:
        # Construct a service object using a local discovery document file.
        with open(discovery_filename) as discovery_file:
            service = discovery.build_from_document(
                discovery_file.read(), base="https://www.googleapis.com/", http=http
            )
    return (service, flags)
