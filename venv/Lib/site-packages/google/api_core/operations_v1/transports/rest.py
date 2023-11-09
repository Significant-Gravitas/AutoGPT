# -*- coding: utf-8 -*-
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
#

import re
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

from requests import __version__ as requests_version

from google.api_core import exceptions as core_exceptions  # type: ignore
from google.api_core import gapic_v1  # type: ignore
from google.api_core import path_template  # type: ignore
from google.api_core import rest_helpers  # type: ignore
from google.api_core import retry as retries  # type: ignore
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.transport.requests import AuthorizedSession  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
from google.protobuf import empty_pb2  # type: ignore
from google.protobuf import json_format  # type: ignore
from .base import DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO, OperationsTransport

OptionalRetry = Union[retries.Retry, object]

DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=BASE_DEFAULT_CLIENT_INFO.gapic_version,
    grpc_version=None,
    rest_version=requests_version,
)


class OperationsRestTransport(OperationsTransport):
    """REST backend transport for Operations.

    Manages long-running operations with an API service.

    When an API method normally takes long time to complete, it can be
    designed to return [Operation][google.api_core.operations_v1.Operation] to the
    client, and the client can use this interface to receive the real
    response asynchronously by polling the operation resource, or pass
    the operation resource to another API (such as Google Cloud Pub/Sub
    API) to receive the response. Any API service that returns
    long-running operations should implement the ``Operations``
    interface so developers can have a consistent client experience.

    This class defines the same methods as the primary client, so the
    primary client can load the underlying transport implementation
    and call it.

    It sends JSON representations of protocol buffers over HTTP/1.1
    """

    def __init__(
        self,
        *,
        host: str = "longrunning.googleapis.com",
        credentials: ga_credentials.Credentials = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        client_cert_source_for_mtls: Optional[Callable[[], Tuple[bytes, bytes]]] = None,
        quota_project_id: Optional[str] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
        always_use_jwt_access: Optional[bool] = False,
        url_scheme: str = "https",
        http_options: Optional[Dict] = None,
        path_prefix: str = "v1",
    ) -> None:
        """Instantiate the transport.

        Args:
            host (Optional[str]):
                 The hostname to connect to.
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.

            credentials_file (Optional[str]): A file with credentials that can
                be loaded with :func:`google.auth.load_credentials_from_file`.
                This argument is ignored if ``channel`` is provided.
            scopes (Optional(Sequence[str])): A list of scopes. This argument is
                ignored if ``channel`` is provided.
            client_cert_source_for_mtls (Callable[[], Tuple[bytes, bytes]]): Client
                certificate to configure mutual TLS HTTP channel. It is ignored
                if ``channel`` is provided.
            quota_project_id (Optional[str]): An optional project to use for billing
                and quota.
            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you're developing
                your own client library.
            always_use_jwt_access (Optional[bool]): Whether self signed JWT should
                be used for service account credentials.
            url_scheme: the protocol scheme for the API endpoint.  Normally
                "https", but for testing or local servers,
                "http" can be specified.
            http_options: a dictionary of http_options for transcoding, to override
                the defaults from operatons.proto.  Each method has an entry
                with the corresponding http rules as value.
            path_prefix: path prefix (usually represents API version). Set to
                "v1" by default.

        """
        # Run the base constructor
        # TODO(yon-mg): resolve other ctor params i.e. scopes, quota, etc.
        # TODO: When custom host (api_endpoint) is set, `scopes` must *also* be set on the
        # credentials object
        maybe_url_match = re.match("^(?P<scheme>http(?:s)?://)?(?P<host>.*)$", host)
        if maybe_url_match is None:
            raise ValueError(
                f"Unexpected hostname structure: {host}"
            )  # pragma: NO COVER

        url_match_items = maybe_url_match.groupdict()

        host = f"{url_scheme}://{host}" if not url_match_items["scheme"] else host

        super().__init__(
            host=host,
            credentials=credentials,
            client_info=client_info,
            always_use_jwt_access=always_use_jwt_access,
        )
        self._session = AuthorizedSession(
            self._credentials, default_host=self.DEFAULT_HOST
        )
        if client_cert_source_for_mtls:
            self._session.configure_mtls_channel(client_cert_source_for_mtls)
        self._prep_wrapped_messages(client_info)
        self._http_options = http_options or {}
        self._path_prefix = path_prefix

    def _list_operations(
        self,
        request: operations_pb2.ListOperationsRequest,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> operations_pb2.ListOperationsResponse:
        r"""Call the list operations method over HTTP.

        Args:
            request (~.operations_pb2.ListOperationsRequest):
                The request object. The request message for
                [Operations.ListOperations][google.api_core.operations_v1.Operations.ListOperations].

            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            ~.operations_pb2.ListOperationsResponse:
                The response message for
                [Operations.ListOperations][google.api_core.operations_v1.Operations.ListOperations].

        """

        http_options = [
            {
                "method": "get",
                "uri": "/{}/{{name=**}}/operations".format(self._path_prefix),
            },
        ]
        if "google.longrunning.Operations.ListOperations" in self._http_options:
            http_options = self._http_options[
                "google.longrunning.Operations.ListOperations"
            ]

        request_kwargs = json_format.MessageToDict(
            request,
            preserving_proto_field_name=True,
            including_default_value_fields=True,
        )
        transcoded_request = path_template.transcode(http_options, **request_kwargs)

        uri = transcoded_request["uri"]
        method = transcoded_request["method"]

        # Jsonify the query params
        query_params_request = operations_pb2.ListOperationsRequest()
        json_format.ParseDict(transcoded_request["query_params"], query_params_request)
        query_params = json_format.MessageToDict(
            query_params_request,
            including_default_value_fields=False,
            preserving_proto_field_name=False,
            use_integers_for_enums=False,
        )

        # Send the request
        headers = dict(metadata)
        headers["Content-Type"] = "application/json"
        response = getattr(self._session, method)(
            "{host}{uri}".format(host=self._host, uri=uri),
            timeout=timeout,
            headers=headers,
            params=rest_helpers.flatten_query_params(query_params),
        )

        # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
        # subclass.
        if response.status_code >= 400:
            raise core_exceptions.from_http_response(response)

        # Return the response
        api_response = operations_pb2.ListOperationsResponse()
        json_format.Parse(response.content, api_response, ignore_unknown_fields=False)
        return api_response

    def _get_operation(
        self,
        request: operations_pb2.GetOperationRequest,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> operations_pb2.Operation:
        r"""Call the get operation method over HTTP.

        Args:
            request (~.operations_pb2.GetOperationRequest):
                The request object. The request message for
                [Operations.GetOperation][google.api_core.operations_v1.Operations.GetOperation].

            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            ~.operations_pb2.Operation:
                This resource represents a long-
                unning operation that is the result of a
                network API call.

        """

        http_options = [
            {
                "method": "get",
                "uri": "/{}/{{name=**/operations/*}}".format(self._path_prefix),
            },
        ]
        if "google.longrunning.Operations.GetOperation" in self._http_options:
            http_options = self._http_options[
                "google.longrunning.Operations.GetOperation"
            ]

        request_kwargs = json_format.MessageToDict(
            request,
            preserving_proto_field_name=True,
            including_default_value_fields=True,
        )
        transcoded_request = path_template.transcode(http_options, **request_kwargs)

        uri = transcoded_request["uri"]
        method = transcoded_request["method"]

        # Jsonify the query params
        query_params_request = operations_pb2.GetOperationRequest()
        json_format.ParseDict(transcoded_request["query_params"], query_params_request)
        query_params = json_format.MessageToDict(
            query_params_request,
            including_default_value_fields=False,
            preserving_proto_field_name=False,
            use_integers_for_enums=False,
        )

        # Send the request
        headers = dict(metadata)
        headers["Content-Type"] = "application/json"
        response = getattr(self._session, method)(
            "{host}{uri}".format(host=self._host, uri=uri),
            timeout=timeout,
            headers=headers,
            params=rest_helpers.flatten_query_params(query_params),
        )

        # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
        # subclass.
        if response.status_code >= 400:
            raise core_exceptions.from_http_response(response)

        # Return the response
        api_response = operations_pb2.Operation()
        json_format.Parse(response.content, api_response, ignore_unknown_fields=False)
        return api_response

    def _delete_operation(
        self,
        request: operations_pb2.DeleteOperationRequest,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> empty_pb2.Empty:
        r"""Call the delete operation method over HTTP.

        Args:
            request (~.operations_pb2.DeleteOperationRequest):
                The request object. The request message for
                [Operations.DeleteOperation][google.api_core.operations_v1.Operations.DeleteOperation].

            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """

        http_options = [
            {
                "method": "delete",
                "uri": "/{}/{{name=**/operations/*}}".format(self._path_prefix),
            },
        ]
        if "google.longrunning.Operations.DeleteOperation" in self._http_options:
            http_options = self._http_options[
                "google.longrunning.Operations.DeleteOperation"
            ]

        request_kwargs = json_format.MessageToDict(
            request,
            preserving_proto_field_name=True,
            including_default_value_fields=True,
        )
        transcoded_request = path_template.transcode(http_options, **request_kwargs)

        uri = transcoded_request["uri"]
        method = transcoded_request["method"]

        # Jsonify the query params
        query_params_request = operations_pb2.DeleteOperationRequest()
        json_format.ParseDict(transcoded_request["query_params"], query_params_request)
        query_params = json_format.MessageToDict(
            query_params_request,
            including_default_value_fields=False,
            preserving_proto_field_name=False,
            use_integers_for_enums=False,
        )

        # Send the request
        headers = dict(metadata)
        headers["Content-Type"] = "application/json"
        response = getattr(self._session, method)(
            "{host}{uri}".format(host=self._host, uri=uri),
            timeout=timeout,
            headers=headers,
            params=rest_helpers.flatten_query_params(query_params),
        )

        # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
        # subclass.
        if response.status_code >= 400:
            raise core_exceptions.from_http_response(response)

        return empty_pb2.Empty()

    def _cancel_operation(
        self,
        request: operations_pb2.CancelOperationRequest,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> empty_pb2.Empty:
        r"""Call the cancel operation method over HTTP.

        Args:
            request (~.operations_pb2.CancelOperationRequest):
                The request object. The request message for
                [Operations.CancelOperation][google.api_core.operations_v1.Operations.CancelOperation].

            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """

        http_options = [
            {
                "method": "post",
                "uri": "/{}/{{name=**/operations/*}}:cancel".format(self._path_prefix),
                "body": "*",
            },
        ]
        if "google.longrunning.Operations.CancelOperation" in self._http_options:
            http_options = self._http_options[
                "google.longrunning.Operations.CancelOperation"
            ]

        request_kwargs = json_format.MessageToDict(
            request,
            preserving_proto_field_name=True,
            including_default_value_fields=True,
        )
        transcoded_request = path_template.transcode(http_options, **request_kwargs)

        # Jsonify the request body
        body_request = operations_pb2.CancelOperationRequest()
        json_format.ParseDict(transcoded_request["body"], body_request)
        body = json_format.MessageToDict(
            body_request,
            including_default_value_fields=False,
            preserving_proto_field_name=False,
            use_integers_for_enums=False,
        )
        uri = transcoded_request["uri"]
        method = transcoded_request["method"]

        # Jsonify the query params
        query_params_request = operations_pb2.CancelOperationRequest()
        json_format.ParseDict(transcoded_request["query_params"], query_params_request)
        query_params = json_format.MessageToDict(
            query_params_request,
            including_default_value_fields=False,
            preserving_proto_field_name=False,
            use_integers_for_enums=False,
        )

        # Send the request
        headers = dict(metadata)
        headers["Content-Type"] = "application/json"
        response = getattr(self._session, method)(
            "{host}{uri}".format(host=self._host, uri=uri),
            timeout=timeout,
            headers=headers,
            params=rest_helpers.flatten_query_params(query_params),
            data=body,
        )

        # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
        # subclass.
        if response.status_code >= 400:
            raise core_exceptions.from_http_response(response)

        return empty_pb2.Empty()

    @property
    def list_operations(
        self,
    ) -> Callable[
        [operations_pb2.ListOperationsRequest], operations_pb2.ListOperationsResponse
    ]:
        return self._list_operations

    @property
    def get_operation(
        self,
    ) -> Callable[[operations_pb2.GetOperationRequest], operations_pb2.Operation]:
        return self._get_operation

    @property
    def delete_operation(
        self,
    ) -> Callable[[operations_pb2.DeleteOperationRequest], empty_pb2.Empty]:
        return self._delete_operation

    @property
    def cancel_operation(
        self,
    ) -> Callable[[operations_pb2.CancelOperationRequest], empty_pb2.Empty]:
        return self._cancel_operation


__all__ = ("OperationsRestTransport",)
