#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#
import logging
import numbers
from abc import ABC, abstractmethod
from functools import wraps
from importlib.util import find_spec
from typing import NamedTuple, Optional, Dict, Iterable, Union, List, Tuple, Any
from collections.abc import Mapping

import certifi
import grpc
from google.protobuf import json_format
from grpc._channel import _InactiveRpcError, _MultiThreadedRendezvous
from tqdm.autonotebook import tqdm
import json

from pinecone import FetchResponse, QueryResponse, ScoredVector, SingleQueryResults, DescribeIndexStatsResponse
from pinecone.config import Config
from pinecone.core.client.model.namespace_summary import NamespaceSummary
from pinecone.core.client.model.vector import Vector as _Vector
from pinecone.core.grpc.protos.vector_service_pb2 import Vector as GRPCVector, \
    QueryVector as GRPCQueryVector, UpsertRequest, UpsertResponse, DeleteRequest, QueryRequest, \
    FetchRequest, UpdateRequest, DescribeIndexStatsRequest, DeleteResponse, UpdateResponse, \
    SparseValues as GRPCSparseValues
from pinecone.core.client.model.sparse_values import SparseValues
from pinecone.core.grpc.protos.vector_service_pb2_grpc import VectorServiceStub
from pinecone.core.grpc.retry import RetryOnRpcErrorClientInterceptor, RetryConfig
from pinecone.core.utils import _generate_request_id, dict_to_proto_struct, fix_tuple_length
from pinecone.core.utils.constants import MAX_MSG_SIZE, REQUEST_ID, CLIENT_VERSION, REQUIRED_VECTOR_FIELDS, \
    OPTIONAL_VECTOR_FIELDS
from pinecone.exceptions import PineconeException

__all__ = ["GRPCIndex", "GRPCVector", "GRPCQueryVector", "GRPCSparseValues"]

_logger = logging.getLogger(__name__)


class GRPCClientConfig(NamedTuple):
    """
    GRPC client configuration options.

    :param secure: Whether to use encrypted protocol (SSL). defaults to True.
    :type traceroute: bool, optional
    :param timeout: defaults to 2 seconds. Fail if gateway doesn't receive response within timeout.
    :type timeout: int, optional
    :param conn_timeout: defaults to 1. Timeout to retry connection if gRPC is unavailable. 0 is no retry.
    :type conn_timeout: int, optional
    :param reuse_channel: Whether to reuse the same grpc channel for multiple requests
    :type reuse_channel: bool, optional
    :param retry_config: RetryConfig indicating how requests should be retried
    :type retry_config: RetryConfig, optional
    :param grpc_channel_options: A dict of gRPC channel arguments
    :type grpc_channel_options: Dict[str, str]
    """
    secure: bool = True
    timeout: int = 20
    conn_timeout: int = 1
    reuse_channel: bool = True
    retry_config: Optional[RetryConfig] = None
    grpc_channel_options: Dict[str, str] = None

    @classmethod
    def _from_dict(cls, kwargs: dict):
        cls_kwargs = {kk: vv for kk, vv in kwargs.items() if kk in cls._fields}
        return cls(**cls_kwargs)


class GRPCIndexBase(ABC):
    """
    Base class for grpc-based interaction with Pinecone indexes
    """

    _pool = None

    def __init__(self, index_name: str, channel=None, grpc_config: GRPCClientConfig = None,
                 _endpoint_override: str = None):
        self.name = index_name

        self.grpc_client_config = grpc_config or GRPCClientConfig()
        self.retry_config = self.grpc_client_config.retry_config or RetryConfig()
        self.fixed_metadata = {
            "api-key": Config.API_KEY,
            "service-name": index_name,
            "client-version": CLIENT_VERSION
        }
        self._endpoint_override = _endpoint_override

        self.method_config = json.dumps(
            {
                "methodConfig": [
                    {
                        "name": [{"service": "VectorService.Upsert"}],
                        "retryPolicy": {
                            "maxAttempts": 5,
                            "initialBackoff": "0.1s",
                            "maxBackoff": "1s",
                            "backoffMultiplier": 2,
                            "retryableStatusCodes": ["UNAVAILABLE"],
                        },
                    },
                    {
                        "name": [{"service": "VectorService"}],
                        "retryPolicy": {
                            "maxAttempts": 5,
                            "initialBackoff": "0.1s",
                            "maxBackoff": "1s",
                            "backoffMultiplier": 2,
                            "retryableStatusCodes": ["UNAVAILABLE"],
                        },
                    }
                ]
            }
        )

        self._channel = channel or self._gen_channel()
        self.stub = self.stub_class(self._channel)


    @property
    @abstractmethod
    def stub_class(self):
        pass

    def _endpoint(self):
        return self._endpoint_override if self._endpoint_override \
            else f"{self.name}-{Config.PROJECT_NAME}.svc.{Config.ENVIRONMENT}.pinecone.io:443"

    def _gen_channel(self, options=None):
        target = self._endpoint()
        default_options = {
            "grpc.max_send_message_length": MAX_MSG_SIZE,
            "grpc.max_receive_message_length": MAX_MSG_SIZE,
            "grpc.service_config": self.method_config,
            "grpc.enable_retries": True
        }
        if self.grpc_client_config.secure:
            default_options['grpc.ssl_target_name_override'] = target.split(':')[0]
        user_provided_options = options or {}
        _options = tuple((k, v) for k, v in {**default_options, **user_provided_options}.items())
        _logger.debug('creating new channel with endpoint %s options %s and config %s',
                      target, _options, self.grpc_client_config)
        if not self.grpc_client_config.secure:
            channel = grpc.insecure_channel(target, options=_options)
        else:
            root_cas = open(certifi.where(), "rb").read()
            tls = grpc.ssl_channel_credentials(root_certificates=root_cas)
            channel = grpc.secure_channel(target, tls, options=_options)

        return channel

    @property
    def channel(self):
        """Creates GRPC channel."""
        if self.grpc_client_config.reuse_channel and self._channel and self.grpc_server_on():
            return self._channel
        self._channel = self._gen_channel()
        return self._channel

    def grpc_server_on(self) -> bool:
        try:
            grpc.channel_ready_future(self._channel).result(timeout=self.grpc_client_config.conn_timeout)
            return True
        except grpc.FutureTimeoutError:
            return False

    def close(self):
        """Closes the connection to the index."""
        try:
            self._channel.close()
        except TypeError:
            pass

    def _wrap_grpc_call(self, func, request, timeout=None, metadata=None, credentials=None, wait_for_ready=None,
                        compression=None):
        @wraps(func)
        def wrapped():
            user_provided_metadata = metadata or {}
            _metadata = tuple((k, v) for k, v in {
                **self.fixed_metadata, **self._request_metadata(), **user_provided_metadata
            }.items())
            try:
                return func(request, timeout=timeout, metadata=_metadata, credentials=credentials,
                            wait_for_ready=wait_for_ready, compression=compression)
            except _InactiveRpcError as e:
                raise PineconeException(e._state.debug_error_string) from e

        return wrapped()

    def _request_metadata(self) -> Dict[str, str]:
        return {REQUEST_ID: _generate_request_id()}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def parse_sparse_values(sparse_values: dict):
    return SparseValues(indices=sparse_values['indices'], values=sparse_values['values']) if sparse_values else SparseValues(indices=[], values=[])


def parse_fetch_response(response: dict):
    vd = {}
    vectors = response.get('vectors')
    if not vectors:
        return None
    for id, vec in vectors.items():
        v_obj = _Vector(id=vec['id'], values=vec['values'], sparse_values=parse_sparse_values(vec.get('sparseValues')),
                        metadata=vec.get('metadata', None), _check_type=False)
        vd[id] = v_obj
    namespace = response.get('namespace', '')
    return FetchResponse(vectors=vd, namespace=namespace, _check_type=False)


def parse_query_response(response: dict, unary_query: bool, _check_type: bool = False):
    res = []

    # TODO: consider deleting this deprecated case
    for match in response.get('results', []):
        namespace = match.get('namespace', '')
        m = []
        if 'matches' in match:
            for item in match['matches']:
                sc = ScoredVector(id=item['id'], score=item.get('score', 0.0), values=item.get('values', []),
                                  sparse_values=parse_sparse_values(item.get('sparseValues')), metadata=item.get('metadata', {}))
                m.append(sc)
        res.append(SingleQueryResults(matches=m, namespace=namespace))

    m = []
    for item in response.get('matches', []):
        sc = ScoredVector(id=item['id'], score=item.get('score', 0.0), values=item.get('values', []),
                          sparse_values=parse_sparse_values(item.get('sparseValues')), metadata=item.get('metadata', {}),
                          _check_type=_check_type)
        m.append(sc)

    kwargs = {'_check_type': _check_type}
    if unary_query:
        kwargs['namespace'] = response.get('namespace', '')
        kwargs['matches'] = m
    else:
        kwargs['results'] = res
    return QueryResponse(**kwargs)


def parse_stats_response(response: dict):
    fullness = response.get('indexFullness', 0.0)
    total_vector_count = response.get('totalVectorCount', 0)
    dimension = response.get('dimension', 0)
    summaries = response.get('namespaces', {})
    namespace_summaries = {}
    for key in summaries:
        vc = summaries[key].get('vectorCount', 0)
        namespace_summaries[key] = NamespaceSummary(vector_count=vc)
    return DescribeIndexStatsResponse(namespaces=namespace_summaries, dimension=dimension, index_fullness=fullness,
                                      total_vector_count=total_vector_count, _check_type=False)


class PineconeGrpcFuture:
    def __init__(self, delegate):
        self._delegate = delegate

    def cancel(self):
        return self._delegate.cancel()

    def cancelled(self):
        return self._delegate.cancelled()

    def running(self):
        return self._delegate.running()

    def done(self):
        return self._delegate.done()

    def add_done_callback(self, fun):
        return self._delegate.add_done_callback(fun)

    def result(self, timeout=None):
        try:
            return self._delegate.result(timeout=timeout)
        except _MultiThreadedRendezvous as e:
            raise PineconeException(e._state.debug_error_string) from e

    def exception(self, timeout=None):
        return self._delegate.exception(timeout=timeout)

    def traceback(self, timeout=None):
        return self._delegate.traceback(timeout=timeout)


class GRPCIndex(GRPCIndexBase):

    """A client for interacting with a Pinecone index via GRPC API."""

    @property
    def stub_class(self):
        return VectorServiceStub

    def upsert(self,
               vectors: Union[List[GRPCVector], List[tuple], List[dict]],
               async_req: bool = False,
               namespace: Optional[str] = None,
               batch_size: Optional[int] = None,
               show_progress: bool = True,
               **kwargs) -> Union[UpsertResponse, PineconeGrpcFuture]:
        """
        The upsert operation writes vectors into a namespace.
        If a new value is upserted for an existing vector id, it will overwrite the previous value.

        Examples:
            >>> index.upsert([('id1', [1.0, 2.0, 3.0], {'key': 'value'}),
                              ('id2', [1.0, 2.0, 3.0])
                              ],
                              namespace='ns1', async_req=True)
            >>> index.upsert([{'id': 'id1', 'values': [1.0, 2.0, 3.0], 'metadata': {'key': 'value'}},
                              {'id': 'id2',
                                        'values': [1.0, 2.0, 3.0],
                                        'sprase_values': {'indices': [1, 8], 'values': [0.2, 0.4]},
                              ])
            >>> index.upsert([GRPCVector(id='id1', values=[1.0, 2.0, 3.0], metadata={'key': 'value'}),
                              GRPCVector(id='id2', values=[1.0, 2.0, 3.0]),
                              GRPCVector(id='id3',
                                         values=[1.0, 2.0, 3.0],
                                         sparse_values=GRPCSparseValues(indices=[1, 2], values=[0.2, 0.4]))])

        Args:
            vectors (Union[List[Vector], List[Tuple]]): A list of vectors to upsert.

                     A vector can be represented by a 1) GRPCVector object, a 2) tuple or 3) a dictionary
                     1) if a tuple is used, it must be of the form (id, values, metadata) or (id, values).
                        where id is a string, vector is a list of floats, and metadata is a dict.
                        Examples: ('id1', [1.0, 2.0, 3.0], {'key': 'value'}), ('id2', [1.0, 2.0, 3.0])

                    2) if a GRPCVector object is used, a GRPCVector object must be of the form
                        GRPCVector(id, values, metadata), where metadata is an optional argument of type
                        Dict[str, Union[str, float, int, bool, List[int], List[float], List[str]]]
                       Examples: GRPCVector(id='id1', values=[1.0, 2.0, 3.0], metadata={'key': 'value'}),
                                 GRPCVector(id='id2', values=[1.0, 2.0, 3.0]),
                                 GRPCVector(id='id3',
                                            values=[1.0, 2.0, 3.0],
                                            sparse_values=GRPCSparseValues(indices=[1, 2], values=[0.2, 0.4]))

                    3) if a dictionary is used, it must be in the form
                       {'id': str, 'values': List[float], 'sparse_values': {'indices': List[int], 'values': List[float]},
                        'metadata': dict}

                    Note: the dimension of each vector must match the dimension of the index.
            async_req (bool): If True, the upsert operation will be performed asynchronously.
                              Cannot be used with batch_size.
                              Defaults to False. See: https://docs.pinecone.io/docs/performance-tuning [optional]
            namespace (str): The namespace to write to. If not specified, the default namespace is used. [optional]
            batch_size (int): The number of vectors to upsert in each batch.
                                Cannot be used with async_req=Ture.
                               If not specified, all vectors will be upserted in a single batch. [optional]
            show_progress (bool): Whether to show a progress bar using tqdm.
                                  Applied only if batch_size is provided. Default is True.

        Returns: UpsertResponse, contains the number of vectors upserted
        """
        if async_req and batch_size is not None:
            raise ValueError('async_req is not supported when batch_size is provided.'
                             'To upsert in parallel, please follow: '
                             'https://docs.pinecone.io/docs/performance-tuning')

        def _dict_to_grpc_vector(item):
            item_keys = set(item.keys())
            if not item_keys.issuperset(REQUIRED_VECTOR_FIELDS):
                raise ValueError(
                    f"Vector dictionary is missing required fields: {list(REQUIRED_VECTOR_FIELDS - item_keys)}")

            excessive_keys = item_keys - (REQUIRED_VECTOR_FIELDS | OPTIONAL_VECTOR_FIELDS)
            if len(excessive_keys) > 0:
                raise ValueError(f"Found excess keys in the vector dictionary: {list(excessive_keys)}. "
                                 f"The allowed keys are: {list(REQUIRED_VECTOR_FIELDS | OPTIONAL_VECTOR_FIELDS)}")

            sparse_values = None
            if 'sparse_values' in item:
                if not isinstance(item['sparse_values'], Mapping):
                    raise TypeError(f"Column `sparse_values` is expected to be a dictionary, found {type(item['sparse_values'])}")
                indices = item['sparse_values'].get('indices', None)
                values = item['sparse_values'].get('values', None)
                try:
                    sparse_values = GRPCSparseValues(indices=indices, values=values)
                except TypeError as e:
                    raise TypeError("Found unexpected data in column `sparse_values`. "
                                     "Expected format is `'sparse_values': {'indices': List[int], 'values': List[float]}`."
                                     ) from e

            metadata = item.get('metadata', None)
            if metadata is not None and not isinstance(metadata, Mapping):
                raise TypeError(f"Column `metadata` is expected to be a dictionary, found {type(metadata)}")

            try:
                return GRPCVector(id=item['id'], values=item['values'], sparse_values=sparse_values,
                                  metadata=dict_to_proto_struct(metadata))

            except TypeError as e:
                # No need to raise a dedicated error for `id` - protobuf's error message is clear enough
                if not isinstance(item['values'], Iterable) or not isinstance(item['values'][0], numbers.Real):
                    raise TypeError(f"Column `values` is expected to be a list of floats")
                raise

        def _vector_transform(item):
            if isinstance(item, GRPCVector):
                return item
            elif isinstance(item, tuple):
                if len(item) > 3:
                    raise ValueError(f"Found a tuple of length {len(item)} which is not supported. " 
                                     f"Vectors can be represented as tuples either the form (id, values, metadata) or (id, values). "
                                     f"To pass sparse values please use either dicts or GRPCVector objects as inputs.")
                id, values, metadata = fix_tuple_length(item, 3)
                return GRPCVector(id=id, values=values, metadata=dict_to_proto_struct(metadata) or {})
            elif isinstance(item, Mapping):
                return _dict_to_grpc_vector(item)
            raise ValueError(f"Invalid vector value passed: cannot interpret type {type(item)}")

        timeout = kwargs.pop('timeout', None)

        vectors = list(map(_vector_transform, vectors))
        if async_req:
            args_dict = self._parse_non_empty_args([('namespace', namespace)])
            request = UpsertRequest(vectors=vectors, **args_dict, **kwargs)
            future = self._wrap_grpc_call(self.stub.Upsert.future, request, timeout=timeout)
            return PineconeGrpcFuture(future)

        if batch_size is None:
            return self._upsert_batch(vectors, namespace, timeout=timeout, **kwargs)

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size must be a positive integer')

        pbar = tqdm(total=len(vectors), disable=not show_progress, desc='Upserted vectors')
        total_upserted = 0
        for i in range(0, len(vectors), batch_size):
            batch_result = self._upsert_batch(vectors[i:i + batch_size], namespace, timeout=timeout, **kwargs)
            pbar.update(batch_result.upserted_count)
            # we can't use here pbar.n for the case show_progress=False
            total_upserted += batch_result.upserted_count

        return UpsertResponse(upserted_count=total_upserted)

    def _upsert_batch(self,
                      vectors: List[GRPCVector],
                      namespace: Optional[str],
                      timeout: Optional[float],
                      **kwargs) -> UpsertResponse:
        args_dict = self._parse_non_empty_args([('namespace', namespace)])
        request = UpsertRequest(vectors=vectors, **args_dict)
        return self._wrap_grpc_call(self.stub.Upsert, request, timeout=timeout, **kwargs)

    def upsert_from_dataframe(self,
                         df,
                         namespace: str = None,
                         batch_size: int = 500,
                         use_async_requests: bool = True,
                         show_progress: bool = True) -> None:
        """Upserts a dataframe into the index.

        Args:
            df: A pandas dataframe with the following columns: id, vector, and metadata.
            namespace: The namespace to upsert into.
            batch_size: The number of rows to upsert in a single batch.
            use_async_requests: Whether to upsert multiple requests at the same time using asynchronous request mechanism.
                                Set to `False`
            show_progress: Whether to show a progress bar.
        """
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError("The `pandas` package is not installed. Please install pandas to use `upsert_from_dataframe()`")

        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Only pandas dataframes are supported. Found: {type(df)}")

        pbar = tqdm(total=len(df), disable=not show_progress, desc="sending upsert requests")
        results = []
        for chunk in self._iter_dataframe(df, batch_size=batch_size):
            res = self.upsert(vectors=chunk, namespace=namespace, async_req=use_async_requests)
            pbar.update(len(chunk))
            results.append(res)

        if use_async_requests:
            results = [async_result.result() for async_result in tqdm(results, desc="collecting async responses")]

        upserted_count = 0
        for res in results:
            upserted_count += res.upserted_count

        return UpsertResponse(upserted_count=upserted_count)

    @staticmethod
    def _iter_dataframe(df, batch_size):
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size].to_dict(orient="records")
            yield batch

    def delete(self,
               ids: Optional[List[str]] = None,
               delete_all: Optional[bool] = None,
               namespace: Optional[str] = None,
               filter: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = None,
               async_req: bool = False,
               **kwargs) -> Union[DeleteResponse, PineconeGrpcFuture]:
        """
        The Delete operation deletes vectors from the index, from a single namespace.
        No error raised if the vector id does not exist.
        Note: for any delete call, if namespace is not specified, the default namespace is used.

        Delete can occur in the following mutual exclusive ways:
        1. Delete by ids from a single namespace
        2. Delete all vectors from a single namespace by setting delete_all to True
        3. Delete all vectors from a single namespace by specifying a metadata filter
           (note that for this option delete all must be set to False)

        Examples:
            >>> index.delete(ids=['id1', 'id2'], namespace='my_namespace')
            >>> index.delete(delete_all=True, namespace='my_namespace')
            >>> index.delete(filter={'key': 'value'}, namespace='my_namespace', async_req=True)

        Args:
            ids (List[str]): Vector ids to delete [optional]
            delete_all (bool): This indicates that all vectors in the index namespace should be deleted.. [optional]
                               Default is False.
            namespace (str): The namespace to delete vectors from [optional]
                             If not specified, the default namespace is used.
            filter (Dict[str, Union[str, float, int, bool, List, dict]]):
                    If specified, the metadata filter here will be used to select the vectors to delete.
                    This is mutually exclusive with specifying ids to delete in the ids param or using delete_all=True.
                     See https://www.pinecone.io/docs/metadata-filtering/.. [optional]
            async_req (bool): If True, the delete operation will be performed asynchronously.
                              Defaults to False. [optional]

        Returns: DeleteResponse (contains no data) or a PineconeGrpcFuture object if async_req is True.
        """

        if filter is not None:
            filter = dict_to_proto_struct(filter)

        args_dict = self._parse_non_empty_args([('ids', ids),
                                                ('delete_all', delete_all),
                                                ('namespace', namespace),
                                                ('filter', filter)])
        timeout = kwargs.pop('timeout', None)

        request = DeleteRequest(**args_dict, **kwargs)
        if async_req:
            future = self._wrap_grpc_call(self.stub.Delete.future, request, timeout=timeout)
            return PineconeGrpcFuture(future)
        else:
            return self._wrap_grpc_call(self.stub.Delete, request, timeout=timeout)

    def fetch(self,
              ids: Optional[List[str]],
              namespace: Optional[str] = None,
              **kwargs) -> FetchResponse:
        """
        The fetch operation looks up and returns vectors, by ID, from a single namespace.
        The returned vectors include the vector data and/or metadata.

        Examples:
            >>> index.fetch(ids=['id1', 'id2'], namespace='my_namespace')
            >>> index.fetch(ids=['id1', 'id2'])

        Args:
            ids (List[str]): The vector IDs to fetch.
            namespace (str): The namespace to fetch vectors from.
                             If not specified, the default namespace is used. [optional]

        Returns: FetchResponse object which contains the list of Vector objects, and namespace name.
        """
        timeout = kwargs.pop('timeout', None)

        args_dict = self._parse_non_empty_args([('namespace', namespace)])

        request = FetchRequest(ids=ids, **args_dict, **kwargs)
        response = self._wrap_grpc_call(self.stub.Fetch, request, timeout=timeout)
        json_response = json_format.MessageToDict(response)
        return parse_fetch_response(json_response)

    def query(self,
              vector: Optional[List[float]] = None,
              id: Optional[str] = None,
              queries: Optional[Union[List[GRPCQueryVector], List[Tuple]]] = None,
              namespace: Optional[str] = None,
              top_k: Optional[int] = None,
              filter: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = None,
              include_values: Optional[bool] = None,
              include_metadata: Optional[bool] = None,
              sparse_vector: Optional[Union[GRPCSparseValues, Dict[str, Union[List[float], List[int]]]]] = None,
              **kwargs) -> QueryResponse:
        """
        The Query operation searches a namespace, using a query vector.
        It retrieves the ids of the most similar items in a namespace, along with their similarity scores.

        Examples:
            >>> index.query(vector=[1, 2, 3], top_k=10, namespace='my_namespace')
            >>> index.query(id='id1', top_k=10, namespace='my_namespace')
            >>> index.query(vector=[1, 2, 3], top_k=10, namespace='my_namespace', filter={'key': 'value'})
            >>> index.query(id='id1', top_k=10, namespace='my_namespace', include_metadata=True, include_values=True)
            >>> index.query(vector=[1, 2, 3], sparse_vector={'indices': [1, 2], 'values': [0.2, 0.4]},
            >>>             top_k=10, namespace='my_namespace')
            >>> index.query(vector=[1, 2, 3], sparse_vector=GRPCSparseValues([1, 2], [0.2, 0.4]),
            >>>             top_k=10, namespace='my_namespace')

        Args:
            vector (List[float]): The query vector. This should be the same length as the dimension of the index
                                  being queried. Each `query()` request can contain only one of the parameters
                                  `queries`, `id` or `vector`.. [optional]
            id (str): The unique ID of the vector to be used as a query vector.
                      Each `query()` request can contain only one of the parameters
                      `queries`, `vector`, or  `id`.. [optional]
            queries ([GRPCQueryVector]): DEPRECATED. The query vectors.
                                     Each `query()` request can contain only one of the parameters
                                     `queries`, `vector`, or  `id`.. [optional]
            top_k (int): The number of results to return for each query. Must be an integer greater than 1.
            namespace (str): The namespace to fetch vectors from.
                             If not specified, the default namespace is used. [optional]
            filter (Dict[str, Union[str, float, int, bool, List, dict]]):
                    The filter to apply. You can use vector metadata to limit your search.
                    See https://www.pinecone.io/docs/metadata-filtering/.. [optional]
            include_values (bool): Indicates whether vector values are included in the response.
                                   If omitted the server will use the default value of False [optional]
            include_metadata (bool): Indicates whether metadata is included in the response as well as the ids.
                                     If omitted the server will use the default value of False  [optional]
            sparse_vector: (Union[SparseValues, Dict[str, Union[List[float], List[int]]]]): sparse values of the query vector.
                            Expected to be either a GRPCSparseValues object or a dict of the form:
                             {'indices': List[int], 'values': List[float]}, where the lists each have the same length.

        Returns: QueryResponse object which contains the list of the closest vectors as ScoredVector objects,
                 and namespace name.
        """
        def _query_transform(item):
            if isinstance(item, GRPCQueryVector):
                return item
            if isinstance(item, tuple):
                values, filter = fix_tuple_length(item, 2)
                filter = dict_to_proto_struct(filter)
                return GRPCQueryVector(values=values, filter=filter)
            if isinstance(item, Iterable):
                return GRPCQueryVector(values=item)
            raise ValueError(f"Invalid query vector value passed: cannot interpret type {type(item)}")

        queries = list(map(_query_transform, queries)) if queries is not None else None

        if filter is not None:
            filter = dict_to_proto_struct(filter)

        sparse_vector = self._parse_sparse_values_arg(sparse_vector)
        args_dict = self._parse_non_empty_args([('vector', vector),
                                                ('id', id),
                                                ('queries', queries),
                                                ('namespace', namespace),
                                                ('top_k', top_k),
                                                ('filter', filter),
                                                ('include_values', include_values),
                                                ('include_metadata', include_metadata),
                                                ('sparse_vector', sparse_vector)])

        request = QueryRequest(**args_dict)

        timeout = kwargs.pop('timeout', None)
        response = self._wrap_grpc_call(self.stub.Query, request, timeout=timeout)
        json_response = json_format.MessageToDict(response)
        return parse_query_response(json_response, vector is not None or id, _check_type=False)

    def update(self,
               id: str,
               async_req: bool = False,
               values: Optional[List[float]] = None,
               set_metadata: Optional[Dict[str,
                                           Union[str, float, int, bool, List[int], List[float], List[str]]]] = None,
               namespace: Optional[str] = None,
               sparse_values: Optional[Union[GRPCSparseValues, Dict[str, Union[List[float], List[int]]]]] = None,
               **kwargs) -> Union[UpdateResponse, PineconeGrpcFuture]:
        """
        The Update operation updates vector in a namespace.
        If a value is included, it will overwrite the previous value.
        If a set_metadata is included,
        the values of the fields specified in it will be added or overwrite the previous value.

        Examples:
            >>> index.update(id='id1', values=[1, 2, 3], namespace='my_namespace')
            >>> index.update(id='id1', set_metadata={'key': 'value'}, namespace='my_namespace', async_req=True)
            >>> index.update(id='id1', values=[1, 2, 3], sparse_values={'indices': [1, 2], 'values': [0.2, 0.4]},
            >>>              namespace='my_namespace')
            >>> index.update(id='id1', values=[1, 2, 3], sparse_values=GRPCSparseValues(indices=[1, 2], values=[0.2, 0.4]),
            >>>              namespace='my_namespace')

        Args:
            id (str): Vector's unique id.
            async_req (bool): If True, the update operation will be performed asynchronously.
                              Defaults to False. [optional]
            values (List[float]): vector values to set. [optional]
            set_metadata (Dict[str, Union[str, float, int, bool, List[int], List[float], List[str]]]]):
                metadata to set for vector. [optional]
            namespace (str): Namespace name where to update the vector.. [optional]
            sparse_values: (Dict[str, Union[List[float], List[int]]]): sparse values to update for the vector.
                           Expected to be either a GRPCSparseValues object or a dict of the form:
                           {'indices': List[int], 'values': List[float]} where the lists each have the same length.


        Returns: UpdateResponse (contains no data) or a PineconeGrpcFuture object if async_req is True.
        """
        if set_metadata is not None:
            set_metadata = dict_to_proto_struct(set_metadata)
        timeout = kwargs.pop('timeout', None)

        sparse_values = self._parse_sparse_values_arg(sparse_values)
        args_dict = self._parse_non_empty_args([('values', values),
                                                ('set_metadata', set_metadata),
                                                ('namespace', namespace),
                                                ('sparse_values', sparse_values)])

        request = UpdateRequest(id=id, **args_dict)
        if async_req:
            future = self._wrap_grpc_call(self.stub.Update.future, request, timeout=timeout)
            return PineconeGrpcFuture(future)
        else:
            return self._wrap_grpc_call(self.stub.Update, request, timeout=timeout)

    def describe_index_stats(self,
                             filter: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = None,
                             **kwargs) -> DescribeIndexStatsResponse:
        """
        The DescribeIndexStats operation returns statistics about the index's contents.
        For example: The vector count per namespace and the number of dimensions.

        Examples:
            >>> index.describe_index_stats()
            >>> index.describe_index_stats(filter={'key': 'value'})

        Args:
            filter (Dict[str, Union[str, float, int, bool, List, dict]]):
            If this parameter is present, the operation only returns statistics for vectors that satisfy the filter.
            See https://www.pinecone.io/docs/metadata-filtering/.. [optional]

        Returns: DescribeIndexStatsResponse object which contains stats about the index.
        """
        if filter is not None:
            filter = dict_to_proto_struct(filter)
        args_dict = self._parse_non_empty_args([('filter', filter)])
        timeout = kwargs.pop('timeout', None)

        request = DescribeIndexStatsRequest(**args_dict)
        response = self._wrap_grpc_call(self.stub.DescribeIndexStats, request, timeout=timeout)
        json_response = json_format.MessageToDict(response)
        return parse_stats_response(json_response)

    @staticmethod
    def _parse_non_empty_args(args: List[Tuple[str, Any]]) -> Dict[str, Any]:
        return {arg_name: val for arg_name, val in args if val is not None}

    @staticmethod
    def _parse_sparse_values_arg(
            sparse_values: Optional[Union[GRPCSparseValues,
                                          Dict[str, Union[List[float], List[int]]]]]) -> Optional[GRPCSparseValues]:
        if sparse_values is None:
            return None

        if isinstance(sparse_values, GRPCSparseValues):
            return sparse_values

        if not isinstance(sparse_values, dict) or "indices" not in sparse_values or "values" not in sparse_values:
            raise ValueError(
                "Invalid sparse values argument. Expected a dict of: {'indices': List[int], 'values': List[float]}."
                f"Received: {sparse_values}")

        return GRPCSparseValues(indices=sparse_values["indices"], values=sparse_values["values"])
