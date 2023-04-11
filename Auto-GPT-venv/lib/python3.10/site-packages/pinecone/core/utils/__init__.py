#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#
import re
import uuid
import warnings
from pathlib import Path
from typing import List

import requests
import urllib3

try:
    from pinecone.core.grpc.protos import vector_column_service_pb2
    from google.protobuf.struct_pb2 import Struct
    from google.protobuf import json_format
    import numpy as np
    import lz4.frame
except Exception:
    pass  # ignore for non-[grpc] installations

DNS_COMPATIBLE_REGEX = re.compile("^[a-z0-9]([a-z0-9]|[-])+[a-z0-9]$")


def dump_numpy_public(np_array: 'np.ndarray', compressed: bool = False) -> 'vector_column_service_pb2.NdArray':
    """
    Dump numpy array to vector_column_service_pb2.NdArray
    """
    protobuf_arr = vector_column_service_pb2.NdArray()
    protobuf_arr.dtype = str(np_array.dtype)
    protobuf_arr.shape.extend(np_array.shape)
    if compressed:
        protobuf_arr.buffer = lz4.frame.compress(np_array.tobytes())
        protobuf_arr.compressed = True
    else:
        protobuf_arr.buffer = np_array.tobytes()
    return protobuf_arr


def dump_strings_public(strs: List[str], compressed: bool = False) -> 'vector_column_service_pb2.NdArray':
    return dump_numpy_public(np.array(strs, dtype='S'), compressed=compressed)


def get_version():
    return Path(__file__).parent.parent.parent.joinpath('__version__').read_text().strip()


def get_environment():
    return Path(__file__).parent.parent.parent.joinpath('__environment__').read_text().strip()


def validate_dns_name(name):
    if not DNS_COMPATIBLE_REGEX.match(name):
        raise ValueError("{} is invalid - service names and node names must consist of lower case "
                         "alphanumeric characters or '-', start with an alphabetic character, and end with an "
                         "alphanumeric character (e.g. 'my-name', or 'abc-123')".format(name))


def _generate_request_id() -> str:
    return str(uuid.uuid4())


def fix_tuple_length(t, n):
    """Extend tuple t to length n by adding None items at the end of the tuple. Return the new tuple."""
    return t + ((None,) * (n - len(t))) if len(t) < n else t


def get_user_agent():
    client_id = f'python-client-{get_version()}'
    user_agent_details = {'requests': requests.__version__, 'urllib3': urllib3.__version__}
    user_agent = '{} ({})'.format(client_id, ', '.join([f'{k}:{v}' for k, v in user_agent_details.items()]))
    return user_agent


def dict_to_proto_struct(d: dict) -> 'Struct':
    if not d:
        d = {}
    s = Struct()
    s.update(d)
    return s


def proto_struct_to_dict(s: 'Struct') -> dict:
    return json_format.MessageToDict(s)


def load_numpy_public(proto_arr: 'vector_column_service_pb2.NdArray') -> 'np.ndarray':
    """
    Load numpy array from protobuf
    :param proto_arr:
    :return:
    """
    if len(proto_arr.shape) == 0:
        return np.array([])
    if proto_arr.compressed:
        numpy_arr = np.frombuffer(lz4.frame.decompress(proto_arr.buffer), dtype=proto_arr.dtype)
    else:
        numpy_arr = np.frombuffer(proto_arr.buffer, dtype=proto_arr.dtype)
    return numpy_arr.reshape(proto_arr.shape)


def load_strings_public(proto_arr: 'vector_column_service_pb2.NdArray') -> List[str]:
    return [str(item, 'utf-8') for item in load_numpy_public(proto_arr)]


def warn_deprecated(description: str = '', deprecated_in: str = None, removal_in: str = None):
    message = f'DEPRECATED since v{deprecated_in} [Will be removed in v{removal_in}]: {description}'
    warnings.warn(message, DeprecationWarning)
