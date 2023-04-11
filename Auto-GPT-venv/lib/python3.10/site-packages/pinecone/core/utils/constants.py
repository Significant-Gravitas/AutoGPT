#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#

import os
import enum

from pinecone.core.utils import get_environment, get_version

PARENT_LOGGER_NAME = 'pinecone'
DEFAULT_PARENT_LOGGER_LEVEL = 'ERROR'

MAX_MSG_SIZE = 128 * 1024 * 1024

MAX_ID_LENGTH = int(os.environ.get("PINECONE_MAX_ID_LENGTH", default="64"))

REQUEST_ID: str = "request_id"
CLIENT_VERSION_HEADER = 'X-Pinecone-Client-Version'


class NodeType(str, enum.Enum):
    STANDARD = 'STANDARD'
    COMPUTE = 'COMPUTE'
    MEMORY = 'MEMORY'
    STANDARD2X = 'STANDARD2X'
    COMPUTE2X = 'COMPUTE2X'
    MEMORY2X = 'MEMORY2X'
    STANDARD4X = 'STANDARD4X'
    COMPUTE4X = 'COMPUTE4X'
    MEMORY4X = 'MEMORY4X'


PACKAGE_ENVIRONMENT = get_environment() or "development"
CLIENT_VERSION = get_version()
CLIENT_ID = f'python-client-{CLIENT_VERSION}'

TCP_KEEPINTVL = 60   # Sec
TCP_KEEPIDLE = 300   # Sec
TCP_KEEPCNT = 4

REQUIRED_VECTOR_FIELDS = {'id', 'values'}
OPTIONAL_VECTOR_FIELDS = {'sparse_values', 'metadata'}
