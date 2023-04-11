#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#

class PineconeException(Exception):
    """The base exception class for all Pinecone client exceptions."""


class PineconeProtocolError(PineconeException):
    """Raised when something unexpected happens mid-request/response."""
