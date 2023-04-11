#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#

from typing import NamedTuple
from pinecone.core.api_base import BaseAPI

__all__ = ["ActionAPI", "VersionResponse", "WhoAmIResponse"]

from pinecone.core.utils import get_version


class WhoAmIResponse(NamedTuple):
    username: str = 'UNKNOWN'
    user_label: str = 'UNKNOWN'
    projectname: str = 'UNKNOWN'


class VersionResponse(NamedTuple):
    server: str
    client: str


class ActionAPI(BaseAPI):
    """User related API calls."""
    client_version = get_version()

    def whoami(self) -> WhoAmIResponse:
        """Returns user information."""
        response = self.get("/actions/whoami")
        return WhoAmIResponse(
            username=response.get("user_name", "UNDEFINED"),
            projectname=response.get("project_name", "UNDEFINED"),
            user_label=response.get("user_label", "UNDEFINED"),
        )

    def version(self) -> VersionResponse:
        """Returns version information."""
        response = self.get("/actions/version")
        return VersionResponse(server=response.get("version", "UNKNOWN"),
                               client=self.client_version)
