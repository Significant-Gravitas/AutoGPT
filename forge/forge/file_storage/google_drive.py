"""
The GoogleDriveFileStorage class provides an interface for interacting with a
file workspace, and stores the files in Google Drive.
"""

from __future__ import annotations

import inspect
import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

from google.oauth2.credentials import Credentials as GoogleCredentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

from forge.models.config import SystemConfiguration, UserConfigurable

from .base import FileStorage, FileStorageConfiguration

if TYPE_CHECKING:
    from googleapiclient._apis.drive.v3 import File as GoogleDriveFile

logger = logging.getLogger(__name__)


class GoogleDriveFileStorageConfiguration(FileStorageConfiguration):
    class Credentials(SystemConfiguration):
        token: str = UserConfigurable(from_env="GOOGLE_DRIVE_TOKEN")
        refresh_token: str = UserConfigurable(from_env="GOOGLE_DRIVE_REFRESH_TOKEN")
        client_id: str = UserConfigurable(from_env="GOOGLE_DRIVE_CLIENT_ID")
        client_secret: str = UserConfigurable(from_env="GOOGLE_DRIVE_CLIENT_SECRET")
        token_uri: str = UserConfigurable(
            from_env="GOOGLE_DRIVE_TOKEN_URI",
            default="https://oauth2.googleapis.com/token",
        )

    credentials: Credentials
    root_folder_id: str = UserConfigurable(from_env="GOOGLE_DRIVE_ROOT_FOLDER_ID")


class GoogleDriveFileStorage(FileStorage):
    """A class that represents Google Drive storage."""

    def __init__(self, config: GoogleDriveFileStorageConfiguration):
        self._root = config.root
        self._credentials = config.credentials
        self._root_folder_id = config.root_folder_id
        self._drive = build(
            "drive",
            "v3",
            credentials=GoogleCredentials(**self._credentials.model_dump()),
        )
        super().__init__()

    @property
    def root(self) -> Path:
        """The root directory of the file storage."""
        return self._root

    @property
    def restrict_to_root(self) -> bool:
        """Whether to restrict generated paths to the root."""
        return True

    @property
    def is_local(self) -> bool:
        """Whether the storage is local (i.e. on the same machine, not cloud-based)."""
        return False

    def initialize(self) -> None:
        logger.debug(f"Initializing {repr(self)}...")
        # Check if root folder exists, create if it doesn't
        if not self._root_folder_id:
            folder_info: GoogleDriveFile = {
                "name": "AutoGPT Root",
                "mimeType": "application/vnd.google-apps.folder",
            }
            folder = self._drive.files().create(body=folder_info, fields="id").execute()
            self._root_folder_id = folder.get("id")

    def get_path(self, relative_path: str | Path) -> Path:
        return super().get_path(relative_path)

    def _get_file_id(self, path: str | Path) -> str:
        path = self.get_path(path)
        query = (
            f"name='{path.name}' "
            f"and '{self._root_folder_id}' in parents "
            f"and trashed=false"
        )
        results = self._drive.files().list(q=query, fields="files(id)").execute()
        files = results.get("files", [])
        if not files:
            raise ValueError(f"No file or folder '{path.name}' in workspace")
        return files[0]["id"]

    @overload
    def open_file(
        self,
        path: str | Path,
        mode: Literal["r", "w"] = "r",
        binary: Literal[False] = False,
    ) -> io.TextIOWrapper:
        ...

    @overload
    def open_file(
        self, path: str | Path, mode: Literal["r", "w"], binary: Literal[True]
    ) -> io.BytesIO:
        ...

    @overload
    def open_file(
        self, path: str | Path, mode: Literal["r", "w"] = "r", binary: bool = False
    ) -> io.TextIOWrapper | io.BytesIO:
        ...

    def open_file(
        self, path: str | Path, mode: Literal["r", "w"] = "r", binary: bool = False
    ) -> io.TextIOWrapper | io.BytesIO:
        """Open a file in the storage."""
        file_id = self._get_file_id(path)
        if mode == "r":
            request = self._drive.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                _, done = downloader.next_chunk()
            fh.seek(0)
            return fh if binary else io.TextIOWrapper(fh)
        elif mode == "w":
            return io.BytesIO() if binary else io.StringIO()

    @overload
    def read_file(self, path: str | Path, binary: Literal[False] = False) -> str:
        """Read a file in the storage as text."""
        ...

    @overload
    def read_file(self, path: str | Path, binary: Literal[True]) -> bytes:
        """Read a file in the storage as binary."""
        ...

    def read_file(self, path: str | Path, binary: bool = False) -> str | bytes:
        """Read a file in the storage."""
        with self.open_file(path, "r", binary) as f:
            return f.read()

    async def write_file(self, path: str | Path, content: str | bytes) -> None:
        """Write to a file in the storage."""
        path = self.get_path(path)
        media = MediaIoBaseUpload(
            io.BytesIO(content.encode() if isinstance(content, str) else content),
            mimetype="application/octet-stream",
        )
        file_metadata: GoogleDriveFile = {
            "name": path.name,
            "parents": [self._root_folder_id],
        }
        self._drive.files().create(
            body=file_metadata,
            media_body=media,
            fields="id",
        ).execute()

        if self.on_write_file:
            path = Path(path)
            if path.is_absolute():
                path = path.relative_to(self.root)
            res = self.on_write_file(path)
            if inspect.isawaitable(res):
                await res

    def list_files(self, path: str | Path = ".") -> list[Path]:
        """List all files (recursively) in a directory in the storage."""
        query = f"'{self._root_folder_id}' in parents and trashed=false"
        results = self._drive.files().list(q=query, fields="files(name)").execute()
        return [Path(item["name"]) for item in results.get("files", [])]

    def list_folders(
        self, path: str | Path = ".", recursive: bool = False
    ) -> list[Path]:
        """List 'directories' directly in a given path or recursively in the storage."""
        query = (
            f"'{self._root_folder_id}' in parents "
            f"and mimeType='application/vnd.google-apps.folder' "
            f"and trashed=false"
        )
        results = self._drive.files().list(q=query, fields="files(name)").execute()
        return [Path(item["name"]) for item in results.get("files", [])]

    def delete_file(self, path: str | Path) -> None:
        """Delete a file in the storage."""
        file_id = self._get_file_id(path)
        if file_id:
            self._drive.files().delete(fileId=file_id).execute()

    def delete_dir(self, path: str | Path) -> None:
        """Delete an empty folder in the storage."""
        folder_id = self._get_file_id(path)
        if folder_id:
            self._drive.files().delete(fileId=folder_id).execute()

    def exists(self, path: str | Path) -> bool:
        """Check if a file or folder exists in Google Drive storage."""
        return bool(self._get_file_id(path))

    def make_dir(self, path: str | Path) -> None:
        """Create a directory in the storage if doesn't exist."""
        path = self.get_path(path)
        folder_metadata: GoogleDriveFile = {
            "name": path.name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [self._root_folder_id],
        }
        self._drive.files().create(body=folder_metadata, fields="id").execute()

    def rename(self, old_path: str | Path, new_path: str | Path) -> None:
        """Rename a file or folder in the storage."""
        file_id = self._get_file_id(old_path)
        new_path = self.get_path(new_path)
        file_metadata: GoogleDriveFile = {"name": new_path.name}
        self._drive.files().update(fileId=file_id, body=file_metadata).execute()

    def copy(self, source: str | Path, destination: str | Path) -> None:
        """Copy a file or folder with all contents in the storage."""
        file_id = self._get_file_id(source)
        destination = self.get_path(destination)
        file_metadata: GoogleDriveFile = {
            "name": destination.name,
            "parents": [self._root_folder_id],
        }
        self._drive.files().copy(fileId=file_id, body=file_metadata).execute()

    def clone_with_subroot(self, subroot: str | Path) -> GoogleDriveFileStorage:
        """Create a new GoogleDriveFileStorage with a subroot of the current storage."""
        subroot_path = self.get_path(subroot)
        subroot_id = self._get_file_id(subroot_path)
        if not subroot_id:
            raise ValueError(f"Subroot {subroot} does not exist")

        config = GoogleDriveFileStorageConfiguration(
            root=subroot_path,
            root_folder_id=subroot_id,
            credentials=self._credentials,
        )
        return GoogleDriveFileStorage(config)

    def __repr__(self) -> str:
        return f"{__class__.__name__}(root={self._root})"
