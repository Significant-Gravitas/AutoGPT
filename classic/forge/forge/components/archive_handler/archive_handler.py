import json
import logging
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Iterator, Optional

from pydantic import BaseModel, Field

from forge.agent.components import ConfigurableComponent
from forge.agent.protocols import CommandProvider, DirectiveProvider
from forge.command import Command, command
from forge.file_storage.base import FileStorage
from forge.models.json_schema import JSONSchema
from forge.utils.exceptions import CommandExecutionError

logger = logging.getLogger(__name__)


class ArchiveHandlerConfiguration(BaseModel):
    max_archive_size: int = Field(
        default=100 * 1024 * 1024,  # 100MB
        description="Maximum archive size in bytes",
    )
    max_extracted_size: int = Field(
        default=500 * 1024 * 1024,  # 500MB
        description="Maximum total size of extracted files",
    )
    max_files: int = Field(
        default=10000,
        description="Maximum number of files in archive",
    )


class ArchiveHandlerComponent(
    DirectiveProvider,
    CommandProvider,
    ConfigurableComponent[ArchiveHandlerConfiguration],
):
    """Provides commands to create, extract, and list archive files."""

    config_class = ArchiveHandlerConfiguration

    def __init__(
        self,
        workspace: FileStorage,
        config: Optional[ArchiveHandlerConfiguration] = None,
    ):
        ConfigurableComponent.__init__(self, config)
        self.workspace = workspace

    def get_resources(self) -> Iterator[str]:
        yield "Ability to create and extract zip/tar archives."

    def get_commands(self) -> Iterator[Command]:
        yield self.create_archive
        yield self.extract_archive
        yield self.list_archive

    def _get_archive_type(self, path: str) -> str:
        """Determine archive type from filename."""
        path_lower = path.lower()
        if path_lower.endswith(".zip"):
            return "zip"
        elif path_lower.endswith((".tar.gz", ".tgz")):
            return "tar.gz"
        elif path_lower.endswith((".tar.bz2", ".tbz2")):
            return "tar.bz2"
        elif path_lower.endswith(".tar"):
            return "tar"
        else:
            return "unknown"

    @command(
        ["create_archive", "zip_files", "compress"],
        "Create a zip or tar archive from files or directories.",
        {
            "output_path": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Output archive path (e.g. 'backup.zip')",
                required=True,
            ),
            "source_paths": JSONSchema(
                type=JSONSchema.Type.ARRAY,
                items=JSONSchema(type=JSONSchema.Type.STRING),
                description="List of files or directories to archive",
                required=True,
            ),
        },
    )
    def create_archive(self, output_path: str, source_paths: list[str]) -> str:
        """Create an archive from specified files/directories.

        Args:
            output_path: Path for the output archive
            source_paths: List of files/directories to include

        Returns:
            str: Success message with archive details
        """
        archive_type = self._get_archive_type(output_path)

        if archive_type == "unknown":
            raise CommandExecutionError(
                "Unsupported archive format. Use .zip, .tar, .tar.gz, or .tar.bz2"
            )

        # Validate source paths exist
        for path in source_paths:
            if not self.workspace.exists(path):
                raise CommandExecutionError(f"Source path '{path}' does not exist")

        full_output = self.workspace.get_path(output_path)

        # Create parent directory if needed
        if directory := os.path.dirname(output_path):
            self.workspace.make_dir(directory)

        file_count = 0
        total_size = 0

        try:
            if archive_type == "zip":
                with zipfile.ZipFile(full_output, "w", zipfile.ZIP_DEFLATED) as zf:
                    for source in source_paths:
                        source_path = self.workspace.get_path(source)
                        if source_path.is_file():
                            zf.write(source_path, source)
                            file_count += 1
                            total_size += source_path.stat().st_size
                        elif source_path.is_dir():
                            for file in source_path.rglob("*"):
                                if file.is_file():
                                    arcname = str(
                                        Path(source) / file.relative_to(source_path)
                                    )
                                    zf.write(file, arcname)
                                    file_count += 1
                                    total_size += file.stat().st_size
            else:
                # Tar formats
                mode = "w"
                if archive_type == "tar.gz":
                    mode = "w:gz"
                elif archive_type == "tar.bz2":
                    mode = "w:bz2"

                with tarfile.open(full_output, mode) as tf:
                    for source in source_paths:
                        source_path = self.workspace.get_path(source)
                        tf.add(source_path, arcname=source)
                        if source_path.is_file():
                            file_count += 1
                            total_size += source_path.stat().st_size
                        else:
                            for file in source_path.rglob("*"):
                                if file.is_file():
                                    file_count += 1
                                    total_size += file.stat().st_size

            archive_size = full_output.stat().st_size
            compression_ratio = (
                round((1 - archive_size / total_size) * 100, 1) if total_size > 0 else 0
            )

            return json.dumps(
                {
                    "archive": output_path,
                    "type": archive_type,
                    "files_added": file_count,
                    "original_size_bytes": total_size,
                    "archive_size_bytes": archive_size,
                    "compression_ratio": f"{compression_ratio}%",
                },
                indent=2,
            )

        except Exception as e:
            raise CommandExecutionError(f"Failed to create archive: {e}")

    @command(
        ["extract_archive", "unzip", "decompress"],
        "Extract files from a zip or tar archive.",
        {
            "archive_path": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Path to the archive file",
                required=True,
            ),
            "destination": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Destination directory (default: current directory)",
                required=False,
            ),
            "members": JSONSchema(
                type=JSONSchema.Type.ARRAY,
                items=JSONSchema(type=JSONSchema.Type.STRING),
                description="Specific files to extract (default: all)",
                required=False,
            ),
        },
    )
    def extract_archive(
        self,
        archive_path: str,
        destination: str = ".",
        members: list[str] | None = None,
    ) -> str:
        """Extract files from an archive.

        Args:
            archive_path: Path to the archive
            destination: Directory to extract to
            members: Specific files to extract

        Returns:
            str: Success message with extraction details
        """
        if not self.workspace.exists(archive_path):
            raise CommandExecutionError(f"Archive '{archive_path}' does not exist")

        archive_type = self._get_archive_type(archive_path)
        full_archive = self.workspace.get_path(archive_path)
        full_dest = self.workspace.get_path(destination)

        # Check archive size
        archive_size = full_archive.stat().st_size
        max_size = self.config.max_archive_size
        if archive_size > max_size:
            raise CommandExecutionError(
                f"Archive too large: {archive_size} bytes (max: {max_size})"
            )

        # Create destination directory
        self.workspace.make_dir(destination)

        extracted_count = 0

        try:
            if archive_type == "zip":
                with zipfile.ZipFile(full_archive, "r") as zf:
                    # Security check for zip slip attack
                    for name in zf.namelist():
                        member_path = (full_dest / name).resolve()
                        if not str(member_path).startswith(str(full_dest.resolve())):
                            raise CommandExecutionError(
                                f"Unsafe archive: '{name}' extracts outside dest"
                            )

                    # Check total uncompressed size
                    total_size = sum(info.file_size for info in zf.infolist())
                    if total_size > self.config.max_extracted_size:
                        raise CommandExecutionError(
                            f"Archive content too large: {total_size} bytes "
                            f"(max: {self.config.max_extracted_size})"
                        )

                    if members:
                        for member in members:
                            zf.extract(member, full_dest)
                            extracted_count += 1
                    else:
                        zf.extractall(full_dest)
                        extracted_count = len(zf.namelist())

            elif archive_type in ("tar", "tar.gz", "tar.bz2"):
                mode = "r"
                if archive_type == "tar.gz":
                    mode = "r:gz"
                elif archive_type == "tar.bz2":
                    mode = "r:bz2"

                with tarfile.open(full_archive, mode) as tf:
                    # Security check for path traversal
                    for member in tf.getmembers():
                        member_path = (full_dest / member.name).resolve()
                        if not str(member_path).startswith(str(full_dest.resolve())):
                            raise CommandExecutionError(
                                f"Unsafe archive: '{member.name}' extracts outside dest"
                            )

                    if members:
                        for member in members:
                            tf.extract(member, full_dest)
                            extracted_count += 1
                    else:
                        tf.extractall(full_dest)
                        extracted_count = len(tf.getmembers())
            else:
                raise CommandExecutionError(
                    f"Unsupported archive format: {archive_type}"
                )

            return json.dumps(
                {
                    "archive": archive_path,
                    "destination": destination,
                    "files_extracted": extracted_count,
                },
                indent=2,
            )

        except (zipfile.BadZipFile, tarfile.TarError) as e:
            raise CommandExecutionError(f"Invalid or corrupted archive: {e}")
        except Exception as e:
            raise CommandExecutionError(f"Extraction failed: {e}")

    @command(
        ["list_archive", "archive_contents"],
        "List the contents of an archive without extracting.",
        {
            "archive_path": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Path to the archive file",
                required=True,
            ),
        },
    )
    def list_archive(self, archive_path: str) -> str:
        """List contents of an archive.

        Args:
            archive_path: Path to the archive

        Returns:
            str: JSON with archive contents
        """
        if not self.workspace.exists(archive_path):
            raise CommandExecutionError(f"Archive '{archive_path}' does not exist")

        archive_type = self._get_archive_type(archive_path)
        full_archive = self.workspace.get_path(archive_path)

        contents = []

        try:
            if archive_type == "zip":
                with zipfile.ZipFile(full_archive, "r") as zf:
                    for info in zf.infolist():
                        contents.append(
                            {
                                "name": info.filename,
                                "size": info.file_size,
                                "compressed_size": info.compress_size,
                                "is_dir": info.is_dir(),
                            }
                        )
            elif archive_type in ("tar", "tar.gz", "tar.bz2"):
                mode = "r"
                if archive_type == "tar.gz":
                    mode = "r:gz"
                elif archive_type == "tar.bz2":
                    mode = "r:bz2"

                with tarfile.open(full_archive, mode) as tf:
                    for member in tf.getmembers():
                        contents.append(
                            {
                                "name": member.name,
                                "size": member.size,
                                "is_dir": member.isdir(),
                            }
                        )
            else:
                raise CommandExecutionError(
                    f"Unsupported archive format: {archive_type}"
                )

            total_size = sum(item.get("size", 0) for item in contents)

            return json.dumps(
                {
                    "archive": archive_path,
                    "type": archive_type,
                    "file_count": len(contents),
                    "total_size_bytes": total_size,
                    "contents": contents,
                },
                indent=2,
            )

        except (zipfile.BadZipFile, tarfile.TarError) as e:
            raise CommandExecutionError(f"Invalid or corrupted archive: {e}")
