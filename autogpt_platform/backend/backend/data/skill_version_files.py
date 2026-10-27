"""Immutable, binary-safe sibling files for a private skill version."""

import base64
from typing import Any

from pydantic import BaseModel, TypeAdapter, field_validator

from backend.data.skill_package import file_sha256, package_tree_sha256
from backend.util.json import SafeJson


class SkillVersionFile(BaseModel):
    relative_path: str
    content_base64: str
    is_executable: bool = False

    @field_validator("content_base64")
    @classmethod
    def valid_base64(cls, value: str) -> str:
        base64.b64decode(value, validate=True)
        return value

    @property
    def content(self) -> bytes:
        return base64.b64decode(self.content_base64, validate=True)

    @classmethod
    def from_content(
        cls, relative_path: str, content: bytes, is_executable: bool = False
    ) -> "SkillVersionFile":
        return cls(
            relative_path=relative_path,
            content_base64=base64.b64encode(content).decode("ascii"),
            is_executable=is_executable,
        )


def parse_version_files(value: Any) -> list[SkillVersionFile] | None:
    if value is None:
        return None
    return TypeAdapter(list[SkillVersionFile]).validate_python(value)


def version_files_json(files: list[SkillVersionFile] | None) -> SafeJson:
    if files is None:
        return SafeJson(None)
    return SafeJson([f.model_dump() for f in files])


def version_package_hash(content: str, files: list[SkillVersionFile]) -> str:
    return package_tree_sha256(
        [("SKILL.md", file_sha256(content.encode("utf-8")), False)]
        + [(f.relative_path, file_sha256(f.content), f.is_executable) for f in files]
    )
