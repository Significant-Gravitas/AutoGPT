"""Package snapshots and bounded file evidence for the nightly reviewer."""

from pydantic import BaseModel, Field

from backend.copilot.tools.skills import (
    MAX_PACKAGE_FILES,
    ParsedSkill,
    SkillFile,
    parse_skill_markdown,
    read_user_skill_package,
)
from backend.data.skill_version_files import SkillVersionFile, version_package_hash

MAX_REVIEW_PACKAGE_CHARS = 48_000
MAX_REVIEW_FILES_CHARS = 96_000


class LearningFile(BaseModel):
    relative_path: str
    content: str
    is_executable: bool = False
    supported_by: list[str] = Field(default_factory=list)

    def as_skill_file(self) -> SkillFile:
        return SkillFile(
            relative_path=self.relative_path,
            content=self.content.encode("utf-8"),
            is_executable=self.is_executable,
        )


class ReviewedPackage(BaseModel):
    package_hash: str
    files: list[LearningFile] = Field(default_factory=list)
    complete: bool


async def load_reviewed_packages(
    user_id: str, expert_id: str | None, skills: list[ParsedSkill]
) -> dict[str, ReviewedPackage]:
    packages: dict[str, ReviewedPackage] = {}
    remaining = MAX_REVIEW_FILES_CHARS
    for index, skill in enumerate(skills):
        if not skill.body:
            continue
        try:
            package = await read_user_skill_package(
                user_id, skill.name, expert_id=expert_id
            )
        except Exception:
            continue
        if package is None:
            continue
        parsed = parse_skill_markdown(package.skill_md)
        if parsed is None or parsed.name != skill.name:
            continue
        skills[index] = parsed
        reviewed = _reviewed_package(
            package.skill_md, package.files, min(remaining, MAX_REVIEW_PACKAGE_CHARS)
        )
        packages[skill.name] = reviewed
        remaining -= sum(len(f.content) for f in reviewed.files)
    return packages


def _reviewed_package(
    content: str, files: list[SkillFile], max_chars: int
) -> ReviewedPackage:
    shown: list[LearningFile] = []
    remaining = max_chars
    for file in files[:MAX_PACKAGE_FILES]:
        try:
            text = file.content.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if len(text) > remaining:
            continue
        remaining -= len(text)
        shown.append(
            LearningFile(
                relative_path=file.relative_path,
                content=text,
                is_executable=file.is_executable,
            )
        )
    return ReviewedPackage(
        package_hash=version_package_hash(content, snapshot_files(files)),
        files=shown,
        complete=len(shown) == len(files),
    )


def snapshot_files(files: list[SkillFile]) -> list[SkillVersionFile]:
    return [
        SkillVersionFile.from_content(f.relative_path, f.content, f.is_executable)
        for f in files
    ]


def restore_files(files: list[SkillVersionFile]) -> list[SkillFile]:
    return [
        SkillFile(
            relative_path=f.relative_path,
            content=f.content,
            is_executable=f.is_executable,
        )
        for f in files
    ]
