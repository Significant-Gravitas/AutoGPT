"""Validate every publication before committing its immutable package."""

from backend.copilot.tools.skills import (
    ParsedSkill,
    SkillPackage,
    canonicalize_skill,
    parse_skill_markdown,
    render_skill_markdown,
    validate_package,
    validate_skill_content,
)
from backend.data.skill_version_files import SkillVersionFile

from .content_checks import check_metadata, check_skill_bundle, safe_diagnostic
from .packages import restore_files
from .publication_models import PublishOutcome, PublishRequest


def _validate_request(request: PublishRequest) -> PublishOutcome | None:
    try:
        parsed = canonicalize_skill(
            ParsedSkill(
                name=request.skill_name,
                description=request.description,
                body=request.body,
                triggers=tuple(request.triggers),
            )
        )
        validate_skill_content(
            parsed.description, parsed.body, parsed.triggers, name=parsed.name
        )
        validate_package(
            SkillPackage(
                skill_md=render_skill_markdown(parsed), files=request.files or []
            )
        )
    except ValueError as exc:
        return PublishOutcome(
            status="invalid_proposal", reason=safe_diagnostic(str(exc))
        )
    return None


def _validate_version_content(
    content: str, files: list[SkillVersionFile] | None
) -> PublishOutcome | None:
    parsed = parse_skill_markdown(content)
    if parsed is None:
        return PublishOutcome(
            status="invalid_proposal", reason="stored content unparseable"
        )
    try:
        validate_skill_content(
            parsed.description, parsed.body, parsed.triggers, name=parsed.name
        )
        validate_package(
            SkillPackage(
                skill_md=content,
                files=restore_files(files) if files is not None else [],
            )
        )
    except ValueError as exc:
        return PublishOutcome(
            status="invalid_proposal", reason=safe_diagnostic(str(exc))
        )
    bundle = {
        f.relative_path: f.content.decode("utf-8", errors="replace")
        for f in files or []
    }
    bundle["SKILL.md"] = content
    failure = check_metadata(
        {str(i): f.relative_path for i, f in enumerate(files or [])}
    ) or check_skill_bundle(bundle)
    if failure is not None:
        return PublishOutcome(
            status="blocked_content",
            reason=failure.describe(),
            pattern_class=failure.pattern_class,
            blocked_step=failure.step,
        )
    return None
