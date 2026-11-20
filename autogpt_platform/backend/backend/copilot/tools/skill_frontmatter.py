"""Reads and writes the YAML frontmatter of a ``SKILL.md`` file."""

from collections.abc import Mapping
from typing import Any

import yaml

from backend.util.bounded_yaml import YAMLLimitError, load_bounded_yaml

MAX_FRONTMATTER_BYTES = 16 * 1024
MAX_FRONTMATTER_EXPANDED_BYTES = 1024 * 1024

SkillFrontmatterError = YAMLLimitError


def load_skill_frontmatter(raw_meta: str) -> Any:
    """Parse ``raw_meta`` as YAML within the frontmatter limits.

    Raises:
        SkillFrontmatterError: over a size limit, nested too deeply, or uses
            anchors or aliases.
        yaml.YAMLError: not valid YAML.
    """
    return load_bounded_yaml(
        raw_meta,
        subject="SKILL.md frontmatter",
        max_bytes=MAX_FRONTMATTER_BYTES,
        max_expanded_bytes=MAX_FRONTMATTER_EXPANDED_BYTES,
        allow_anchors=False,
    )


def dump_skill_frontmatter(meta: Mapping[str, Any]) -> str:
    """Render ``meta`` as frontmatter YAML that :func:`load_skill_frontmatter`
    reads back: a value used twice is written out twice, never as an alias."""
    return yaml.dump(dict(meta), Dumper=_FrontmatterDumper, sort_keys=False).strip()


class _FrontmatterDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True
