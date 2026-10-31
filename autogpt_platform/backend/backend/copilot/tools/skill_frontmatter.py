"""Bounded YAML loading for ``SKILL.md`` frontmatter.

Frontmatter is written by users, so it is parsed with the safe loader plus
three limits: no YAML anchors or aliases, a cap on the raw text, and a cap on
the size of the document it expands to.
"""

from collections.abc import Mapping
from typing import Any, NoReturn

import yaml

MAX_FRONTMATTER_BYTES = 64 * 1024
MAX_FRONTMATTER_EXPANDED_BYTES = 1024 * 1024


class SkillFrontmatterError(ValueError):
    """Frontmatter refused for breaking one of the limits above."""


def load_skill_frontmatter(raw_meta: str) -> Any:
    """Parse ``raw_meta`` as YAML within the frontmatter limits.

    Raises:
        SkillFrontmatterError: over a size limit, nested too deeply, or uses
            anchors or aliases.
        yaml.YAMLError: not valid YAML.
    """
    if len(raw_meta.encode("utf-8")) > MAX_FRONTMATTER_BYTES:
        raise SkillFrontmatterError(
            "SKILL.md frontmatter is larger than "
            f"{MAX_FRONTMATTER_BYTES // 1024} KiB."
        )
    loader = _FrontmatterLoader(raw_meta)
    try:
        node = loader.get_single_node()
        if node is None:
            return None
        _check_expanded_size(node)
        return loader.construct_document(node)
    except RecursionError:
        raise SkillFrontmatterError(
            "SKILL.md frontmatter is nested too deeply."
        ) from None
    finally:
        loader.dispose()


def dump_skill_frontmatter(meta: Mapping[str, Any]) -> str:
    """Render ``meta`` as frontmatter YAML that :func:`load_skill_frontmatter`
    reads back: a value used twice is written out twice, never as an alias."""
    return yaml.dump(dict(meta), Dumper=_FrontmatterDumper, sort_keys=False).strip()


class _FrontmatterLoader(yaml.SafeLoader):
    def fetch_alias(self) -> NoReturn:
        raise SkillFrontmatterError("SKILL.md frontmatter cannot use YAML aliases.")

    def fetch_anchor(self) -> NoReturn:
        raise SkillFrontmatterError("SKILL.md frontmatter cannot use YAML anchors.")


class _FrontmatterDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def _check_expanded_size(root: yaml.Node) -> None:
    """Refuse a document whose scalars and collections add up to more than
    :data:`MAX_FRONTMATTER_EXPANDED_BYTES`, before any of it is constructed."""
    size = 0
    pending: list[yaml.Node] = [root]
    while pending:
        node = pending.pop()
        size += 1
        if isinstance(node, yaml.ScalarNode):
            size += len(node.value.encode("utf-8"))
        elif isinstance(node, yaml.SequenceNode):
            pending.extend(node.value)
        elif isinstance(node, yaml.MappingNode):
            for key, value in node.value:
                pending.extend((key, value))
        if size > MAX_FRONTMATTER_EXPANDED_BYTES:
            raise SkillFrontmatterError(
                "SKILL.md frontmatter expands to more than "
                f"{MAX_FRONTMATTER_EXPANDED_BYTES // (1024 * 1024)} MiB."
            )
