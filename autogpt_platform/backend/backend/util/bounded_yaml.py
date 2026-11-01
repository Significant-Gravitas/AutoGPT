"""Safe YAML loading within limits on input size, expanded size and nesting."""

from typing import Any, NoReturn

import yaml


class YAMLLimitError(ValueError):
    """YAML refused for breaking one of the limits of :func:`load_bounded_yaml`."""


def load_bounded_yaml(
    text: str,
    *,
    subject: str,
    max_bytes: int,
    max_expanded_bytes: int,
    allow_anchors: bool,
) -> Any:
    """Parse ``text`` as one YAML document with the safe loader.

    The expanded size counts each scalar's bytes plus one per node, with a
    node reached through an alias counted every time it is reached. It is
    checked before any Python object is built. ``subject`` names the input
    in error messages.

    Raises:
        YAMLLimitError: over ``max_bytes`` or ``max_expanded_bytes``, nested
            too deeply, refers to itself, or uses anchors or aliases when
            ``allow_anchors`` is False.
        yaml.YAMLError: not valid YAML.
    """
    if len(text.encode("utf-8")) > max_bytes:
        raise YAMLLimitError(f"{subject} is larger than {_format_bytes(max_bytes)}.")
    loader = yaml.SafeLoader(text) if allow_anchors else _NoAnchorLoader(text, subject)
    try:
        node = loader.get_single_node()
        if node is None:
            return None
        _check_expanded_size(node, subject, max_expanded_bytes)
        return loader.construct_document(node)
    except RecursionError:
        raise YAMLLimitError(f"{subject} is nested too deeply.") from None
    finally:
        loader.dispose()


class _NoAnchorLoader(yaml.SafeLoader):
    def __init__(self, text: str, subject: str) -> None:
        super().__init__(text)
        self.subject = subject

    def fetch_alias(self) -> NoReturn:
        raise YAMLLimitError(f"{self.subject} cannot use YAML aliases.")

    def fetch_anchor(self) -> NoReturn:
        raise YAMLLimitError(f"{self.subject} cannot use YAML anchors.")


def _check_expanded_size(root: yaml.Node, subject: str, limit: int) -> None:
    """Refuse a document whose expanded size is over ``limit``.

    Each distinct node is sized once and its size reused wherever it is
    referenced, so the check takes time in proportion to the input, not to
    what it expands to.
    """
    sizes: dict[int, int] = {}
    open_nodes: set[int] = set()
    pending: list[tuple[yaml.Node, bool]] = [(root, False)]
    while pending:
        node, children_sized = pending.pop()
        key = id(node)
        if children_sized:
            open_nodes.discard(key)
            size = 1 + sum(sizes[id(child)] for child in _children(node))
            if isinstance(node, yaml.ScalarNode):
                size += len(node.value.encode("utf-8"))
            if size > limit:
                raise YAMLLimitError(
                    f"{subject} expands to more than {_format_bytes(limit)}."
                )
            sizes[key] = size
        elif key in open_nodes:
            raise YAMLLimitError(f"{subject} refers to itself.")
        elif key not in sizes:
            open_nodes.add(key)
            pending.append((node, True))
            pending.extend((child, False) for child in _children(node))


def _children(node: yaml.Node) -> list[yaml.Node]:
    if isinstance(node, yaml.SequenceNode):
        return list(node.value)
    if isinstance(node, yaml.MappingNode):
        return [part for pair in node.value for part in pair]
    return []


def _format_bytes(size: int) -> str:
    if size >= 1024 * 1024 and size % (1024 * 1024) == 0:
        return f"{size // (1024 * 1024)} MiB"
    if size >= 1024 and size % 1024 == 0:
        return f"{size // 1024} KiB"
    return f"{size:,} bytes"
