"""Safe YAML loading within limits on input size, expanded size and nesting."""

from typing import Any, NoReturn

import yaml

# Deepest nesting of lists and mappings a document may have, counted through aliases.
MAX_YAML_DEPTH = 32

# Most nodes a document may compose, counting each alias as one. Composing
# builds a Python object per node before any size check can run, so this caps
# the memory a document can take there. Ordinary documents come nowhere near
# it: 250,000 nodes is a file of several megabytes of records, while frontmatter
# is held to 16 KiB by its caller.
MAX_YAML_NODES = 250_000


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
    node reached through an alias counted every time it is reached. The
    depth counts levels of lists and mappings, following aliases. Both are
    checked before any Python object is built. ``subject`` names the input
    in error messages.

    Raises:
        YAMLLimitError: over ``max_bytes`` or ``max_expanded_bytes``, more
            than :data:`MAX_YAML_NODES` nodes, nested more than :data:`MAX_YAML_DEPTH` levels deep, refers to itself,
            or uses anchors or aliases when ``allow_anchors`` is False.
        yaml.YAMLError: not valid YAML.
    """
    if len(text.encode("utf-8")) > max_bytes:
        raise YAMLLimitError(f"{subject} is larger than {_format_bytes(max_bytes)}.")
    loader_class = _DepthLimitedLoader if allow_anchors else _NoAnchorLoader
    loader = loader_class(text, subject)
    try:
        node = loader.get_single_node()
        if node is None:
            return None
        _check_expanded_document(node, subject, max_expanded_bytes)
        return loader.construct_document(node)
    except RecursionError:
        raise _too_deep(subject) from None
    finally:
        loader.dispose()


class _DepthLimitedLoader(yaml.SafeLoader):
    """Safe loader that stops composing once lists and mappings nest more
    than :data:`MAX_YAML_DEPTH` levels deep or past :data:`MAX_YAML_NODES`
    nodes, before reading any further.

    Depth reached through aliases is checked after composing, in
    :func:`_check_expanded_document`.
    """

    def __init__(self, text: str, subject: str) -> None:
        super().__init__(text)
        self.subject = subject
        self.collection_depth = 0
        self.node_count = 0

    def compose_node(self, parent: yaml.Node | None, index: Any) -> yaml.Node | None:
        self.node_count += 1
        if self.node_count > MAX_YAML_NODES:
            raise YAMLLimitError(
                f"{self.subject} has more than {MAX_YAML_NODES:,} nodes."
            )
        if not self.check_event(yaml.SequenceStartEvent, yaml.MappingStartEvent):
            return super().compose_node(parent, index)
        if self.collection_depth >= MAX_YAML_DEPTH:
            raise _too_deep(self.subject)
        self.collection_depth += 1
        try:
            return super().compose_node(parent, index)
        finally:
            self.collection_depth -= 1


class _NoAnchorLoader(_DepthLimitedLoader):
    def fetch_alias(self) -> NoReturn:
        raise YAMLLimitError(f"{self.subject} cannot use YAML aliases.")

    def fetch_anchor(self) -> NoReturn:
        raise YAMLLimitError(f"{self.subject} cannot use YAML anchors.")


def _check_expanded_document(root: yaml.Node, subject: str, limit: int) -> None:
    """Refuse a document whose expanded size is over ``limit`` or that nests
    more than :data:`MAX_YAML_DEPTH` levels deep.

    Each distinct node is measured once and its size and depth reused
    wherever it is referenced, so the check takes time in proportion to the
    input, not to what it expands to.
    """
    sizes: dict[int, int] = {}
    depths: dict[int, int] = {}
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
            depth = _depth(node, depths)
            if depth > MAX_YAML_DEPTH:
                raise _too_deep(subject)
            sizes[key] = size
            depths[key] = depth
        elif key in open_nodes:
            raise YAMLLimitError(f"{subject} refers to itself.")
        elif key not in sizes:
            open_nodes.add(key)
            pending.append((node, True))
            pending.extend((child, False) for child in _children(node))


def _depth(node: yaml.Node, depths: dict[int, int]) -> int:
    if isinstance(node, yaml.ScalarNode):
        return 0
    return 1 + max((depths[id(child)] for child in _children(node)), default=0)


def _too_deep(subject: str) -> YAMLLimitError:
    return YAMLLimitError(
        f"{subject} is nested more than {MAX_YAML_DEPTH} levels deep."
    )


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
