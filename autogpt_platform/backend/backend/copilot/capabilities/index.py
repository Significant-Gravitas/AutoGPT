"""Lexical index over capability entries: BM25 plus exact-name lookup.

A search returns ranked entries filtered by context, kind and the turn's
permissions, with primitives weighted below matching services.  When the
query names a service ("linear", "gmail", "mcp.sentry.dev") the main list
is restricted to that service and up to three primitives are returned
separately as ``fallback`` so the model can still build the request by hand.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from .models import SKILL_TOOL, CapabilityContext, CapabilityEntry, CapabilityKindName
from .ranking import ConnectionState, class_weight, resolve_connected, tier
from .text import normalize_name, query_groups, tokenize

if TYPE_CHECKING:
    from backend.copilot.permissions import CopilotPermissions

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
# Exact-name hits sit above every lexical hit; a name match is unambiguous.
_EXACT_BONUS = 100.0
_NAME_WEIGHT = 2  # repeat name tokens so the name outweighs the purpose text

DEFAULT_LIMIT = 8
DEFAULT_FALLBACK_LIMIT = 3


class SearchHit(BaseModel):
    entry: CapabilityEntry
    score: float
    # Query concepts the entry matched: 1 per token matched as written, 0.5
    # when only a synonym matched.  Ranking is coverage first, then
    # connection, then BM25: two entries that both match "create" and
    # "issue" are ordered by whether the user has connected them, not by
    # which description happens to be shorter.
    coverage: float = 0.0
    connected: bool | None = None
    reason: str = "search"  # "exact_id" | "exact_name" | "search"


class SearchResult(BaseModel):
    query: str
    hits: list[SearchHit]
    fallback: list[SearchHit] = []
    service: str | None = None

    @property
    def ids(self) -> list[str]:
        return [hit.entry.id for hit in self.hits]

    @property
    def names(self) -> list[str]:
        return [hit.entry.name for hit in self.hits]


class CapabilityIndex:
    def __init__(
        self,
        entries: Sequence[CapabilityEntry],
        *,
        documents: Sequence[list[str]] | None = None,
    ):
        self.entries: list[CapabilityEntry] = list(entries)
        self._by_id = {entry.id: entry for entry in self.entries}
        # First entry wins a bare ref: the platform's ``web_search`` stays
        # reachable by name when a session's skill is called the same.
        self._by_ref: dict[str, CapabilityEntry] = {}
        for entry in self.entries:
            for impl in entry.implementations:
                self._by_ref.setdefault(impl.ref, entry)
        self._by_name: dict[str, list[int]] = defaultdict(list)
        self._service_tags: dict[str, set[int]] = defaultdict(set)
        for idx, entry in enumerate(self.entries):
            key = normalize_name(entry.name)
            self._by_name[key].append(idx)
            if entry.kind == "block" and key.endswith("block"):
                self._by_name[key[: -len("block")]].append(idx)
            if entry.klass == "service" and entry.kind != "tool":
                for tag in _service_tags(entry):
                    self._service_tags[tag].add(idx)
        # Longest service name in words ("azure_cosmos_db" is three), so
        # `_service_query` knows how far to look ahead when rejoining a
        # multi-word name the query spelled out with spaces.
        self._service_words = max(
            (tag.count("_") + 1 for tag in self._service_tags), default=1
        )
        if documents is None:
            documents = [_document(e) for e in self.entries]
        if len(documents) != len(self.entries):
            raise ValueError("one document per entry")
        self._documents = list(documents)
        self._token_sets = [frozenset(doc) for doc in self._documents]
        self._bm25 = BM25Okapi(self._documents or [[""]])

    def __len__(self) -> int:
        return len(self.entries)

    def with_entries(self, extra: Sequence[CapabilityEntry]) -> CapabilityIndex:
        """This index plus *extra*: the per-session layer (the owner's
        skills) over the platform registry.  The platform documents are
        reused, so only *extra* is tokenised; this index is left as is."""
        if not extra:
            return self
        return CapabilityIndex(
            [*self.entries, *extra],
            documents=[*self._documents, *(_document(e) for e in extra)],
        )

    def get(self, capability_id: str) -> CapabilityEntry | None:
        """Look up by entry id, or by a bare implementation ref (block uuid,
        tool name) so callers holding legacy identifiers still resolve."""
        return self._by_id.get(capability_id) or self._by_ref.get(capability_id)

    def search(
        self,
        query: str,
        *,
        context: CapabilityContext = "direct",
        kind: CapabilityKindName | None = None,
        connections: ConnectionState | None = None,
        permissions: "CopilotPermissions | None" = None,
        limit: int = DEFAULT_LIMIT,
        fallback_limit: int = DEFAULT_FALLBACK_LIMIT,
    ) -> SearchResult:
        query = " ".join((query or "").split())
        if not query:
            return SearchResult(query=query, hits=[])

        allowed = self._allowed_indices(context, kind, permissions)
        scores = self._scores(query, allowed)
        if not scores:
            return SearchResult(query=query, hits=[])

        def to_hit(idx: int, reason: str) -> SearchHit:
            entry = self.entries[idx]
            connected = resolve_connected(entry, connections)
            score, coverage = scores[idx]
            if reason == "search":
                score *= class_weight(entry, connected)
            return SearchHit(
                entry=entry,
                score=score,
                coverage=coverage,
                connected=connected,
                reason=reason,
            )

        exact = self._exact_indices(query, allowed)
        hits = [
            to_hit(idx, "exact_id" if _UUID_RE.match(query) else "exact_name")
            for idx in exact
        ]
        service, service_indices = self._service_query(query)
        rest = [idx for idx in scores if idx not in exact]
        fallback: list[SearchHit] = []
        if service_indices is not None:
            # A skill is not a service, but the one written for the named
            # service is the best answer there is, so skills stay in.
            main = [
                idx
                for idx in rest
                if idx in service_indices or self.entries[idx].kind == "skill"
            ]
            others = [idx for idx in rest if idx not in main]
            fallback = _ranked(
                [
                    to_hit(idx, "search")
                    for idx in others
                    if self.entries[idx].klass == "primitive"
                ]
            )[:fallback_limit]
        else:
            main = rest
        hits += _ranked([to_hit(idx, "search") for idx in main])
        return SearchResult(
            query=query, hits=hits[:limit], fallback=fallback, service=service
        )

    # ------------------------------------------------------------------

    def _allowed_indices(
        self,
        context: CapabilityContext,
        kind: CapabilityKindName | None,
        permissions: "CopilotPermissions | None",
    ) -> set[int]:
        allowed_tools: frozenset[str] | None = None
        if permissions is not None:
            all_tools = frozenset(e.name for e in self.entries if e.kind == "tool")
            allowed_tools = permissions.effective_allowed_tools(
                all_tools | {SKILL_TOOL}
            )
        allowed: set[int] = set()
        for idx, entry in enumerate(self.entries):
            if not entry.available_in(context):
                continue
            if kind is not None and entry.kind != kind:
                continue
            if permissions is not None:
                if entry.kind == "block" and not permissions.is_block_allowed(
                    entry.implementations[0].ref, entry.name
                ):
                    continue
                if (
                    entry.kind == "tool"
                    and allowed_tools is not None
                    and entry.name not in allowed_tools
                ):
                    continue
                # A skill runs through ``read_skill``; a turn that may not
                # call it may not be shown skills either.
                if (
                    entry.kind == "skill"
                    and allowed_tools is not None
                    and SKILL_TOOL not in allowed_tools
                ):
                    continue
            allowed.add(idx)
        return allowed

    def _scores(self, query: str, allowed: set[int]) -> dict[int, tuple[float, float]]:
        """``idx -> (bm25 score, concepts covered)`` for every allowed entry
        with a non-zero score.  A concept is a query token plus its synonyms:
        the token itself counts 1, a synonym-only match 0.5."""
        groups = query_groups(query)
        tokens = [token for group in groups for token in group]
        scores: dict[int, tuple[float, float]] = {}
        if tokens:
            for idx, score in enumerate(self._bm25.get_scores(tokens)):
                if score > 0 and idx in allowed:
                    covered = _coverage(groups, self._token_sets[idx])
                    scores[idx] = (float(score), covered)
        for idx in self._exact_indices(query, allowed):
            score, covered = scores.get(idx, (0.0, float(len(groups))))
            scores[idx] = (score + _EXACT_BONUS, covered)
        return scores

    def _exact_indices(self, query: str, allowed: set[int]) -> list[int]:
        if _UUID_RE.match(query):
            entry = self._by_ref.get(query.lower())
            if entry is None:
                return []
            idx = self.entries.index(entry)
            return [idx] if idx in allowed else []
        key = normalize_name(query)
        if not key:
            return []
        indices = list(self._by_name.get(key, []))
        if key.endswith("block"):
            indices += self._by_name.get(key[: -len("block")], [])
        return [idx for idx in dict.fromkeys(indices) if idx in allowed]

    def _service_query(self, query: str) -> tuple[str | None, set[int] | None]:
        """Services named in the query restrict the main list to them.

        Every named service counts, not just the first: "send a linear issue
        to notion" needs both, and stopping at the first match returned only
        Linear and hid Notion entirely.
        """
        raw = [token for token in re.split(r"[^a-z0-9.]+", query.lower()) if token]
        named: list[str] = []
        indices: set[int] = set()

        def take(tag: str) -> None:
            if tag not in named:
                named.append(tag)
                indices.update(self._service_tags[tag])

        # Service names are stored raw, so a two-word one — "google maps",
        # keyed ``google_maps`` — never matched a single query token. The
        # query matched the bare "google" instead and restricted the search
        # to the Google-provider entries, which is precisely the set that
        # excludes the Maps block: asking for Google Maps returned Docs and
        # Sheets and dropped the one block that was asked for. Rejoin
        # adjacent words and let the longest name win.
        position = 0
        while position < len(raw):
            for size in range(min(self._service_words, len(raw) - position), 0, -1):
                candidate = "_".join(raw[position : position + size])
                if candidate in self._service_tags:
                    take(candidate)
                    position += size
                    break
            else:
                position += 1
        # Stemmed matches still count, for a query that inflects the name.
        for token in tokenize(query):
            if token in self._service_tags:
                take(token)
        if not named:
            return None, None
        return " ".join(named), indices


def _coverage(groups: list[list[str]], doc: frozenset[str]) -> float:
    covered = 0.0
    for token, *synonyms in groups:
        if token in doc:
            covered += 1.0
        elif any(synonym in doc for synonym in synonyms):
            covered += 0.5
    return covered


def _document(entry: CapabilityEntry) -> list[str]:
    tokens = tokenize(entry.name) * _NAME_WEIGHT
    tokens += tokenize(entry.description or entry.purpose)
    for tag in entry.tags:
        tokens += tokenize(tag) or [tag.lower()]
    tokens += tokenize(" ".join(entry.argument_names))
    return tokens


def _service_tags(entry: CapabilityEntry) -> Iterable[str]:
    """Tags that name the service itself: the provider slug for blocks; the
    catalog slug, host and one-word display name for MCP servers.  Raw, not
    tokenised: "ai" or "agent" must never become a service name."""
    if entry.connection.key_type == "provider" and entry.connection.key:
        yield entry.connection.key.lower()
    if entry.kind == "mcp_server":
        # The marker is appended last, so find it from the end: a display
        # name that tokenises to "mcp" would otherwise be mistaken for it
        # and drag the whole sorted prefix in as service names.
        marker = (
            len(entry.tags) - 1 - entry.tags[::-1].index("mcp")
            if "mcp" in entry.tags
            else len(entry.tags)
        )
        yield from (tag.lower() for tag in entry.tags[marker + 1 :])


def _ranked(hits: list[SearchHit]) -> list[SearchHit]:
    """Coverage first; among equals a connected capability, then a platform
    tool (first-party, no credentials, already trusted by the model), then
    the class-weighted BM25 score (``score`` already carries the weight)."""
    return sorted(
        hits,
        key=lambda h: (
            -h.coverage,
            tier(h.entry, h.connected),
            h.entry.kind != "tool",
            -h.score,
            h.entry.name.lower(),
        ),
    )
