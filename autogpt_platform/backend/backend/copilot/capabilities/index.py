"""Lexical index over capability entries: BM25 plus exact-name lookup.

A search returns ranked entries filtered by context, kind and the turn's
permissions, with primitives weighted below matching services.  When the
query names a service ("linear", "gmail", "mcp.sentry.dev") the main list
is restricted to that service plus the platform's own tools, and up to three
primitives are returned separately as ``fallback`` so the model can still
build the request by hand.
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
# Entries of other kinds a ``kind`` filter hid that are listed beside it.
OTHER_KINDS_LIMIT = 3
# How far below the best coverage a connected MCP server the query names
# still counts as covering the query; one whole concept.
CONNECTED_COVERAGE_MARGIN = 1.0
# Platform tools that help connect an account and never do a service's job,
# so a query naming a service does not pull them into its list.
CONNECTION_HELPER_TOOLS = frozenset({"connect_integration"})


class SearchHit(BaseModel):
    entry: CapabilityEntry
    score: float
    # Query concepts the entry matched: 1 per token matched as written, 0.5
    # when only a synonym matched.  Ranking is coverage first, then
    # connection and platform tools (see ``_ranked``), then BM25: two entries
    # that both match "create" and "issue" are ordered by whether the user has
    # connected them, not by which description happens to be shorter.
    coverage: float = 0.0
    connected: bool | None = None
    reason: str = "search"  # "exact_id" | "exact_name" | "search"


class SearchResult(BaseModel):
    query: str
    hits: list[SearchHit]
    fallback: list[SearchHit] = []
    service: str | None = None
    # Set only when ``kind`` filtered the search: entries of other kinds it
    # hid that are connected or cover more of the query than ``hits`` do, and
    # how many there were before the cap.
    other_kinds: list[SearchHit] = []
    hidden_by_kind: int = 0

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
        prefer_mcp: bool = False,
    ) -> SearchResult:
        """Ranked entries for *query*.

        Integrations such as Gmail, Slack and Sheets are blocks, so
        ``kind="tool"`` hides every one of them and the model got unrelated
        tools, or nothing, and went looking for an MCP server off the
        catalog (SECRT-2820).  A filtered search therefore also runs without
        the filter and returns what it hid in ``other_kinds``.
        """
        result = self._search(
            query,
            context=context,
            kind=kind,
            connections=connections,
            permissions=permissions,
            limit=limit,
            fallback_limit=fallback_limit,
            prefer_mcp=prefer_mcp,
        )
        if kind is not None and result.query:
            self._add_other_kinds(
                result,
                context=context,
                kind=kind,
                connections=connections,
                permissions=permissions,
                prefer_mcp=prefer_mcp,
            )
        return result

    def _search(
        self,
        query: str,
        *,
        context: CapabilityContext,
        kind: CapabilityKindName | None,
        connections: ConnectionState | None,
        permissions: "CopilotPermissions | None",
        limit: int,
        fallback_limit: int,
        prefer_mcp: bool = False,
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
        service, service_indices, service_names = self._service_query(query)
        rest = [idx for idx in scores if idx not in exact]
        fallback: list[SearchHit] = []
        if service_indices is not None:
            # A platform tool is first-party and belongs to no single service,
            # so it carries no service tag and the restriction dropped it:
            # "post a message to discord" could not return
            # post_to_chat_platform at all, only the Discord blocks.  A tool
            # that names the service in its own text is about that service and
            # belongs here; one that merely shares a verb ("read a notion
            # page" matching read_workspace_file) does not.  A skill is not a
            # service either, but the one written for the named service is
            # the best answer there is, so skills stay in.
            main = [
                idx
                for idx in rest
                if idx in service_indices
                or self.entries[idx].kind == "skill"
                or self._tool_names(idx, service_names)
            ]
            chosen = set(main)
            others = [idx for idx in rest if idx not in chosen]
            fallback = _ranked(
                [
                    to_hit(idx, "search")
                    for idx in others
                    if self.entries[idx].klass == "primitive"
                ]
            )[:fallback_limit]
            unpinned = self._unpinned_exact(query, exact, main, connections)
            if unpinned:
                hits = [
                    to_hit(idx, "exact_name") for idx in exact if idx not in unpinned
                ]
                main += unpinned
        else:
            main = rest
        hits += _ranked(
            [to_hit(idx, "search") for idx in main],
            lift=service_indices is not None,
            mcp_first=prefer_mcp
            and service_indices is not None
            and any(self.entries[idx].kind == "mcp_server" for idx in main),
        )
        return SearchResult(
            query=query, hits=hits[:limit], fallback=fallback, service=service
        )

    def _add_other_kinds(
        self,
        result: SearchResult,
        *,
        context: CapabilityContext,
        kind: CapabilityKindName,
        connections: ConnectionState | None,
        permissions: "CopilotPermissions | None",
        prefer_mcp: bool,
    ) -> None:
        """Fill ``result.other_kinds`` from the same search without *kind*.

        An entry of another kind counts when the user has connected it or it
        covers more of the query than anything the filter kept, and among
        equal coverage a connected one is listed first: those are what the
        filter hid.  A lookup by name (an exact hit) found what it asked for
        and gets nothing: most ``kind`` calls are those, and they must not be
        told to search again.  The second search is uncapped: entries of
        *kind* fill its top slots and are then dropped, so a capped one would
        lose a connected entry ranked just below them and undercount
        ``hidden_by_kind``.
        """
        if any(hit.reason != "search" for hit in result.hits):
            return
        best = max((hit.coverage for hit in result.hits), default=0.0)
        unfiltered = self._search(
            result.query,
            context=context,
            kind=None,
            connections=connections,
            permissions=permissions,
            limit=len(self.entries),
            fallback_limit=0,
            prefer_mcp=prefer_mcp,
        )
        hidden = sorted(
            (
                hit
                for hit in unfiltered.hits
                if hit.entry.kind != kind and (hit.connected or hit.coverage > best)
            ),
            key=lambda hit: (-hit.coverage, not hit.connected),
        )
        result.other_kinds = hidden[:OTHER_KINDS_LIMIT]
        result.hidden_by_kind = len(hidden)

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

    def _tool_names(self, idx: int, service_names: list[str]) -> bool:
        """Whether entry *idx* is a platform tool whose name, purpose or tags
        name one of the services the query asked for.

        Not the whole description: connect_integration says "do NOT call it
        for Google, Gmail, Slack", and matching that put it in every Google,
        Slack and GitHub query.  A connection helper never joins, whatever it
        names: connect_integration's purpose lists "github", and it led every
        GitHub query over the GitHub blocks and MCP server."""
        entry = self.entries[idx]
        if entry.kind != "tool" or entry.name in CONNECTION_HELPER_TOOLS:
            return False
        document = {
            *tokenize(entry.name),
            *tokenize(entry.purpose),
            *(token for tag in entry.tags for token in tokenize(tag)),
        }
        for name in service_names:
            tokens = tokenize(name.replace("_", " "))
            if tokens and all(token in document for token in tokens):
                return True
        return False

    def _unpinned_exact(
        self,
        query: str,
        exact: list[int],
        main: list[int],
        connections: ConnectionState | None,
    ) -> list[int]:
        """Exact-name hits that rank with the rest instead of leading.

        An exact name normally leads the list, but "Linear create issue" is
        how AutoPilot phrases a job, not a request for the block that happens
        to be called that, and the pin put ``LinearCreateIssueBlock`` over
        the user's connected Linear MCP server.  So a block the user holds no
        credential for is ranked like any other hit when the service query
        also found a connected server or a platform tool.  A query that
        spells the class name ("LinearCreateIssueBlock", "linear create
        issue block") still asks for the block and keeps the pin, and so does
        a block the user has connected.
        """
        words = query.lower().split()
        if len(words) < 2 or words[-1] == "block":
            return []
        # An exact hit can be the alternative too: a tool sharing the block's
        # name is pinned itself, so it is never in *main*.
        alternative = any(
            self.entries[idx].kind == "tool"
            or (
                self.entries[idx].kind == "mcp_server"
                and resolve_connected(self.entries[idx], connections)
            )
            for idx in (*exact, *main)
        )
        if not alternative:
            return []
        return [
            idx
            for idx in exact
            if self.entries[idx].kind == "block"
            and resolve_connected(self.entries[idx], connections) is False
        ]

    def _service_query(
        self, query: str
    ) -> tuple[str | None, set[int] | None, list[str]]:
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
            return None, None, []
        return " ".join(named), indices, named


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


def _ranked(
    hits: list[SearchHit], *, lift: bool = False, mcp_first: bool = False
) -> list[SearchHit]:
    """Coverage first, then a connected capability, then a platform tool,
    then the class-weighted BM25 score (``score`` already carries the weight).

    With *lift* a platform tool comes before connection.  The list is then
    the named service's own entries plus the tools that name it, so a tool
    tying on coverage is the first-party way to do that service's job
    (post_to_chat_platform for Discord).  On a query that names no service a
    tool that ties has only shared a word with it ("email" and
    create_feature_request), and putting it first moved 19 such queries off
    the user's connected blocks.

    With *lift*, a connected MCP server within
    :data:`CONNECTED_COVERAGE_MARGIN` of the best coverage is read as covering
    the query as well as the best match.  One catalog entry stands for a whole
    server, so its text describes the service rather than each action it
    offers, and it can never match an action verb the way a block named for
    that one action does — "LinearCreateIssueBlock" tokenises to the whole of
    "create linear issue".  Without the lift the block always won on coverage
    and the connection signal was never reached.

    The caller lifts only on a query that names a service.  The main list is
    then restricted to that service, so the server has already matched the
    service's name and the concept it lacks is the action.  On a query that
    names no service the server matched some other word — "updates" in
    Linear's description for "update spreadsheet" — and lifting it put Linear
    over the Sheets block that matched both words, for any server the user
    happened to have connected.

    Only MCP servers are lifted, because only they have that problem: a block
    is named for its action and competes on coverage honestly. Lifting every
    connected entry carried whole families of connected blocks over a
    first-party tool that matched the query better. Nothing is pushed down, so
    a job with no connected server ranks exactly as it did before.

    ``mcp_first`` is for an expert chat that named a service and found its
    MCP server: connection leads, and within a tier the service's MCP server
    comes before its blocks, which stay listed for what the server cannot do.
    It replaces the order above, *lift* included.
    """
    if mcp_first:
        ranked = sorted(
            hits,
            key=lambda h: (
                tier(h.entry, h.connected),
                -h.coverage,
                h.entry.kind != "tool",
                -h.score,
                h.entry.name.lower(),
            ),
        )
        service_hits: dict[str, list[SearchHit]] = defaultdict(list)
        for hit in ranked:
            if hit.entry.service:
                service_hits[hit.entry.service].append(hit)
        for service, entries in service_hits.items():
            service_hits[service] = sorted(
                entries,
                key=lambda h: (
                    tier(h.entry, h.connected),
                    h.entry.kind != "mcp_server",
                    -h.coverage,
                    h.entry.kind != "tool",
                    -h.score,
                    h.entry.name.lower(),
                ),
            )
        offsets: dict[str, int] = defaultdict(int)
        result: list[SearchHit] = []
        for hit in ranked:
            service = hit.entry.service
            if not service:
                result.append(hit)
                continue
            result.append(service_hits[service][offsets[service]])
            offsets[service] += 1
        return result
    best = max((h.coverage for h in hits), default=0.0)

    def key(hit: SearchHit) -> tuple[float, int, int, float, str]:
        rank = tier(hit.entry, hit.connected)
        not_tool = int(hit.entry.kind != "tool")
        coverage = hit.coverage
        if (
            lift
            and hit.entry.kind == "mcp_server"
            and rank == 0
            and coverage >= best - CONNECTED_COVERAGE_MARGIN
        ):
            coverage = best
        first, second = (not_tool, rank) if lift else (rank, not_tool)
        return (-coverage, first, second, -hit.score, hit.entry.name.lower())

    return sorted(hits, key=key)
