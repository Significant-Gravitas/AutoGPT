"""Collapse near-duplicate writes within one dream pass (#13387)."""

from __future__ import annotations

import re

from .schemas import ConsolidatedFact

# Intra-pass near-duplicate write rejection (#13387). The consolidate
# prompt asks the model to merge near-duplicates, but a single pass still
# emits the same fact phrased 2-3 ways ("Nick uses Terminus on iPhone for
# CLI work" / "Nick wants Terminus on iPhone to show more ASCII"). This is
# a CONSERVATIVE, deterministic backstop: it only collapses writes that are
# near-identical by content-word overlap, keeping the longest (most
# specific) of each cluster. It deliberately does NOT do semantic
# (embedding) merging or compare against the user's EXISTING active facts —
# both need real-data threshold tuning + edge-merge design and are tracked
# as the follow-up P2 dedup pass. High threshold so distinct facts about
# the same entity ("prefers Python" vs "prefers Rust") are never merged.
_DEDUP_JACCARD_THRESHOLD = 0.7
# A short fact whose content words are (nearly) all contained in a longer
# fact is a duplicate the longer one subsumes — even when the longer fact's
# extra detail drags Jaccard below the threshold ("Nick uses Terminus on
# iPhone for CLI" ⊂ "…for CLI and wants more ASCII"). Set high (0.9): for
# facts under ~10 words it effectively requires the shorter to be a strict
# SUBSET of the longer, so a single distinguishing word keeps them apart
# ("deployed the AUTH service" vs "deployed the BILLING service" → 4/5=0.8,
# NOT merged). Only applied when the smaller fact has enough words to be
# specific, so 1-2 word fragments don't spuriously match.
_DEDUP_CONTAINMENT_THRESHOLD = 0.9
_DEDUP_CONTAINMENT_MIN_TOKENS = 3
# Facts with IDENTICAL content-word sets can still describe different
# events when word order differs ("Alice introduced Bob to Carol" vs
# "Alice introduced Carol to Bob"). Equal token sets therefore also
# require agreement on adjacent-word bigrams before merging.
_DEDUP_BIGRAM_THRESHOLD = 0.6
# A negation marker present in one fact but not the other flips the
# meaning ("uses vim" vs "never uses vim") — never merge across it.
# The tokenizer splits contractions ("doesn't" → "doesn", "t"), so the
# bare stems cover the apostrophe forms; a false veto only ever KEEPS
# both facts, which is the conservative direction.
_DEDUP_NEGATION_MARKERS = frozenset(
    {
        "no",
        "not",
        "never",
        "none",
        "nor",
        "cannot",
        "nt",
        "dont",
        "don",
        "doesnt",
        "doesn",
        "didnt",
        "didn",
        "isnt",
        "isn",
        "wasnt",
        "wasn",
        "werent",
        "weren",
        "wont",
        "cant",
        "hasnt",
        "hasn",
        "havent",
        "haven",
        "hadnt",
        "hadn",
        "wouldnt",
        "wouldn",
        "shouldnt",
        "shouldn",
        "couldnt",
        "couldn",
        "arent",
        "aren",
    }
)
_DEDUP_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "to",
        "of",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "for",
        "on",
        "in",
        "at",
        "by",
        "with",
        "that",
        "this",
        "it",
        "their",
        "they",
        "them",
        "his",
        "her",
        "its",
        "as",
        "be",
        "has",
        "have",
        "had",
        "s",
        "so",
        "more",
        "user",
        "users",
    }
)


def _content_tokens(content: str) -> frozenset[str]:
    """Lowercased alphanumeric content words, minus a small stopword set.
    ``user`` is a stopword because nearly every fact begins with it."""
    return frozenset(
        t
        for t in re.findall(r"[a-z0-9]+", content.lower())
        if t not in _DEDUP_STOPWORDS
    )


def _word_bigrams(content: str) -> frozenset[tuple[str, str]]:
    """Adjacent word pairs over the full (unfiltered) word sequence — the
    order-sensitive signal used when two facts' token sets are equal."""
    words = re.findall(r"[a-z0-9]+", content.lower())
    return frozenset(zip(words, words[1:]))


def _near_duplicate(
    a: frozenset[str],
    b: frozenset[str],
    a_bigrams: frozenset[tuple[str, str]],
    b_bigrams: frozenset[tuple[str, str]],
) -> bool:
    """Two content-word sets are near-duplicate when their Jaccard overlap
    is high, OR the smaller (sufficiently specific) set is nearly contained
    in the larger one. A negation marker on only one side always vetoes;
    identical sets additionally need word-order (bigram) agreement."""
    if not a or not b:
        return False
    if _DEDUP_NEGATION_MARKERS & (a ^ b):
        return False
    if a == b:
        bigram_union = len(a_bigrams | b_bigrams)
        if not bigram_union:
            return True
        return len(a_bigrams & b_bigrams) / bigram_union >= _DEDUP_BIGRAM_THRESHOLD
    inter = len(a & b)
    union = len(a | b)
    if union and inter / union >= _DEDUP_JACCARD_THRESHOLD:
        return True
    smaller = min(len(a), len(b))
    if smaller < _DEDUP_CONTAINMENT_MIN_TOKENS:
        return False
    return inter / smaller >= _DEDUP_CONTAINMENT_THRESHOLD


def dedupe_near_duplicate_writes(
    writes: list[ConsolidatedFact],
) -> tuple[list[ConsolidatedFact], int]:
    """Collapse near-duplicate writes, keeping the longest of each cluster.

    Greedy by content length (desc) so the most specific phrasing survives;
    output preserves the writes' original order. Writes are only compared
    within the same ``scope`` (facts must never merge across scopes), and a
    dropped write's ``source_episode_uuids`` are unioned into its cluster's
    survivor so provenance is never lost. Returns (kept, dropped)."""
    by_len_desc = sorted(
        range(len(writes)), key=lambda i: len(writes[i].content), reverse=True
    )
    tokens = [_content_tokens(w.content) for w in writes]
    bigrams = [_word_bigrams(w.content) for w in writes]
    kept_idx: list[int] = []
    absorbed: dict[int, list[str]] = {}
    for i in by_len_desc:
        survivor = next(
            (
                k
                for k in kept_idx
                if writes[k].scope == writes[i].scope
                and _near_duplicate(tokens[i], tokens[k], bigrams[i], bigrams[k])
            ),
            None,
        )
        if survivor is None:
            kept_idx.append(i)
            absorbed[i] = []
        else:
            absorbed[survivor].extend(writes[i].source_episode_uuids)
    kept = [_absorb_provenance(writes[i], absorbed[i]) for i in sorted(kept_idx)]
    return kept, len(writes) - len(kept)


def _absorb_provenance(
    write: ConsolidatedFact, absorbed_uuids: list[str]
) -> ConsolidatedFact:
    """Union dropped near-duplicates' episode uuids into the survivor."""
    extra = [
        u for u in dict.fromkeys(absorbed_uuids) if u not in write.source_episode_uuids
    ]
    if not extra:
        return write
    return write.model_copy(
        update={"source_episode_uuids": [*write.source_episode_uuids, *extra]}
    )
