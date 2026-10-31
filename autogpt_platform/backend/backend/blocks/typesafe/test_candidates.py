import gc
import weakref
from typing import Any

from backend.blocks.typesafe import pick_best


class TrackedDescription(str):
    pass


def test_candidate_options_releases_full_descriptions_during_iteration(monkeypatch):
    serialize = pick_best.json.dumps
    descriptions: list[weakref.ReferenceType[TrackedDescription]] = []
    candidates = ["é" * 3000, {"text": "b" * 4000}, "short", [1, 2]]
    expected = {
        f"candidate_{index + 1}": serialize(
            candidate, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )[: pick_best.CANDIDATE_DESCRIPTION_LIMIT]
        for index, candidate in enumerate(candidates)
    }

    def track_description(candidate: Any, **kwargs) -> str:
        gc.collect()
        assert sum(reference() is not None for reference in descriptions) <= 1
        description = TrackedDescription(serialize(candidate, **kwargs))
        descriptions.append(weakref.ref(description))
        return description

    monkeypatch.setattr(pick_best.json, "dumps", track_description)
    options, note = pick_best.candidate_options(candidates)
    assert list(options.items()) == list(expected.items())
    assert (
        note
        == "Candidate descriptions capped at 2000 characters: candidate_1, candidate_2."
    )
