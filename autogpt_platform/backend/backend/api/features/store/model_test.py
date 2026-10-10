import pydantic
import pytest

from .model import (
    SUB_HEADING_MAX_LENGTH,
    StoreSubmissionEditRequest,
    StoreSubmissionRequest,
)

SUBMISSION_MODELS = [StoreSubmissionRequest, StoreSubmissionEditRequest]


@pytest.mark.parametrize("model", SUBMISSION_MODELS)
@pytest.mark.parametrize("blank", ["", "   ", "\t\n "])
def test_a_blank_sub_heading_is_rejected(model, blank):
    with pytest.raises(pydantic.ValidationError) as exc:
        build(model, blank)
    assert exc.value.errors()[0]["type"] == "string_too_short"


@pytest.mark.parametrize("model", SUBMISSION_MODELS)
def test_a_cta_sub_heading_is_accepted(model):
    cta = "Find decision-makers at any company in seconds"
    assert build(model, cta).sub_heading == cta


@pytest.mark.parametrize("model", SUBMISSION_MODELS)
def test_surrounding_whitespace_is_stripped(model):
    assert build(model, "  Find leads fast  ").sub_heading == "Find leads fast"


@pytest.mark.parametrize("model", SUBMISSION_MODELS)
def test_the_length_ceiling_is_enforced_on_the_stripped_value(model):
    at_limit = "x" * SUB_HEADING_MAX_LENGTH
    assert build(model, f"  {at_limit}  ").sub_heading == at_limit

    with pytest.raises(pydantic.ValidationError) as exc:
        build(model, "x" * (SUB_HEADING_MAX_LENGTH + 1))
    assert exc.value.errors()[0]["type"] == "string_too_long"


@pytest.mark.parametrize("model", SUBMISSION_MODELS)
def test_sub_heading_has_no_default(model):
    """A missing key must 422 rather than silently persisting an empty line."""
    with pytest.raises(pydantic.ValidationError) as exc:
        model.model_validate(
            {
                "graph_id": "graph-1",
                "graph_version": 1,
                "slug": "lead-finder",
                "name": "Lead Finder",
                "categories": ["sales"],
            }
        )
    assert any(e["loc"] == ("sub_heading",) for e in exc.value.errors())


def build(model: type[pydantic.BaseModel], sub_heading: str):
    common = {
        "name": "Lead Finder",
        "sub_heading": sub_heading,
        "categories": ["sales"],
    }
    if model is StoreSubmissionRequest:
        common |= {"graph_id": "graph-1", "graph_version": 1, "slug": "lead-finder"}
    return model(**common)


@pytest.mark.parametrize("score", [0, 6, -5, 1_000_000])
def test_review_score_outside_1_to_5_is_rejected(score):
    """#15304: one review with an unbounded score skews a listing's average."""
    from .model import StoreReviewCreate

    with pytest.raises(pydantic.ValidationError):
        StoreReviewCreate(store_listing_version_id="v1", score=score)


@pytest.mark.parametrize("score", [1, 3, 5])
def test_review_score_within_1_to_5_is_accepted(score):
    from .model import StoreReviewCreate

    assert StoreReviewCreate(store_listing_version_id="v1", score=score).score == score
