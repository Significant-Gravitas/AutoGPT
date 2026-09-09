import pytest

from .categories import (
    CATEGORY_DESCRIPTIONS,
    CATEGORY_LABELS,
    StoreCategory,
    all_category_match_values,
    category_match_values,
    normalize_categories,
    normalize_category,
    validate_canonical_categories,
)


def test_every_category_has_a_label_and_description():
    for category in StoreCategory:
        assert CATEGORY_LABELS[category]
        assert CATEGORY_DESCRIPTIONS[category]


def test_the_canonical_set_is_the_eight_the_product_promises():
    assert {category.value for category in StoreCategory} == {
        "marketing",
        "sales",
        "finance",
        "support",
        "operations",
        "research",
        "content",
        "development",
    }


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("marketing", StoreCategory.MARKETING),
        ("  Marketing  ", StoreCategory.MARKETING),
        ("MARKETING", StoreCategory.MARKETING),
        ("writing", StoreCategory.CONTENT),
        ("Creative & Design", StoreCategory.CONTENT),
        ("productivity", StoreCategory.OPERATIONS),
        ("", None),
        ("   ", None),
        ("business", None),
        ("personal", None),
        ("other", None),
        ("testing", None),
    ],
)
def test_normalize_category(raw, expected):
    assert normalize_category(raw) is expected


def test_normalize_categories_drops_unmappable_and_deduplicates():
    assert normalize_categories(["writing", "creative", "testing", "sales"]) == [
        StoreCategory.CONTENT,
        StoreCategory.SALES,
    ]


def test_category_match_values_covers_the_alias_that_folds_onto_it():
    matches = category_match_values("content")
    assert "content" in matches
    assert "writing" in matches
    assert "marketing" not in matches


def test_an_alias_and_its_canonical_form_select_the_same_listings():
    assert set(category_match_values("writing")) == set(
        category_match_values("content")
    )


def test_category_match_values_passes_an_unknown_value_through():
    assert category_match_values("testing") == ["testing"]


def test_all_category_match_values_holds_every_canonical_value():
    matches = set(all_category_match_values())
    assert {category.value for category in StoreCategory} <= matches
    assert "writing" in matches
    assert "testing" not in matches


def test_validate_canonical_categories_normalizes():
    assert validate_canonical_categories(["Writing", "sales"]) == ["content", "sales"]


@pytest.mark.parametrize("values", [[], [""], ["testing"], ["other"]])
def test_validate_canonical_categories_rejects(values):
    with pytest.raises(ValueError):
        validate_canonical_categories(values)
