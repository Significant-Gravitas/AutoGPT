"""Unit tests for the shared Websets request-body builders."""

import pytest

from backend.blocks.exa.websets_payloads import (
    criteria_payload,
    enrichment_payload,
    entity_payload,
    import_payload,
    options_payload,
    scope_payload,
    sources_payload,
)


class TestEntityPayload:
    def test_auto_and_empty_let_exa_detect(self):
        assert entity_payload("auto", None) is None
        assert entity_payload(None, None) is None

    def test_fixed_entity(self):
        assert entity_payload("person", "ignored") == {"type": "person"}

    def test_custom_entity_carries_description(self):
        assert entity_payload("custom", " LLM repos ") == {
            "type": "custom",
            "description": "LLM repos",
        }

    @pytest.mark.parametrize("description", [None, "x", "y" * 201])
    def test_custom_entity_needs_2_to_200_characters(self, description):
        with pytest.raises(ValueError, match="2-200 characters"):
            entity_payload("custom", description)


class TestCriteriaPayload:
    def test_empty_means_generate(self):
        assert criteria_payload([]) is None
        assert criteria_payload(["  "]) is None

    def test_more_than_five_is_rejected(self):
        with pytest.raises(ValueError, match="at most 5 criteria"):
            criteria_payload([f"c{i}" for i in range(6)])

    def test_builds_description_objects(self):
        assert criteria_payload(["a", "b"]) == [
            {"description": "a"},
            {"description": "b"},
        ]


class TestSources:
    def test_types_default_to_import(self):
        assert sources_payload(["a", "b"], ["webset"]) == [
            {"source": "webset", "id": "a"},
            {"source": "import", "id": "b"},
        ]

    def test_unknown_type_is_rejected(self):
        with pytest.raises(ValueError, match="'import' or 'webset'"):
            sources_payload(["a"], ["website"])

    def test_import_evaluate_stays_aligned_when_an_id_is_blank(self):
        assert import_payload(["a", "", "c"], [], [True, False, False]) == [
            {"source": "import", "id": "a", "evaluate": True},
            {"source": "import", "id": "c", "evaluate": False},
        ]


class TestScopePayload:
    def test_relationship_becomes_hop_search(self):
        assert scope_payload(["w1", "i1"], ["webset"], ["investors of"], [3]) == [
            {
                "source": "webset",
                "id": "w1",
                "relationship": {"definition": "investors of", "limit": 3},
            },
            {"source": "import", "id": "i1"},
        ]

    @pytest.mark.parametrize("limits", [[], [0], [11]])
    def test_relationship_needs_limit_from_1_to_10(self, limits):
        with pytest.raises(ValueError, match="limit from 1 to 10"):
            scope_payload(["w1"], [], ["investors of"], limits)


class TestEnrichmentPayload:
    def test_url_format_and_options(self):
        assert enrichment_payload("Website", "url", [], None) == {
            "description": "Website",
            "format": "url",
        }
        assert enrichment_payload("Tier", "options", ["A", "B"], {"k": "v"}) == {
            "description": "Tier",
            "format": "options",
            "options": [{"label": "A"}, {"label": "B"}],
            "metadata": {"k": "v"},
        }

    def test_options_cap_is_150(self):
        assert len(options_payload([str(i) for i in range(150)])) == 150
        with pytest.raises(ValueError, match="at most 150 options"):
            options_payload([str(i) for i in range(151)])
