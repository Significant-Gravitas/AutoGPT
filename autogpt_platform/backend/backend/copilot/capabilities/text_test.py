import pytest

from .text import normalize_name, query_groups, stem, tokenize


@pytest.mark.parametrize(
    "a,b",
    [
        ("issue", "issues"),
        ("generate", "generation"),
        ("generate", "generator"),
        ("execute", "execution"),
        ("execute", "executing"),
        ("format", "formatter"),
        ("format", "formatting"),
        ("condition", "conditional"),
        ("query", "queries"),
        ("search", "searches"),
        ("call", "calling"),
        ("run", "running"),
    ],
)
def test_stem_maps_word_forms_together(a: str, b: str):
    assert stem(a) == stem(b)


def test_tokenize_splits_camel_case_and_drops_block():
    assert tokenize("LinearCreateIssueBlock") == ["linear", "creat", "issu"]
    assert tokenize("Send an email to the user") == ["send", "email", "user"]


def test_query_groups_expand_synonyms_once_per_concept():
    groups = query_groups("run python code")
    assert groups[0][0] == "run" and stem("execute") in groups[0]
    assert groups[1][0] == "python" and stem("code") in groups[1]
    assert groups[2] == [stem("code")]


def test_query_groups_link_save_store_and_write():
    """Saving, storing and writing are one intent across three entries
    (FileStoreBlock, write_workspace_file, memory_store); a query using any
    of the verbs must reach the entries described with the others."""
    for verb in ("save", "store", "write"):
        (group,) = query_groups(verb)
        assert group[0] == stem(verb)
        assert {stem("save"), stem("store"), stem("write")} <= set(group)


def test_normalize_name():
    assert normalize_name("Orchestrator Block") == "orchestratorblock"
