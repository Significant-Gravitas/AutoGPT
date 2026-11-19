from pathlib import Path

import pytest

from backend.util import architecture_test as rules


@pytest.mark.parametrize(
    "rule, source",
    [
        (
            "test_patterns_from_input_go_through_compile_user_regex",
            "from regex import compile as c\nc(pattern)",
        ),
        (
            "test_patterns_from_input_go_through_compile_user_regex",
            "import ｒｅ as r\np = input_data.pattern\nr.search(p, text)",
        ),
        (
            "test_patterns_from_input_go_through_compile_user_regex",
            "import re\nif x:\n    q = input_data.pattern\np = q\nre.search(p, text)",
        ),
        (
            "test_only_cli_io_writes_to_the_cli_stdin",
            "from claude_agent_sdk import query as q",
        ),
        (
            "test_only_cli_io_writes_to_the_cli_stdin",
            "import ｃｌａｕｄｅ_ａｇｅｎｔ_ｓｄｋ as sdk\nclient._transport",
        ),
        (
            "test_zip_members_and_workbooks_are_read_through_the_capped_readers",
            "with zipfile.ZipFile(f) as z:\n    z.read('a')",
        ),
        (
            "test_zip_members_and_workbooks_are_read_through_the_capped_readers",
            "def f(z: 'Z' 'ip' 'File'):\n    z.ｅｘｔｒａｃｔａｌｌ(d)",
        ),
        (
            "test_zip_members_and_workbooks_are_read_through_the_capped_readers",
            r"def f(z: '\x5aipFile'):" + "\n    z.open('a')",
        ),
        (
            "test_zip_members_and_workbooks_are_read_through_the_capped_readers",
            "pd.ｒｅａｄ_ｅｘｃｅｌ(f)",
        ),
        (
            "test_sandbox_output_is_read_through_a_limit",
            "sandbox.ｆｉｌｅｓ.read(p)",
        ),
        (
            "test_sandbox_output_is_read_through_a_limit",
            "sandbox.commands.run(cmd)",
        ),
        (
            "test_sandbox_output_is_read_through_a_limit",
            "from e2b_code_interpreter import AsyncSandbox as S\ncreate_sandbox(S)",
        ),
        (
            "test_sandbox_output_is_read_through_a_limit",
            "from e2b_code_interpreter import AsyncSandbox as S\nconnect_sandbox(S)",
        ),
        (
            "test_org_ledger_is_read_only_where_it_is_decided",
            "def f():\n    return db.ｇｅｔ_ｏｒｇ_ｃｒｅｄｉｔｓ(user)",
        ),
        ("test_no_csv_sniffer", "from csv import Sniffer as S\nS()"),
        ("test_no_csv_sniffer", "import csv as c\nc.Ｓｎｉｆｆｅｒ()"),
    ],
)
def test_rules_read_added_changed_and_deleted_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rule: str, source: str
):
    monkeypatch.setattr(rules, "BACKEND_ROOT", tmp_path)
    monkeypatch.setattr(rules, "_ORG_LEDGER_CALL_SITES", {})
    path = tmp_path / "blocks/example.py"
    path.parent.mkdir()
    check = getattr(rules, rule)
    path.write_text("value = 1\n", encoding="utf-8")
    check()
    path.write_text(source, encoding="utf-8")
    with pytest.raises(AssertionError):
        check()
    path.write_text("value = 2\n", encoding="utf-8")
    check()
    path.unlink()
    check()
