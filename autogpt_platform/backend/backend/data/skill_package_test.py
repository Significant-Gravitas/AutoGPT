from backend.data.skill_package import (
    PackageFile,
    merge_packages,
    merge_text,
    package_hash,
    package_tree_sha256,
)


def test_tree_hash_matches_the_catalog_formula():
    """The catalog computes sha256(canonical_json(sorted files)); the same
    inputs must give the same digest here or a copy can never be verified."""
    files = [
        ("references/API.md", "b" * 64, False),
        ("SKILL.md", "a" * 64, False),
        ("scripts/run.py", "c" * 64, True),
    ]
    expected = package_tree_sha256(sorted(files))
    assert package_tree_sha256(files) == expected
    # Independent of input order, sensitive to the executable bit.
    assert package_tree_sha256(reversed(files)) == expected
    assert package_tree_sha256([(p, h, not x) for p, h, x in files]) != expected


def test_package_hash_covers_every_file_including_skill_md():
    package = {
        "SKILL.md": PackageFile(content=b"---\nname: demo\n---\n"),
        "references/a.md": PackageFile(content=b"# a\n"),
    }
    changed = {**package, "SKILL.md": PackageFile(content=b"---\nname: demo\n---\nx\n")}
    assert package_hash(package) != package_hash(changed)
    assert package_hash(package) == package_hash(dict(reversed(package.items())))


def test_merge_text_takes_the_only_side_that_changed():
    base = "a\nb\nc\n"
    assert merge_text(base, base, "a\nB\nc\n") == merge_text(base, base, "a\nB\nc\n")
    assert merge_text(base, base, "a\nB\nc\n").text == "a\nB\nc\n"
    assert merge_text(base, "a\nb\nC\n", base).text == "a\nb\nC\n"


def test_merge_text_combines_edits_in_different_hunks():
    base = "one\ntwo\nthree\nfour\nfive\n"
    ours = "ONE\ntwo\nthree\nfour\nfive\n"
    theirs = "one\ntwo\nthree\nfour\nFIVE\n"
    result = merge_text(base, ours, theirs)
    assert result.text == "ONE\ntwo\nthree\nfour\nFIVE\n"
    assert not result.conflicted


def test_merge_text_keeps_the_users_hunk_on_conflict():
    base = "one\ntwo\nthree\n"
    ours = "one\nTWO (mine)\nthree\n"
    theirs = "one\nTWO (upstream)\nthree\n"
    result = merge_text(base, ours, theirs)
    assert result.text == ours
    assert result.conflicted


def test_merge_text_agreeing_edits_are_not_a_conflict():
    base = "one\ntwo\n"
    result = merge_text(base, "one\nTWO\n", "one\nTWO\n")
    assert (result.text, result.conflicted) == ("one\nTWO\n", False)


def test_merge_text_handles_insertions_and_deletions_around_stable_lines():
    base = "intro\nkeep\noutro\n"
    ours = "intro\nkeep\nmy addition\noutro\n"
    theirs = "new intro\nkeep\noutro\n"
    result = merge_text(base, ours, theirs)
    assert result.text == "new intro\nkeep\nmy addition\noutro\n"
    assert not result.conflicted


def test_merge_text_without_trailing_newline():
    result = merge_text("a\nb", "a\nb\nc", "A\nb")
    assert result.text == "A\nb\nc"
    assert not result.conflicted


def test_merge_packages_fast_forwards_untouched_files_and_keeps_user_edits():
    base = {
        "SKILL.md": PackageFile(content=b"one\ntwo\n"),
        "references/a.md": PackageFile(content=b"a\n"),
        "references/gone.md": PackageFile(content=b"gone\n"),
    }
    ours = {
        "SKILL.md": PackageFile(content=b"one\ntwo\nmine\n"),
        "references/a.md": PackageFile(content=b"a\n"),
        "references/gone.md": PackageFile(content=b"gone\n"),
        "notes.md": PackageFile(content=b"user notes\n"),
    }
    theirs = {
        "SKILL.md": PackageFile(content=b"ONE\ntwo\n"),
        "references/a.md": PackageFile(content=b"a v2\n"),
        "scripts/run.py": PackageFile(content=b"print(1)\n", executable=True),
    }
    merged = merge_packages(base, ours, theirs)
    assert merged.files["SKILL.md"].content == b"ONE\ntwo\nmine\n"
    assert merged.files["references/a.md"].content == b"a v2\n"
    assert "references/gone.md" not in merged.files
    assert merged.files["notes.md"].content == b"user notes\n"
    assert merged.files["scripts/run.py"] == PackageFile(
        content=b"print(1)\n", executable=True
    )
    assert not merged.conflicted


def test_merge_packages_conflicts_resolve_to_the_user():
    base = {
        "a.md": PackageFile(content=b"x\n"),
        "b.md": PackageFile(content=b"y\n"),
        "c.md": PackageFile(content=b"z\n"),
    }
    ours = {
        "a.md": PackageFile(content=b"x mine\n"),
        "c.md": PackageFile(content=b"z mine\n"),
    }
    theirs = {
        "a.md": PackageFile(content=b"x theirs\n"),
        "b.md": PackageFile(content=b"y theirs\n"),
    }
    merged = merge_packages(base, ours, theirs)
    assert merged.files["a.md"].content == b"x mine\n"
    assert "b.md" not in merged.files  # user deleted it; the update's edit loses
    assert (
        merged.files["c.md"].content == b"z mine\n"
    )  # update deleted; user's edit stays
    assert merged.conflicts == ["a.md", "b.md", "c.md"]


def test_merge_packages_binary_conflict_keeps_ours():
    base = {"img.bin": PackageFile(content=b"\xff\x00")}
    ours = {"img.bin": PackageFile(content=b"\xff\x01")}
    theirs = {"img.bin": PackageFile(content=b"\xff\x02")}
    merged = merge_packages(base, ours, theirs)
    assert merged.files["img.bin"].content == b"\xff\x01"
    assert merged.conflicts == ["img.bin"]


def test_merge_packages_executable_bit_follows_the_changed_side():
    base = {"s.py": PackageFile(content=b"x\n", executable=False)}
    ours = {"s.py": PackageFile(content=b"x\n", executable=False)}
    theirs = {"s.py": PackageFile(content=b"x\n", executable=True)}
    assert merge_packages(base, ours, theirs).files["s.py"].executable is True
