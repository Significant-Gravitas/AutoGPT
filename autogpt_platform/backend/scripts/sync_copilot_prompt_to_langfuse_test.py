from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langfuse.api.resources.commons.errors.not_found_error import NotFoundError

from scripts import sync_copilot_prompt_to_langfuse as sync

NAME = "CoPilot Prompt"
REPO = "You are Otto. " + "x" * 2500
OLD = "You are Otto. " + "y" * 2500


class FakeLangfuse:
    """In-memory Langfuse holding versions of one prompt, with labels."""

    def __init__(self, versions: dict[int, tuple[str, list[str]]]):
        self.versions = {
            v: [text, list(labels)] for v, (text, labels) in versions.items()
        }
        self.api = MagicMock()
        self.api.prompts.get.side_effect = self._get
        self.api.prompts.list.side_effect = self._list
        self.create_prompt = MagicMock(side_effect=self._create)
        self.update_prompt = MagicMock(side_effect=self._update)

    def _prompt(self, version: int) -> SimpleNamespace:
        text, labels = self.versions[version]
        return SimpleNamespace(version=version, prompt=text, labels=list(labels))

    def _get(self, name, *, version=None, label=None):
        assert name == "CoPilot%20Prompt"
        if version is not None:
            return self._prompt(version)
        for v, (_, labels) in self.versions.items():
            if label in labels:
                return self._prompt(v)
        raise NotFoundError(body="not found")

    def _list(self, *, name, limit):
        if not self.versions:
            return SimpleNamespace(data=[])
        return SimpleNamespace(data=[SimpleNamespace(versions=list(self.versions))])

    def _create(self, *, name, prompt, type, labels, config, commit_message):
        for entry in self.versions.values():
            if "latest" in entry[1]:
                entry[1].remove("latest")
        version = max(self.versions, default=0) + 1
        self.versions[version] = [prompt, [*labels, "latest"]]
        return self._prompt(version)

    def _update(self, *, name, version, new_labels):
        for entry in self.versions.values():
            entry[1] = [lb for lb in entry[1] if lb not in new_labels]
        self.versions[version][1] = list(
            dict.fromkeys([*self.versions[version][1], *new_labels])
        )

    def wrote(self) -> bool:
        return self.create_prompt.called or self.update_prompt.called


@pytest.mark.parametrize(
    "text",
    [
        "",
        "short prompt",
        REPO + "{{name}}",
        REPO + "}}",
        REPO + "{users_information}",
    ],
)
def test_guards_refuse_bad_text(text):
    with pytest.raises(sync.GuardError):
        sync.check_prompt_text(text)


def test_guard_accepts_repo_text():
    sync.check_prompt_text(REPO)


def test_main_refuses_before_any_client_when_guard_fails(monkeypatch):
    make_client = MagicMock()
    monkeypatch.setattr(sync, "load_repo_prompt", lambda: "{{x}}" * 1000)
    monkeypatch.setattr(sync, "make_client", make_client)
    assert sync.main(["--publish-latest"]) == 1
    make_client.assert_not_called()


def test_main_refuses_without_credentials(monkeypatch):
    make_client = MagicMock()
    monkeypatch.setattr(sync, "load_repo_prompt", lambda: REPO)
    monkeypatch.setattr(sync, "make_client", make_client)
    monkeypatch.setattr(
        sync,
        "load_credentials",
        MagicMock(side_effect=sync.GuardError("no keys")),
    )
    assert sync.main(["--promote-production"]) == 1
    make_client.assert_not_called()


def test_main_defaults_to_check(monkeypatch):
    client = FakeLangfuse({1: (OLD, ["latest", "production"])})
    monkeypatch.setattr(sync, "load_repo_prompt", lambda: REPO)
    monkeypatch.setattr(sync, "load_credentials", lambda: ("pk", "sk", "host"))
    monkeypatch.setattr(sync, "make_client", lambda *_: client)
    monkeypatch.setattr(sync, "load_prompt_name", lambda: NAME)
    assert sync.main([]) == 0
    assert not client.wrote()


def test_check_reports_stale_labels_and_never_writes(capsys):
    client = FakeLangfuse({1: (OLD, ["production"]), 2: (REPO, ["latest"])})
    assert sync.run_check(client, NAME, REPO) == ["production"]
    assert not client.wrote()
    out = capsys.readouterr().out
    assert "[latest] v2 matches" in out
    assert "[production] v1 differs" in out


def test_check_reports_missing_labels():
    client = FakeLangfuse({})
    assert sync.run_check(client, NAME, REPO) == ["latest", "production"]
    assert not client.wrote()


def test_publish_is_noop_when_latest_matches():
    client = FakeLangfuse({1: (REPO, ["latest", "production"])})
    assert sync.publish_latest(client, NAME, REPO).version == 1
    assert not client.wrote()


def test_publish_creates_once_then_noops(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "abc123")
    client = FakeLangfuse({1: (OLD, ["latest", "production"])})
    created = sync.publish_latest(client, NAME, REPO)
    sync.publish_latest(client, NAME, REPO)

    assert created.version == 2
    client.create_prompt.assert_called_once()
    kwargs = client.create_prompt.call_args.kwargs
    assert kwargs["type"] == "text"
    assert kwargs["labels"] == []
    assert kwargs["config"] == {
        "git_sha": "abc123",
        "text_sha256": sync.text_sha256(REPO),
    }
    assert "abc123" in kwargs["commit_message"]
    assert client.versions[1][1] == ["production"]
    assert client.versions[2][1] == ["latest"]


def test_publish_does_not_duplicate_an_older_identical_version():
    client = FakeLangfuse({1: (REPO, ["production"]), 2: (OLD, ["latest"])})
    assert sync.publish_latest(client, NAME, REPO).version == 1
    assert not client.wrote()


def test_promote_moves_only_the_production_label():
    client = FakeLangfuse(
        {1: (OLD, ["production", "staging"]), 2: (REPO, ["latest", "canary"])}
    )
    sync.promote_production(client, NAME, REPO)

    client.create_prompt.assert_not_called()
    client.update_prompt.assert_called_once_with(
        name=NAME, version=2, new_labels=["canary", "production"]
    )
    assert client.versions[1][1] == ["staging"]
    assert client.versions[2][1] == ["latest", "canary", "production"]


def test_promote_creates_the_version_when_missing_then_noops():
    client = FakeLangfuse({1: (OLD, ["latest", "production"])})
    sync.promote_production(client, NAME, REPO)
    sync.promote_production(client, NAME, REPO)

    client.create_prompt.assert_called_once()
    client.update_prompt.assert_called_once_with(
        name=NAME, version=2, new_labels=["production"]
    )
    assert client.versions[1][1] == []
    assert client.versions[2][1] == ["latest", "production"]


def test_promote_is_noop_when_production_matches():
    client = FakeLangfuse({1: (REPO, ["production"]), 2: (OLD, ["latest"])})
    sync.promote_production(client, NAME, REPO)
    assert not client.wrote()


def test_repo_prompt_passes_the_guards():
    sync.check_prompt_text(sync.load_repo_prompt())
