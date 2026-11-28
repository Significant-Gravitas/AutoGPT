import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langfuse.api.resources.commons.errors.not_found_error import NotFoundError

from scripts import sync_copilot_prompt_to_langfuse as sync

NAME = "CoPilot Prompt"
REPO = "You are Otto. " + "x" * 2500
OLD = "You are Otto. " + "y" * 2500
DEV = "You are Otto. " + "z" * 2500


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


def test_credentials_fall_back_to_the_default_host_when_it_is_blank(monkeypatch):
    # CI passes an unset secret as an empty string, which would beat the default.
    # Settings is patched, not the env: it reads the env once, on first import.
    secrets = SimpleNamespace(
        langfuse_public_key="pk", langfuse_secret_key="sk", langfuse_host=""
    )
    monkeypatch.setattr(
        "backend.util.settings.Settings", lambda: SimpleNamespace(secrets=secrets)
    )
    assert sync.load_credentials() == ("pk", "sk", "https://cloud.langfuse.com")


def test_main_drops_a_blank_host_before_importing_the_service(monkeypatch):
    # Importing backend.copilot.service builds the process's Langfuse client,
    # which the SDK reuses for every later client with the same key, and the SDK
    # takes LANGFUSE_HOST="" over its default host.
    monkeypatch.setenv("LANGFUSE_HOST", "")
    hosts_at_import = []

    def load_repo_prompt():
        hosts_at_import.append(os.environ.get("LANGFUSE_HOST"))
        return REPO

    monkeypatch.setattr(sync, "load_repo_prompt", load_repo_prompt)
    monkeypatch.setattr(sync, "load_credentials", lambda: ("pk", "sk", "host"))
    monkeypatch.setattr(sync, "make_client", lambda *_: FakeLangfuse({}))
    monkeypatch.setattr(sync, "load_prompt_name", lambda: NAME)
    assert sync.main(["--check"]) == 0
    assert hosts_at_import == [None]


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


def test_publish_moves_latest_back_after_a_revert():
    client = FakeLangfuse({1: (REPO, ["production"]), 2: (OLD, ["latest"])})
    assert sync.publish_latest(client, NAME, REPO).version == 3

    client.create_prompt.assert_called_once()
    assert client.versions[3] == [REPO, ["latest"]]
    assert client.versions[1][1] == ["production"]


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

    assert client.create_prompt.call_count == 2
    client.update_prompt.assert_called_once_with(
        name=NAME, version=2, new_labels=["production"]
    )
    assert client.versions[2] == [REPO, ["production"]]
    assert client.versions[3] == [OLD, ["latest"]]


def test_promote_of_a_master_only_text_keeps_latest_on_dev_text():
    client = FakeLangfuse({1: (OLD, ["production"]), 2: (DEV, ["latest"])})
    sync.promote_production(client, NAME, REPO)

    production = sync.fetch_label(client, NAME, "production")
    latest = sync.fetch_label(client, NAME, "latest")
    assert production is not None and production.text == REPO
    assert latest is not None and latest.text == DEV
    restored = client.create_prompt.call_args.kwargs
    assert "v2" in restored["commit_message"]


def test_publish_puts_latest_back_after_a_failed_restore(capsys):
    client = FakeLangfuse({1: (OLD, ["production"]), 2: (DEV, ["latest"])})
    creates = []

    def create_then_fail(**kwargs):
        creates.append(kwargs)
        if len(creates) == 2:
            raise ConnectionError("Langfuse went away")
        return client._create(**kwargs)

    client.create_prompt.side_effect = create_then_fail
    with pytest.raises(ConnectionError):
        sync.promote_production(client, NAME, REPO)
    assert "--publish-latest on dev" in capsys.readouterr().err
    sync.publish_latest(client, NAME, DEV)

    production = sync.fetch_label(client, NAME, "production")
    latest = sync.fetch_label(client, NAME, "latest")
    assert production is not None and production.text == REPO
    assert latest is not None and latest.text == DEV


def test_master_rerun_keeps_latest_once_dev_has_the_hotfix():
    client = FakeLangfuse({1: (OLD, ["production"]), 2: (DEV, ["latest"])})
    creates = []

    def create_then_fail(**kwargs):
        creates.append(kwargs)
        if len(creates) == 2:
            raise ConnectionError("Langfuse went away")
        return client._create(**kwargs)

    client.create_prompt.side_effect = create_then_fail
    with pytest.raises(ConnectionError):
        sync.promote_production(client, NAME, REPO)
    # Dev merges the hotfix back, so latest already holds dev's text.
    sync.publish_latest(client, NAME, REPO)
    sync.promote_production(client, NAME, REPO)

    latest = sync.fetch_label(client, NAME, "latest")
    assert latest is not None and latest.text == REPO


def test_failed_production_move_still_puts_latest_back():
    client = FakeLangfuse({1: (OLD, ["production"]), 2: (DEV, ["latest"])})
    client.update_prompt.side_effect = [
        TimeoutError("Langfuse timed out"),
        client._update,
    ]
    with pytest.raises(TimeoutError):
        sync.promote_production(client, NAME, REPO)

    latest = sync.fetch_label(client, NAME, "latest")
    assert latest is not None and latest.text == DEV
    # The rerun finds the version the failed run created and only moves the label.
    client.update_prompt.side_effect = client._update
    sync.promote_production(client, NAME, REPO)
    production = sync.fetch_label(client, NAME, "production")
    latest = sync.fetch_label(client, NAME, "latest")
    assert production is not None and production.text == REPO
    assert latest is not None and latest.text == DEV


def test_failed_production_move_and_restore_raises_the_move_error(capsys):
    client = FakeLangfuse({1: (OLD, ["production"]), 2: (DEV, ["latest"])})
    client.update_prompt.side_effect = TimeoutError("Langfuse timed out")
    creates = []

    def create_then_fail(**kwargs):
        creates.append(kwargs)
        if len(creates) == 2:
            raise ConnectionError("Langfuse went away")
        return client._create(**kwargs)

    client.create_prompt.side_effect = create_then_fail
    with pytest.raises(TimeoutError):
        sync.promote_production(client, NAME, REPO)
    err = capsys.readouterr().err
    assert "--publish-latest on dev" in err
    assert "production is done" not in err


def test_promote_is_noop_when_production_matches():
    client = FakeLangfuse({1: (REPO, ["production"]), 2: (OLD, ["latest"])})
    sync.promote_production(client, NAME, REPO)
    assert not client.wrote()


def test_repo_prompt_passes_the_guards():
    sync.check_prompt_text(sync.load_repo_prompt())
