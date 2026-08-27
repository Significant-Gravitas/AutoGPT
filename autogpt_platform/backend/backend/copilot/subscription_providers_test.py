"""The provider table is a contract, not a convenience.

Everything downstream -- the picker chip, the Settings row, the offer that
explains an upsell -- reads these rows and shows them to a person. A row with
a missing label or a lock with nowhere to go does not crash; it renders wrong,
which is harder to notice. These tests are the shape check the type system
cannot do.
"""

import pytest

from backend.copilot.subscription_providers import (
    is_enabled,
    known_profiles,
    linked_profiles,
    profile_for,
)


def test_every_provider_can_describe_itself() -> None:
    """No row may ship with a blank the UI would render."""
    for profile in known_profiles():
        assert profile.display_name.strip(), f"{profile.key} has no display name"
        assert profile.provider_family.strip(), f"{profile.key} has no family"
        assert profile.auth_method.strip(), f"{profile.key} has no auth method"
        assert profile.backed_by_label.strip(), f"{profile.key} says nothing backs it"
        assert profile.description.strip(), f"{profile.key} has no description"


def test_a_lock_says_why_and_where() -> None:
    """A gated provider that cannot explain itself is a dead end.

    Telling someone they may not use a connection, without saying what would
    change that, is worse than not offering it at all.
    """
    for profile in known_profiles():
        if profile.entitlement is None:
            continue
        assert profile.lock_reason, f"{profile.key} is gated but gives no reason"
        assert profile.unlock_href, f"{profile.key} is gated with nowhere to go"


def test_linked_providers_carry_a_credential_and_the_platform_does_not() -> None:
    """The one invariant the rest of the routing code leans on.

    `save_default_chat_route` and the session-creation path both decide
    "credential required?" from this distinction rather than from the provider
    name, so a row that breaks it would make those checks wrong.
    """
    for profile in known_profiles():
        if profile.credential_strategy == "platform":
            assert profile.credential_provider is None
            assert profile.entitlement is None, "the built-in route cannot be gated"
        else:
            assert profile.credential_provider is not None, (
                f"{profile.key} is a linked account but names no credential provider"
            )


def test_linked_is_what_this_deployment_offers() -> None:
    """`linked_profiles` answers "what can a user connect here", which is not
    the same as "what providers exist" -- an operator may not have opted into
    all of them. The platform route is never in it either way: there is no
    account to link."""
    linked = {p.key for p in linked_profiles()}
    everything = {p.key for p in known_profiles()}
    assert "platform" in everything
    assert "platform" not in linked
    assert linked <= everything - {"platform"}
    # Everything not gated behind an opt-in is always on offer.
    always = {
        p.key
        for p in known_profiles()
        if p.opt_in_env is None and p.credential_strategy != "platform"
    }
    assert always <= linked


def test_a_provider_that_runs_a_sign_in_allows_time_for_it() -> None:
    """A browser sign-in cannot answer inside a normal request timeout."""
    for profile in linked_profiles():
        assert profile.login_timeout_seconds, (
            f"{profile.key} needs a login timeout or its sign-in will be cut off"
        )


def test_an_unknown_provider_raises_rather_than_guessing() -> None:
    """Falling back to the platform row would name the wrong plan on someone's
    screen, and bill against a route they did not pick."""
    with pytest.raises(ValueError):
        profile_for("not-a-provider")


def test_keys_are_unique() -> None:
    keys = [p.key for p in known_profiles()]
    assert len(keys) == len(set(keys))


class TestOperatorOptIn:
    """A provider an operator has not opted into is not offered at all.

    This is the mechanism behind the one provider whose sign-in works but is
    not ours to enable on someone's behalf. If it ever defaults to on, that
    is a decision made for an operator about their own relationship with a
    vendor -- so it is asserted rather than left to a default nobody rereads.
    """

    def test_a_gated_provider_is_off_until_the_operator_says_otherwise(
        self, monkeypatch
    ) -> None:
        gated = [p for p in known_profiles() if p.opt_in_env]
        assert gated, "expected at least one opt-in provider"
        for profile in gated:
            monkeypatch.delenv(profile.opt_in_env, raising=False)
            assert not is_enabled(profile)
            assert profile.key not in {p.key for p in linked_profiles()}

    def test_the_operator_can_turn_it_on(self, monkeypatch) -> None:
        for profile in [p for p in known_profiles() if p.opt_in_env]:
            monkeypatch.setenv(profile.opt_in_env, "true")
            assert is_enabled(profile)
            assert profile.key in {p.key for p in linked_profiles()}

    def test_only_a_truthy_value_counts(self, monkeypatch) -> None:
        """ "false" must not read as "set, therefore on"."""
        for profile in [p for p in known_profiles() if p.opt_in_env]:
            for value in ("", "false", "0", "no", "off"):
                monkeypatch.setenv(profile.opt_in_env, value)
                assert not is_enabled(profile), f"{value!r} enabled {profile.key}"

    def test_everything_else_needs_no_permission(self) -> None:
        for profile in known_profiles():
            if profile.opt_in_env is None:
                assert is_enabled(profile)
