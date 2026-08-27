"""xAI's error envelope actively resists the obvious classifier.

`code` carries four different kinds of thing -- a gRPC code, a namespaced
well-known code, a bare integer, and sometimes an entire sentence -- and a
machine code that is not in `code` is appended to the message as
`[WKE=ns:code]` instead. xAI's own parser has a test called
`sentence_shaped_codes_never_become_prefixes`, which is the tell that their
client cannot trust the field either.

The two status rules below look backwards and are load-bearing. 403 in
particular must never trigger a re-auth: xAI's client refuses to, because
doing so races their failed-grant counter and can destroy a working stored
credential -- turning a temporary refusal into a broken connection.
"""

from backend.integrations.grok.errors import (
    MAX_RETRY_AFTER_SECONDS,
    GrokFailure,
    classify,
)


class TestWhereTheCodeHides:
    def test_reads_the_code_field_when_it_holds_one(self) -> None:
        error = classify(429, {"code": "subscription:free-usage-exhausted"})

        assert error.failure is GrokFailure.FREE_QUOTA_EXHAUSTED
        assert error.code == "subscription:free-usage-exhausted"

    def test_reads_the_code_out_of_the_message_when_it_is_not_in_the_field(
        self,
    ) -> None:
        """Appended inline as `[WKE=ns:code]`. A classifier that only reads
        `code` misses these entirely and falls through to "unknown"."""
        error = classify(
            429,
            {
                "code": 7,
                "error": "You have used your allowance "
                "[WKE=subscription:free-usage-exhausted]",
            },
        )

        assert error.failure is GrokFailure.FREE_QUOTA_EXHAUSTED
        assert error.code == "subscription:free-usage-exhausted"

    def test_a_sentence_in_the_code_field_is_not_treated_as_a_code(self) -> None:
        """The failure mode this guards: matching on prose, which turns any
        message containing a keyword into a confident misclassification."""
        error = classify(
            500, {"code": "Something went wrong, please try again", "error": "oops"}
        )

        assert error.code is None
        assert error.failure is GrokFailure.UNKNOWN

    def test_a_numeric_code_is_not_mistaken_for_a_name(self) -> None:
        error = classify(500, {"code": 13, "error": "internal"})

        assert error.code is None


class TestTheTwoStatusRulesThatLookBackwards:
    def test_402_is_always_a_spending_block_with_no_message_filter(self) -> None:
        assert classify(402, {"error": "anything at all"}).failure is (
            GrokFailure.SPENDING_BLOCKED
        )
        assert classify(402, {}).failure is GrokFailure.SPENDING_BLOCKED

    def test_403_is_a_spending_block_only_when_the_body_says_so(self) -> None:
        out_of_credits = classify(403, {"error": "You have run out of credits."})
        assert out_of_credits.failure is GrokFailure.SPENDING_BLOCKED

        plain = classify(
            403, {"code": "permission-denied", "error": "Access to the chat endpoint"}
        )
        assert plain.failure is GrokFailure.CLIENT_REJECTED

    def test_a_403_never_triggers_a_re_auth(self) -> None:
        """The sharp one. Refreshing on a 403 races xAI's failed-grant
        counter and can wipe a working credential, so a temporary refusal
        becomes a connection the user has to rebuild."""
        error = classify(403, {"code": "permission-denied", "error": "denied"})

        assert error.should_refresh_credentials is False

    def test_a_401_does_trigger_a_re_auth(self) -> None:
        assert classify(401, {"error": "expired"}).should_refresh_credentials is True


class TestWhatTheUserIsToldToDo:
    def test_a_free_account_out_of_quota_is_a_supported_state(self) -> None:
        """Not an onboarding failure: on xAI a subscription raises a limit
        rather than unlocking access, so this account works and has simply
        used its allowance."""
        error = classify(429, {"code": "subscription:free-usage-exhausted"})

        assert error.failure is GrokFailure.FREE_QUOTA_EXHAUSTED
        assert error.should_refresh_credentials is False

    def test_a_spending_limit_is_told_apart_from_a_rate_limit(self) -> None:
        """One is waited out; the other needs someone to change a setting."""
        spend = classify(429, {"code": "personal-team-blocked:spending-limit"})
        rate = classify(429, {"error": "slow down"})

        assert spend.failure is GrokFailure.SPENDING_BLOCKED
        assert spend.is_retryable is False
        assert rate.failure is GrokFailure.RATE_LIMITED
        assert rate.is_retryable is True

    def test_a_client_version_refusal_is_not_something_to_retry(self) -> None:
        """426 needs a different build, not another attempt."""
        error = classify(426, {"error": "version (none)"})

        assert error.failure is GrokFailure.CLIENT_REJECTED
        assert error.is_retryable is False

    def test_every_failure_carries_something_sayable(self) -> None:
        for status in (401, 402, 403, 426, 429, 500, None):
            assert classify(status, {}).message.strip()


class TestRetryAfter:
    def test_is_clamped_so_a_chat_does_not_hang_on_it(self) -> None:
        """A provider is free to ask for an hour. A chat waiting that long is
        a hang, and telling the user to come back beats pretending."""
        error = classify(429, {}, retry_after=3600)

        assert error.retry_after_seconds == MAX_RETRY_AFTER_SECONDS

    def test_a_short_wait_is_passed_through(self) -> None:
        assert classify(429, {}, retry_after="12").retry_after_seconds == 12

    def test_an_http_date_is_not_guessed_at(self) -> None:
        """Legal in the header, and resolving it needs a clock the caller
        owns. Saying nothing beats inventing a number."""
        error = classify(429, {}, retry_after="Wed, 21 Oct 2026 07:28:00 GMT")

        assert error.retry_after_seconds is None

    def test_no_header_means_no_answer(self) -> None:
        """There are no `x-ratelimit-*` headers on this API to fall back to,
        so absent really is unknown."""
        assert classify(429, {}).retry_after_seconds is None


class TestBadInput:
    def test_a_non_dict_body_does_not_raise(self) -> None:
        """An HTML error page from a proxy in front of the API is a body
        too, and it must not take the turn down with a TypeError."""
        assert classify(502, "<html>bad gateway</html>").failure is (
            GrokFailure.UNKNOWN
        )
        assert classify(502, None).failure is GrokFailure.UNKNOWN
