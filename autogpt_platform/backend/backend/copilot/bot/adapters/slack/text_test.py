"""Tests for Slack mrkdwn conversion + mention substitution."""

from .text import to_mrkdwn


class TestBold:
    def test_double_asterisk_becomes_single(self):
        assert to_mrkdwn("hello **world**") == "hello *world*"

    def test_multiple_bolds(self):
        assert to_mrkdwn("**a** then **b**") == "*a* then *b*"

    def test_no_bold_left_alone(self):
        assert to_mrkdwn("plain text") == "plain text"


class TestLinks:
    def test_markdown_link_becomes_mrkdwn(self):
        assert (
            to_mrkdwn("see [docs](https://example.com)")
            == "see <https://example.com|docs>"
        )

    def test_multiple_links(self):
        rendered = to_mrkdwn("[a](http://a.com) and [b](http://b.com)")
        assert rendered == "<http://a.com|a> and <http://b.com|b>"


class TestMentions:
    def test_resolves_known_user(self):
        rendered = to_mrkdwn(
            "@bently any update?",
            mentionable_users=(("bently", "U123"),),
        )
        assert rendered == "<@U123> any update?"

    def test_unknown_user_stays_plain(self):
        rendered = to_mrkdwn(
            "@stranger check this",
            mentionable_users=(("bently", "U123"),),
        )
        assert rendered == "@stranger check this"

    def test_longest_name_first(self):
        # "@John Smith" matches before "@John" — order-independent allowlist.
        rendered = to_mrkdwn(
            "ping @John Smith now",
            mentionable_users=(("John", "U1"), ("John Smith", "U2")),
        )
        assert rendered == "ping <@U2> now"

    def test_word_boundary_avoids_emails(self):
        rendered = to_mrkdwn(
            "email me@example.com",
            mentionable_users=(("me", "U123"),),
        )
        assert rendered == "email me@example.com"

    def test_empty_allowlist_returns_plain(self):
        assert to_mrkdwn("@anyone hi") == "@anyone hi"
