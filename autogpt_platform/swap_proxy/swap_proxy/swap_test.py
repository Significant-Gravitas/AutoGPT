"""The swap rules, ported from spark-vm's conformance tests
(``proxy/test_swap_addon.py``): each guards a finding from its review."""

import base64
import json
import urllib.parse

import pytest
from mitmproxy import http

from swap_proxy.swap import (
    Credential,
    RequestSwap,
    host_in_list,
    normalize_path,
    path_allowed,
    placeholder_names,
    scrub_text,
    totp_code,
)

HOST = "api.github.com"
TOKEN = 'ghp_"quoted"\\and&reserved=chars%'
GITHUB = Credential("github", {"access_token": TOKEN}, (HOST, ".githubusercontent.com"))
ACME = Credential(
    "acme",
    {
        "username": "someone@example.com",
        "password": "correct horse battery",
        "totp": "JBSWY3DPEHPK3PXP",
        "pin": "1234",
    },
    ("login.acme.test",),
    no_scrub=frozenset({"username"}),
)


def make_request(path="/", headers=None, content=b"", method="GET", host=HOST):
    return http.Request.make(
        method, f"https://{host}{path}", content=content, headers=headers or {}
    )


def swap(request, *credentials, host=HOST):
    s = RequestSwap(
        {c.name: c for c in credentials}, host, request.method, request.path
    )
    s.request(request)
    return s


class TestHeaders:
    def test_bearer(self):
        r = make_request(headers={"Authorization": "Bearer hsurr:github"})
        s = swap(r, GITHUB)
        assert r.headers["Authorization"] == f"Bearer {TOKEN}"
        assert [(e.kind, e.placeholder) for e in s.events] == [
            ("swapped", "hsurr:github")
        ]

    def test_basic_auth_is_decoded_swapped_and_reencoded(self):
        """git and ``curl -u`` hide the placeholder inside base64."""
        pair = base64.b64encode(b"x-access-token:hsurr:github").decode()
        r = make_request(headers={"Authorization": f"Basic {pair}"})
        swap(r, GITHUB)
        decoded = base64.b64decode(r.headers["Authorization"].split()[1]).decode()
        assert decoded == f"x-access-token:{TOKEN}"

    @pytest.mark.parametrize("header", ["Referer", "Origin"])
    def test_referer_and_origin_are_never_swapped(self, header):
        r = make_request(headers={header: "https://x.test/?k=hsurr:github"})
        swap(r, GITHUB)
        assert r.headers[header] == "https://x.test/?k=hsurr:github"

    def test_cookie_only_for_a_credential_that_says_so(self):
        r = make_request(headers={"Cookie": "session=hsurr:github"})
        swap(r, GITHUB)
        assert r.headers["Cookie"] == "session=hsurr:github"
        cookie = Credential("github", {"access_token": "abc"}, (HOST,), cookie=True)
        swap(r, cookie)
        assert r.headers["Cookie"] == "session=abc"

    def test_custom_header(self):
        r = make_request(headers={"X-Api-Key": "hsurr:github"})
        swap(r, GITHUB)
        assert r.headers["X-Api-Key"] == TOKEN


class TestBinding:
    def test_an_unbound_host_is_refused_and_says_so(self):
        r = make_request(headers={"Authorization": "Bearer hsurr:github"})
        s = swap(r, GITHUB, host="evil.test")
        assert r.headers["Authorization"] == "Bearer hsurr:github"
        assert [(e.kind, e.reason) for e in s.events] == [("refused", "unbound-host")]

    def test_a_credential_with_no_hosts_never_swaps(self):
        r = make_request(headers={"Authorization": "Bearer hsurr:github"})
        swap(r, Credential("github", {"access_token": "abc"}, ()))
        assert r.headers["Authorization"] == "Bearer hsurr:github"

    def test_leading_dot_binds_subdomains_only(self):
        for host, swapped in [
            ("raw.githubusercontent.com", True),
            ("githubusercontent.com", False),
            ("evilgithubusercontent.com", False),
        ]:
            r = make_request(headers={"X-K": "hsurr:github"}, host=host)
            swap(r, GITHUB, host=host)
            assert (r.headers["X-K"] == TOKEN) is swapped, host

    def test_an_unknown_name_is_left_alone(self):
        r = make_request(headers={"X-K": "hsurr:gitlab"})
        assert swap(r, GITHUB).events == [] and r.headers["X-K"] == "hsurr:gitlab"

    def test_an_unknown_entry_keeps_the_whole_placeholder(self):
        """``hsurr:github:8080`` must keep its ``:8080``."""
        r = make_request(headers={"X-K": "hsurr:github:8080"})
        s = swap(r, GITHUB)
        assert r.headers["X-K"] == "hsurr:github:8080"
        assert s.events[0].reason == "unknown-entry"


class TestBodies:
    @pytest.mark.parametrize(
        "content_type",
        [
            "application/vnd.api+json",
            "application/merge-patch+json; charset=utf-8",
            "Application/JSON",
            " application/json ;charset=UTF-8",
        ],
    )
    def test_every_json_media_type_is_escaped_as_json(self, content_type):
        """TOKEN holds ``"`` and ``\\``: substituted plainly they would break
        the document."""
        body = json.dumps({"data": {"password": "hsurr:github"}}).encode()
        r = make_request(
            content=body, method="POST", headers={"Content-Type": content_type}
        )
        swap(r, GITHUB)
        assert json.loads(r.content or b"") == {"data": {"password": TOKEN}}

    @pytest.mark.parametrize(
        "content_type", ["application/x-www-form-urlencoded-not", "text/plain"]
    )
    def test_only_the_form_media_type_is_parsed_as_a_form(self, content_type):
        r = make_request(
            content=b"token=hsurr:github",
            method="POST",
            headers={"Content-Type": content_type},
        )
        swap(r, GITHUB)
        assert r.content == f"token={TOKEN}".encode()  # plain substitution

    def test_json_values_are_escaped(self):
        body = json.dumps({"token": "hsurr:github"}).encode()
        r = make_request(
            content=body, method="POST", headers={"Content-Type": "application/json"}
        )
        swap(r, GITHUB)
        assert json.loads(r.content or b"") == {"token": TOKEN}

    @pytest.mark.parametrize("sent", ["hsurr:github", "hsurr%3Agithub"])
    def test_form_fields_are_reencoded(self, sent):
        """curl sends the colon, a browser percent-encodes it; either way a
        value with ``&`` or ``=`` cannot corrupt the fields."""
        r = make_request(
            content=f"user=me&token={sent}&next=1".encode(),
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        swap(r, GITHUB)
        assert urllib.parse.parse_qsl((r.content or b"").decode()) == [
            ("user", "me"),
            ("token", TOKEN),
            ("next", "1"),
        ]

    @pytest.mark.parametrize("name", ["hsurr:github", "hsurr%3Agithub"])
    def test_form_field_names_are_never_swapped_in_either_spelling(self, name):
        body = f"{name}=1".encode()
        r = make_request(
            content=body,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        s = swap(r, GITHUB)
        assert r.content == body and s.events == []

    def test_plain_text(self):
        r = make_request(content=b"key=hsurr:github", method="POST")
        swap(r, GITHUB)
        assert r.content == f"key={TOKEN}".encode()

    def test_binary_bodies_are_left_alone(self):
        body = b"\xff\xfe hsurr:github"
        r = make_request(content=body, method="POST")
        swap(r, GITHUB)
        assert r.content == body


class TestUrl:
    def test_query_values(self):
        r = make_request(path="/x?access_token=hsurr:github&keep=a%20b")
        swap(r, GITHUB)
        assert dict(r.query) == {"access_token": TOKEN, "keep": "a b"}

    def test_path_segment_cannot_be_escaped_by_the_value(self):
        r = make_request(path="/hooks/hsurr:github/fire")
        swap(r, Credential("github", {"access_token": "a/b?c"}, (HOST,)))
        assert r.path == "/hooks/a%2Fb%3Fc/fire"

    def test_unchanged_segments_stay_byte_identical(self):
        r = make_request(path="/a%25b/%2F~:@/hsurr:github")
        swap(r, Credential("github", {"access_token": "tok"}, (HOST,)))
        assert r.path == "/a%25b/%2F~:@/tok"

    def test_a_query_no_value_of_which_changed_stays_byte_identical(self):
        """A path swap must not re-encode a query that carries its own
        escaping: ``%20`` would become ``+`` and a signed URL would break."""
        query = "?sig=a%20b%2Fc&exp=1~2&flag&empty="
        r = make_request(path="/hooks/hsurr:github/fire" + query)
        swap(r, Credential("github", {"access_token": "tok"}, (HOST,)))
        assert r.path == "/hooks/tok/fire" + query

    def test_a_query_with_a_swapped_value_is_rebuilt(self):
        r = make_request(path="/hsurr:github?t=hsurr:github&keep=a%20b")
        swap(r, Credential("github", {"access_token": "t/k"}, (HOST,)))
        assert r.path == "/t%2Fk?t=t%2Fk&keep=a+b"


class TestEntriesAndTotp:
    def test_named_entries(self):
        r = make_request(
            host="login.acme.test",
            headers={"X-U": "hsurr:acme:username", "X-P": "hsurr:acme:password"},
        )
        swap(r, ACME, host="login.acme.test")
        assert (r.headers["X-U"], r.headers["X-P"]) == (
            "someone@example.com",
            "correct horse battery",
        )

    def test_totp_yields_the_code_never_the_seed(self):
        r = make_request(host="login.acme.test", headers={"X-C": "hsurr:acme:totp"})
        swap(r, ACME, host="login.acme.test")
        assert r.headers["X-C"] == totp_code("JBSWY3DPEHPK3PXP")
        assert "JBSWY3DPEHPK3PXP" not in r.headers["X-C"]

    def test_totp_matches_rfc_6238(self):
        seed = base64.b32encode(b"12345678901234567890").decode()
        assert totp_code(seed, at=59) == "287082"
        assert totp_code(seed, at=1111111109) == "081804"

    def test_a_bad_seed_is_refused_not_raised(self):
        bad = Credential("acme", {"totp": "not base32!"}, (HOST,))
        r = make_request(headers={"X-C": "hsurr:acme:totp"})
        assert swap(r, bad).events[0].reason == "bad-totp-seed"


class TestMethodAndPathLimits:
    def limited(self, **limits):
        return Credential("github", {"access_token": "tok"}, (HOST,), **limits)

    def attempt(self, credential, method="GET", path="/"):
        r = make_request(path=path, method=method, headers={"X-K": "hsurr:github"})
        s = swap(r, credential)
        return r.headers["X-K"] == "tok", [e.reason for e in s.events if e.reason]

    def test_method_limit(self):
        c = self.limited(allowed_methods=("GET",))
        assert self.attempt(c, "get") == (True, [])
        assert self.attempt(c, "POST") == (False, ["method-not-allowed"])

    def test_path_prefix_is_segment_aligned(self):
        c = self.limited(allowed_paths=("/repos/",))
        assert self.attempt(c, path="/repos/x")[0]
        assert self.attempt(c, path="/repos")[0]
        assert self.attempt(c, path="/repository") == (False, ["path-not-allowed"])

    @pytest.mark.parametrize(
        "path",
        [
            "/repos/../admin",
            "/repos/%2e%2e/admin",
            "/repos/%252e%252e/admin",
            "/repos/x;param",
            "/repos/x%5C..%5Cadmin",
        ],
    )
    def test_smuggling_shapes_are_refused(self, path):
        c = self.limited(allowed_paths=("/repos/",))
        assert self.attempt(c, path=path) == (False, ["path-not-allowed"])

    def test_empty_lists_fail_closed(self):
        assert not self.attempt(self.limited(allowed_methods=()))[0]
        assert not self.attempt(self.limited(allowed_paths=()))[0]

    def test_without_a_path_a_path_bound_credential_never_swaps(self):
        """Websocket messages and tunnels: nothing to check the path against."""
        c = self.limited(allowed_paths=("/repos/",))
        s = RequestSwap({"github": c}, HOST)
        assert s.text("hsurr:github") == "hsurr:github"

    def test_normalize_path(self):
        assert normalize_path("/a/./b/../c?q=1") == "/a/c"
        assert normalize_path("%2Fa%2F%2e%2e%2Fb") == "/b"
        assert path_allowed("/repos/x", ["/repos/"])
        assert not path_allowed("/repository", ["/repos/"])


class TestPlaceholderNames:
    def test_finds_them_everywhere_the_swap_looks(self):
        pair = base64.b64encode(b"u:hsurr:inbasic").decode()
        r = make_request(
            path="/p/hsurr%3Ainpath?q=hsurr:inquery",
            method="POST",
            headers={"Authorization": f"Basic {pair}", "X-K": "hsurr:inheader"},
            content=b"a=hsurr%3Ainform&b=hsurr:inbody:entry",
        )
        assert placeholder_names(r) == {
            "inbasic",
            "inpath",
            "inquery",
            "inheader",
            "inform",
            "inbody",
        }

    def test_none(self):
        assert placeholder_names(make_request(content=b"\xff\x00")) == set()


class TestScrub:
    def test_values_go_back_to_placeholders(self):
        page = f'{{"token": "{TOKEN}", "pw": "correct horse battery"}}'
        assert scrub_text(page, [GITHUB, ACME]) == (
            '{"token": "hsurr:github", "pw": "hsurr:acme:password"}'
        )

    def test_short_values_are_never_scrubbed(self):
        """A short value would mangle the page: "Next" with pin "e"."""
        assert scrub_text("pin 1234 and 12345", [ACME]) == "pin 1234 and 12345"

    def test_opted_out_entries_are_left(self):
        assert "someone@example.com" in scrub_text("hi someone@example.com", [ACME])

    def test_totp_codes_only_as_whole_tokens(self):
        code = totp_code("JBSWY3DPEHPK3PXP")
        text = scrub_text(f"code {code}, order 9{code}9", [ACME])
        assert text == f"code hsurr:acme:totp, order 9{code}9"

    def test_the_longer_of_two_overlapping_values_wins(self):
        a = Credential("a", {"access_token": "secretvalue"}, (HOST,))
        b = Credential("b", {"access_token": "secretvalue-extended"}, (HOST,))
        assert scrub_text("x secretvalue-extended", [a, b]) == "x hsurr:b"


def test_a_credential_never_prints_its_values():
    assert TOKEN not in repr(GITHUB) and TOKEN not in str(GITHUB)
    assert "correct horse" not in f"{ACME!r}"


# Mirrored in backend/backend/copilot/swap_credentials_test.py; keep the two
# identical.  The backend checks the binding with its own ``_host_is_bound``
# (the packages cannot import each other), and that suite also loads this
# package's ``host_in_list`` by path to assert the two agree.
HOST_BINDING_TABLE = [
    # (host, bound entries, may the credential go there?)
    ("api.github.com", ["api.github.com"], True),
    ("API.GitHub.com", ["api.github.com"], True),
    ("api.github.com", ["API.GITHUB.COM"], True),
    ("api.github.com:443", ["api.github.com"], True),
    ("github.com", ["api.github.com"], False),
    ("api.github.com.evil.test", ["api.github.com"], False),
    ("evilapi.github.com", ["api.github.com"], False),
    ("notgithub.com", ["github.com"], False),
    ("sub.github.com", ["github.com"], False),  # exact names do not cover subdomains
    ("raw.githubusercontent.com", [".githubusercontent.com"], True),
    ("a.b.githubusercontent.com", [".githubusercontent.com"], True),
    ("githubusercontent.com", [".githubusercontent.com"], False),
    ("evilgithubusercontent.com", [".githubusercontent.com"], False),
    ("githubusercontent.com.evil.test", [".githubusercontent.com"], False),
    ("api.github.com", [], False),
    ("", ["api.github.com"], False),
    ("api.github.com", ["github.com", "api.github.com"], True),
]


@pytest.mark.parametrize("host, entries, bound", HOST_BINDING_TABLE)
def test_host_binding_is_the_table_the_backend_is_held_to(host, entries, bound):
    assert host_in_list(host, entries) is bound
