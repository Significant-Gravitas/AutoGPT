"""The addon's hooks against mitmproxy's own test flows: what the e2e suites
cannot reach over a hand-written HTTP/1 client (websockets, HTTP/2 framing)
and the body buffer on its own."""

import gzip
import json
import logging

import pytest
from mitmproxy import http
from mitmproxy.test import tflow
from wsproto.frame_protocol import Opcode

from swap_proxy.addon import (
    MAX_BODY_BYTES,
    MAX_DECODED_BYTES,
    BufferedBody,
    SwapProxyAddon,
    known_size,
)
from swap_proxy.egress import EgressGuard
from swap_proxy.owners import Owner, OwnerDirectory
from swap_proxy.swap import Credential

TOKEN = "ghp_userAsecretvalue0001"
HOST = "api.github.com"


class Source:
    async def bound_names(self, host):
        return {"github"} if host == HOST else set()

    async def resolve(self, user_id, name, host):
        return Credential("github", {"access_token": TOKEN}, (HOST,))


class NoRedis:
    async def get(self, name):
        return None


def addon_for(flow: http.HTTPFlow, swaps: bool = True) -> SwapProxyAddon:
    addon = SwapProxyAddon(OwnerDirectory(NoRedis()), Source(), EgressGuard())
    addon._owners[flow.client_conn] = Owner("session:s-a", "user-a", "sb-1", swaps)
    flow.request.host, flow.request.scheme = HOST, "https"
    flow.request.headers["host"] = HOST
    flow.server_conn.sni = HOST
    return addon


def audit(caplog) -> list[tuple]:
    return [
        (line["event"], line.get("placeholder") or line.get("reason"))
        for line in (json.loads(r.message) for r in caplog.records)
    ]


# ------------------------------------------------------------ websockets


def websocket_flow(text: str, from_client: bool) -> http.HTTPFlow:
    flow = tflow.twebsocketflow()
    assert flow.websocket is not None
    message = flow.websocket.messages[-1]
    message.from_client, message.content = from_client, text.encode()
    return flow


def last_message(flow: http.HTTPFlow) -> str:
    assert flow.websocket is not None
    return flow.websocket.messages[-1].content.decode()


async def test_a_placeholder_in_a_frame_from_the_box_is_swapped(caplog):
    flow = websocket_flow('{"auth": "hsurr:github"}', from_client=True)
    addon = addon_for(flow)
    with caplog.at_level(logging.INFO, logger="swap_proxy.audit"):
        await addon.websocket_message(flow)
    assert last_message(flow) == '{"auth": "%s"}' % TOKEN
    assert audit(caplog) == [("swapped", "hsurr:github")]


async def test_a_token_echoed_in_a_frame_from_the_server_is_scrubbed(caplog):
    flow = websocket_flow('{"ack": "%s"}' % TOKEN, from_client=False)
    addon = addon_for(flow)
    with caplog.at_level(logging.INFO, logger="swap_proxy.audit"):
        await addon.websocket_message(flow)
    assert last_message(flow) == '{"ack": "hsurr:github"}'
    assert audit(caplog) == [("scrubbed", None)]
    assert TOKEN not in caplog.text


async def test_a_placeholder_in_a_frame_from_the_server_is_not_swapped():
    """The swap is for what the box sends out, never for what comes back."""
    flow = websocket_flow("hsurr:github", from_client=False)
    await addon_for(flow).websocket_message(flow)
    assert last_message(flow) == "hsurr:github"


@pytest.mark.parametrize("from_client", [True, False])
async def test_frames_of_a_box_that_does_not_swap_are_left_alone(from_client):
    text = "hsurr:github" if from_client else TOKEN
    flow = websocket_flow(text, from_client)
    await addon_for(flow, swaps=False).websocket_message(flow)
    assert last_message(flow) == text


async def test_a_binary_frame_is_left_alone():
    flow = websocket_flow(TOKEN, from_client=False)
    assert flow.websocket is not None
    flow.websocket.messages[-1].type = Opcode.BINARY
    await addon_for(flow).websocket_message(flow)
    assert last_message(flow) == TOKEN


# ------------------------------------------------------------ the body buffer


def buffered(limit=10, release=True):
    reasons: list[str] = []

    def overflow(reason):
        reasons.append(reason)
        return release

    return BufferedBody(limit, lambda body: body.upper(), overflow), reasons


def test_a_body_is_held_until_its_end_and_released_transformed():
    body, reasons = buffered()
    # Lists, never b"": an empty chunk on the wire ends a chunked body.
    assert [body(b"abc"), body(b"def"), body(b"")] == [[], [], [b"ABCDEF"]]
    assert reasons == []


def test_an_empty_body_sends_nothing():
    body, _ = buffered()
    assert body(b"") == []


def test_past_the_limit_a_request_body_is_released_as_it_was():
    body, reasons = buffered(limit=4, release=True)
    assert body(b"abc") == []
    assert body(b"def") == [b"abc", b"def"]
    assert body(b"ghi") == [b"ghi"] and body(b"") == []
    assert reasons == ["body-too-large"]


def test_past_the_limit_a_response_body_is_withheld_for_good():
    body, reasons = buffered(limit=4, release=False)
    assert [body(b"abc"), body(b"def"), body(b"ghi"), body(b"")] == [[], [], [], []]
    assert reasons == ["body-too-large"]


def test_data_after_the_end_marker_is_not_transformed_in_two_halves():
    """An empty DATA frame mid-body looks like the end.  A value split around
    it would survive two separate scrubs, so the rest is refused."""
    body, reasons = buffered(limit=100, release=False)
    assert body(b"ghp_userAsecr") == [] and body(b"") == [b"GHP_USERASECR"]
    assert body(b"etvalue0001") == []
    assert reasons == ["data-after-end"]


# ------------------------------------------------------------ what is known early


def test_the_size_of_a_body_is_known_only_when_its_framing_says_so():
    flow = tflow.tflow(resp=True)
    assert flow.response is not None
    flow.request.headers["content-length"] = "12"
    assert known_size(flow.request, None) == 12
    flow.response.headers["content-length"] = "7"
    assert known_size(flow.request, flow.response) == 7
    del flow.response.headers["content-length"]
    flow.response.headers["transfer-encoding"] = "chunked"
    assert known_size(flow.request, flow.response) is None
    del flow.response.headers["transfer-encoding"]
    assert known_size(flow.request, flow.response) is None  # read until close
    flow.request.headers["content-length"] = "nonsense"
    assert known_size(flow.request, None) is None


def test_http2_without_a_length_is_unknown_not_empty():
    flow = tflow.tflow()
    flow.request.http_version = "HTTP/2.0"
    flow.request.headers.pop("content-length", None)
    assert known_size(flow.request, None) is None


# ------------------------------------------------------------ the header hooks


async def test_an_http2_upload_without_a_length_is_swapped_head_first_then_body(
    caplog,
):
    flow = tflow.tflow()
    addon = addon_for(flow)
    flow.request.http_version = "HTTP/2.0"
    flow.request.headers.pop("content-length", None)
    flow.request.headers["authorization"] = "Bearer hsurr:github"
    flow.request.headers["content-type"] = "application/json"
    flow.request.raw_content = None
    with caplog.at_level(logging.INFO, logger="swap_proxy.audit"):
        await addon.requestheaders(flow)
        # The head is final before the body exists.
        assert flow.request.headers["authorization"] == f"Bearer {TOKEN}"
        stream = flow.request.stream
        assert isinstance(stream, BufferedBody)
        sent = [*stream(b'{"t": "hsurr:gi'), *stream(b'thub"}'), *stream(b"")]
        await addon.request(flow)  # fires after the bytes left: must add nothing
    assert b"".join(sent) == b'{"t": "%s"}' % TOKEN.encode()
    assert audit(caplog) == [("swapped", "hsurr:github")] * 2


async def test_a_gzipped_response_without_a_length_is_scrubbed_inside_its_encoding(
    caplog,
):
    flow = tflow.tflow(resp=True)
    assert flow.response is not None
    addon = addon_for(flow)
    flow.response.headers["content-type"] = "application/json"
    flow.response.headers["content-encoding"] = "gzip"
    flow.response.text = '{"echo": "%s"}' % TOKEN
    wire = flow.response.raw_content or b""
    del flow.response.headers["content-length"]
    flow.response.headers["transfer-encoding"] = "chunked"
    with caplog.at_level(logging.INFO, logger="swap_proxy.audit"):
        await addon.responseheaders(flow)
        stream = flow.response.stream
        assert isinstance(stream, BufferedBody)
        sent = [*stream(wire[:9]), *stream(wire[9:]), *stream(b"")]
    seen = flow.response.copy()
    seen.raw_content = b"".join(sent)
    assert seen.text == '{"echo": "hsurr:github"}'
    assert audit(caplog) == [("scrubbed", None)]


async def test_a_response_that_outgrows_the_buffer_kills_the_flow(caplog):
    flow = tflow.tflow(resp=True)
    assert flow.response is not None
    addon = addon_for(flow)
    flow.live = True
    flow.response.headers["content-type"] = "text/plain"
    del flow.response.headers["content-length"]
    flow.response.headers["transfer-encoding"] = "chunked"
    with caplog.at_level(logging.INFO, logger="swap_proxy.audit"):
        await addon.responseheaders(flow)
        stream = flow.response.stream
        assert isinstance(stream, BufferedBody)
        assert stream(b"x" * MAX_BODY_BYTES) == []
        assert stream(TOKEN.encode()) == [] and stream(b"") == []
    assert flow.error is not None and not flow.killable
    assert audit(caplog) == [("refused-response", "too-large-to-scrub")]


@pytest.mark.parametrize("content_type", ["image/png", "application/zip"])
async def test_a_binary_response_is_left_to_stream(content_type):
    flow = tflow.tflow(resp=True)
    assert flow.response is not None
    addon = addon_for(flow)
    flow.response.headers["content-type"] = content_type
    flow.response.headers["content-length"] = str(MAX_BODY_BYTES * 4)
    await addon.responseheaders(flow)
    assert flow.response.stream is False and flow.error is None


async def test_a_box_that_does_not_swap_has_no_response_refused():
    flow = tflow.tflow(resp=True)
    assert flow.response is not None
    addon = addon_for(flow, swaps=False)
    flow.response.headers["content-type"] = "text/plain"
    flow.response.headers["content-length"] = str(MAX_BODY_BYTES * 4)
    await addon.responseheaders(flow)
    assert flow.response.stream is False and flow.error is None


# ------------------------------------------------------------ what a body decodes to
#
# MAX_BODY_BYTES counts bytes on the wire.  A few kilobytes of gzip can stand
# for far more, decoded on the event loop every box shares.

BOMB = gzip.compress(
    b'{"echo": "%s", "pad": "' % TOKEN.encode() + bytes(2 * MAX_DECODED_BYTES)
)


@pytest.fixture
def no_unbounded_decode(monkeypatch):
    """mitmproxy's own decode has no bound: a bomb must never reach it."""

    def decode(*args, **kwargs):
        raise AssertionError("decoded without a bound")

    monkeypatch.setattr("mitmproxy.http.encoding.decode", decode)


def test_the_bomb_is_small_on_the_wire():
    assert len(BOMB) < MAX_BODY_BYTES // 50


def bombed_response(chunked: bool) -> http.HTTPFlow:
    flow = tflow.tflow(resp=True)
    assert flow.response is not None
    flow.live = True
    flow.response.headers["content-type"] = "application/json"
    flow.response.headers["content-encoding"] = "gzip"
    if chunked:
        del flow.response.headers["content-length"]
        flow.response.headers["transfer-encoding"] = "chunked"
    else:
        flow.response.raw_content = BOMB
        flow.response.headers["content-length"] = str(len(BOMB))
    return flow


async def test_a_response_with_a_length_that_decodes_past_the_cap_is_refused(
    caplog, no_unbounded_decode
):
    flow = bombed_response(chunked=False)
    assert flow.response is not None
    addon = addon_for(flow)
    with caplog.at_level(logging.INFO, logger="swap_proxy.audit"):
        await addon.responseheaders(flow)
        assert flow.response.stream is False  # small on the wire: held whole
        await addon.response(flow)
    assert flow.error is not None and not flow.killable
    assert audit(caplog) == [("refused-response", "decoded-too-large")]


async def test_a_held_response_that_decodes_past_the_cap_is_refused(
    caplog, no_unbounded_decode
):
    flow = bombed_response(chunked=True)
    assert flow.response is not None
    addon = addon_for(flow)
    with caplog.at_level(logging.INFO, logger="swap_proxy.audit"):
        await addon.responseheaders(flow)
        stream = flow.response.stream
        assert isinstance(stream, BufferedBody)
        assert [*stream(BOMB[:100]), *stream(BOMB[100:]), *stream(b"")] == []
    assert flow.error is not None and not flow.killable
    assert audit(caplog) == [("refused-response", "decoded-too-large")]


@pytest.mark.parametrize("encoding", ["compress", "gzip, br"])
async def test_a_response_in_an_encoding_that_cannot_be_bounded_is_refused(
    caplog, encoding
):
    flow = bombed_response(chunked=False)
    assert flow.response is not None
    flow.response.headers["content-encoding"] = encoding
    addon = addon_for(flow)
    with caplog.at_level(logging.INFO, logger="swap_proxy.audit"):
        await addon.response(flow)
    assert flow.error is not None
    assert audit(caplog) == [("refused-response", "undecodable-encoding")]


async def test_a_box_with_nothing_to_scrub_gets_its_bomb_untouched(
    caplog, no_unbounded_decode
):
    """No value of this owner's can be in it, so it is not the proxy's to read."""
    flow = bombed_response(chunked=False)
    assert flow.response is not None
    addon = addon_for(flow, swaps=False)
    with caplog.at_level(logging.INFO, logger="swap_proxy.audit"):
        await addon.response(flow)
    assert flow.error is None and flow.response.raw_content == BOMB
    assert audit(caplog) == []


def bombed_request(chunked: bool) -> http.HTTPFlow:
    flow = tflow.tflow()
    flow.request.method = "POST"
    flow.request.headers["authorization"] = "Bearer hsurr:github"
    flow.request.headers["content-type"] = "application/json"
    flow.request.headers["content-encoding"] = "gzip"
    if chunked:
        flow.request.headers.pop("content-length", None)
        flow.request.headers["transfer-encoding"] = "chunked"
        flow.request.raw_content = None
    else:
        flow.request.raw_content = BOMB
        flow.request.headers["content-length"] = str(len(BOMB))
    return flow


async def test_a_request_body_that_decodes_past_the_cap_goes_out_as_it_is(
    caplog, no_unbounded_decode
):
    flow = bombed_request(chunked=False)
    addon = addon_for(flow)
    with caplog.at_level(logging.INFO, logger="swap_proxy.audit"):
        await addon.requestheaders(flow)
        await addon.request(flow)
    assert flow.error is None and flow.request.raw_content == BOMB
    # The head is still swapped; no swap is claimed for the body.
    assert flow.request.headers["authorization"] == f"Bearer {TOKEN}"
    assert audit(caplog) == [
        ("body-not-swapped", "decoded-too-large"),
        ("swapped", "hsurr:github"),
    ]


async def test_a_held_request_body_that_decodes_past_the_cap_goes_out_as_it_is(
    caplog, no_unbounded_decode
):
    flow = bombed_request(chunked=True)
    addon = addon_for(flow)
    with caplog.at_level(logging.INFO, logger="swap_proxy.audit"):
        await addon.requestheaders(flow)
        stream = flow.request.stream
        assert isinstance(stream, BufferedBody)
        sent = [*stream(BOMB[:100]), *stream(BOMB[100:]), *stream(b"")]
    assert b"".join(sent) == BOMB
    assert audit(caplog) == [
        ("swapped", "hsurr:github"),
        ("body-not-swapped", "decoded-too-large"),
    ]


async def test_a_gzipped_request_body_within_the_cap_is_swapped_inside_its_encoding(
    caplog, no_unbounded_decode
):
    flow = tflow.tflow()
    addon = addon_for(flow)
    flow.request.method = "POST"
    flow.request.headers["content-type"] = "application/json"
    flow.request.headers["content-encoding"] = "gzip"
    flow.request.raw_content = gzip.compress(b'{"token": "hsurr:github"}')
    flow.request.headers["content-length"] = str(len(flow.request.raw_content))
    with caplog.at_level(logging.INFO, logger="swap_proxy.audit"):
        await addon.requestheaders(flow)
        await addon.request(flow)
    sent = gzip.decompress(flow.request.raw_content or b"")
    assert json.loads(sent) == {"token": TOKEN}
    assert flow.request.headers["content-length"] == str(len(flow.request.raw_content))
    assert audit(caplog) == [("swapped", "hsurr:github")]


async def test_a_token_in_the_second_gzip_member_of_a_response_is_scrubbed(caplog):
    """A client decodes every member; the scrub must see every member too."""
    flow = tflow.tflow(resp=True)
    assert flow.response is not None
    addon = addon_for(flow)
    flow.response.headers["content-type"] = "application/json"
    flow.response.headers["content-encoding"] = "gzip"
    flow.response.raw_content = gzip.compress(b'{"a": "') + gzip.compress(
        b'%s"}' % TOKEN.encode()
    )
    with caplog.at_level(logging.INFO, logger="swap_proxy.audit"):
        await addon.response(flow)
    assert gzip.decompress(flow.response.raw_content or b"") == b'{"a": "hsurr:github"}'
    assert audit(caplog) == [("scrubbed", None)]


async def test_a_response_with_data_after_its_gzip_stream_is_refused(caplog):
    flow = tflow.tflow(resp=True)
    assert flow.response is not None
    addon = addon_for(flow)
    flow.live = True
    flow.response.headers["content-type"] = "application/json"
    flow.response.headers["content-encoding"] = "gzip"
    flow.response.raw_content = gzip.compress(b"{}") + TOKEN.encode()
    with caplog.at_level(logging.INFO, logger="swap_proxy.audit"):
        await addon.response(flow)
    assert flow.error is not None
    assert audit(caplog) == [("refused-response", "undecodable-encoding")]
