"""What the runtime does with the two things the shell tells it about the
install it belongs to (src/identity.js runtimeEnvironment): the public port
to prefer, and the name under which it keeps what would otherwise be shared
with other installs (files outside the data directory, the browser's cookies).

The shell and the runtime are two programs in two languages; these tests
fail when one side changes a name or a number the other relies on.
"""

import logging
import re
import socket
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web

from autogpt_desktop import cookies, install, ports, proxy, rabbitmq

IDENTITY = Path(__file__).resolve().parents[2] / "src" / "identity.js"


def shell_constant(name: str) -> int:
    match = re.search(rf"^const {name} = (\d+);$", IDENTITY.read_text("utf-8"), re.M)
    assert (
        match
    ), f"src/identity.js no longer defines {name}; runtime/autogpt_desktop/ports.py relies on it"
    return int(match.group(1))


def free_port() -> int:
    return ports._free_port(exclude=set())


def test_the_shell_and_the_runtime_agree_on_the_normal_apps_port():
    assert shell_constant("NORMAL_PUBLIC_PORT") == ports.PREFERRED["public"], (
        "src/identity.js and ports.PREFERRED must name the same port: an installed "
        "app keeps the address it has, and the two must not disagree about a new one"
    )


def test_every_port_a_variant_can_prefer_is_one_the_runtime_accepts():
    first = shell_constant("VARIANT_PORT_FIRST")
    count = shell_constant("VARIANT_PORT_COUNT")
    assert (
        first in ports.PORT_RANGE and first + count - 1 in ports.PORT_RANGE
    ), "src/identity.js picks variant ports outside ports.PORT_RANGE; the runtime ignores those"
    assert ports.PREFERRED["public"] not in range(
        first, first + count
    ), "a variant could be given the normal app's port"
    assert range(first, first + count) == ports.VARIANT_PUBLIC_PORTS, (
        "src/identity.js and ports.VARIANT_PUBLIC_PORTS must name the same ports: "
        "the runtime keeps every other use out of them"
    )


def test_no_port_is_picked_by_chance_where_variants_have_their_addresses(monkeypatch):
    """A port is remembered for good: one install's database on another
    install's address would move that install for ever."""
    band = ports.VARIANT_PUBLIC_PORTS
    draws = iter([band.start, band.stop - 1, band.start + 5954])
    choose = ports.random.choice
    monkeypatch.setattr(
        ports.random, "choice", lambda options: next(draws, None) or choose(options)
    )
    assert ports._free_port(exclude=set()) not in band
    assert (
        next(draws, None) is None
    ), "the ports in the band were not all drawn and passed over"
    for _ in range(200):
        assert ports._free_port(exclude=set()) not in band
    outside = [port for port in ports.PORT_RANGE if port not in band]
    assert len(outside) >= 5000, "too few ports are left for an install's services"


def test_every_service_of_a_new_install_is_outside_the_variants_addresses(
    tmp_path, monkeypatch
):
    for asked in (None, str(ports.VARIANT_PUBLIC_PORTS.start + 5954)):
        if asked:
            monkeypatch.setenv(ports.PUBLIC_PORT_VARIABLE, asked)
        allocated = ports.allocate(tmp_path / f"ports-{asked}.json")
        others = {name: port for name, port in allocated.items() if name != "public"}
        assert not [
            name for name, port in others.items() if port in ports.VARIANT_PUBLIC_PORTS
        ]


def test_the_shell_passes_the_variables_the_runtime_reads():
    source = IDENTITY.read_text("utf-8")
    for variable in (ports.PUBLIC_PORT_VARIABLE, install.VARIABLE):
        assert (
            f"{variable}:" in source
        ), f"src/identity.js runtimeEnvironment no longer sets {variable}"
    # The names the shell passes: `AutoGPT`, and `AutoGPT-<slug>` for a variant.
    assert "dirName: `AutoGPT-${variant}`" in source
    for name in ("AutoGPT", "AutoGPT-voice", "AutoGPT-a-b-1", f"AutoGPT-{'x' * 24}"):
        assert install.NAME.fullmatch(name), name


def test_without_the_shell_the_runtime_prefers_the_normal_apps_port():
    assert ports._preferences() == ports.PREFERRED


def test_a_new_install_takes_the_port_the_shell_asked_for(tmp_path, monkeypatch):
    asked = free_port()
    monkeypatch.setenv(ports.PUBLIC_PORT_VARIABLE, str(asked))
    first = ports.allocate(tmp_path / "ports.json")
    assert first["public"] == asked
    assert ports.allocate(tmp_path / "ports.json") == first


def test_an_install_that_has_a_port_keeps_it_whatever_the_shell_asks_for(
    tmp_path, monkeypatch
):
    first = ports.allocate(tmp_path / "ports.json")
    monkeypatch.setenv(ports.PUBLIC_PORT_VARIABLE, str(free_port()))
    assert ports.allocate(tmp_path / "ports.json") == first


def test_a_taken_port_is_only_a_preference(tmp_path, monkeypatch):
    asked = free_port()
    monkeypatch.setenv(ports.PUBLIC_PORT_VARIABLE, str(asked))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
        holder.bind(("127.0.0.1", asked))
        holder.listen(1)
        allocated = ports.allocate(tmp_path / "ports.json")
    assert allocated["public"] != asked
    assert allocated["public"] in ports.PORT_RANGE


@pytest.mark.parametrize("asked", ["", "abc", "80", "65536", "-1", "18473.0", " 20001"])
def test_a_value_that_is_not_a_port_is_no_preference_at_all(
    tmp_path, monkeypatch, caplog, asked
):
    """Not the default either: that would put a variant on the normal app's port."""
    monkeypatch.setenv(ports.PUBLIC_PORT_VARIABLE, asked)
    with caplog.at_level(logging.WARNING, logger="autogpt_desktop"):
        allocated = ports.allocate(tmp_path / "ports.json")
    assert allocated["public"] != ports.PREFERRED["public"]
    assert allocated["public"] in ports.PORT_RANGE
    assert ports.PUBLIC_PORT_VARIABLE in caplog.text


def test_no_service_is_given_the_normal_apps_port_or_this_installs_own(monkeypatch):
    asked = free_port()
    monkeypatch.setenv(ports.PUBLIC_PORT_VARIABLE, str(asked))
    reserved = {ports.PREFERRED["public"], asked}
    draws = iter([ports.PREFERRED["public"], asked])
    choose = ports.random.choice
    monkeypatch.setattr(
        ports.random, "choice", lambda options: next(draws, None) or choose(options)
    )
    assert ports._free_port(exclude=set()) not in reserved


def test_the_install_name_is_one_folder_name_or_the_default(monkeypatch):
    assert install.name() == "AutoGPT"
    monkeypatch.setenv(install.VARIABLE, "AutoGPT-voice")
    assert install.name() == "AutoGPT-voice"
    for bad in (
        "",
        "..",
        "AutoGPT/x",
        "AutoGPT-Voice",
        "AutoGPT voice",
        "autogpt-voice",
        "AutoGPT-voice\n",
    ):
        monkeypatch.setenv(install.VARIABLE, bad)
        assert install.name() == "AutoGPT", bad


def test_each_install_keeps_its_links_in_a_folder_of_its_own(tmp_path, monkeypatch):
    # The cache directory of Linux, on every system the tests run on.
    monkeypatch.setattr(rabbitmq.sys, "platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    if any(character.isspace() for character in str(tmp_path)):
        pytest.skip("the temporary directory has a space in its path")
    assert rabbitmq._alias_root() == tmp_path / "AutoGPT" / "links"
    monkeypatch.setenv(install.VARIABLE, "AutoGPT-voice")
    assert rabbitmq._alias_root() == tmp_path / "AutoGPT-voice" / "links"
    assert (tmp_path / "AutoGPT-voice" / "links").is_dir()


# --- cookies -----------------------------------------------------------------
#
# Every install is on 127.0.0.1, and a browser keeps cookies per host.

SESSION = "better-auth.session_token"


def test_the_normal_app_keeps_its_cookie_names_and_a_variant_marks_its_own():
    assert cookies.mark_of("AutoGPT") == ""
    assert cookies.mark_of("AutoGPT-voice") == "AutoGPT-voice."
    set_cookie = f"{SESSION}=abc.def; Path=/; HttpOnly; SameSite=Lax"
    assert cookies.to_browser(set_cookie, "") == set_cookie
    assert (
        cookies.to_browser(set_cookie, "AutoGPT-voice.")
        == f"AutoGPT-voice.{set_cookie}"
    )
    # Browsers give these two prefixes a meaning; they stay in front.
    assert (
        cookies.to_browser("__Secure-x=1; Secure", "AutoGPT-voice.")
        == "__Secure-AutoGPT-voice.x=1; Secure"
    )
    assert (
        cookies.to_browser("__Host-x=1", "AutoGPT-voice.") == "__Host-AutoGPT-voice.x=1"
    )


def test_a_cookie_one_install_set_is_never_passed_on_by_another():
    """What a browser sends to any install: the cookies of all of them."""
    sent = "; ".join(
        [
            f"{SESSION}=normal",
            f"AutoGPT-voice.{SESSION}=voice",
            f"AutoGPT-voice-2.{SESSION}=voice2",
            f"AutoGPT-lab.{SESSION}=lab",
            "__Secure-AutoGPT-voice.s=secure",
            "sidebar_state=true",
        ]
    )
    assert cookies.to_platform(sent, "") == f"{SESSION}=normal; sidebar_state=true"
    assert (
        cookies.to_platform(sent, "AutoGPT-voice.")
        == f"{SESSION}=voice; __Secure-s=secure"
    )
    assert cookies.to_platform(sent, "AutoGPT-voice-2.") == f"{SESSION}=voice2"
    assert cookies.to_platform(sent, "AutoGPT-lab.") == f"{SESSION}=lab"
    assert cookies.to_platform(sent, "AutoGPT-other.") is None
    for mark in ("AutoGPT-voice.", "AutoGPT-voice-2.", "AutoGPT-lab."):
        assert "normal" not in (cookies.to_platform(sent, mark) or "")


def test_the_normal_apps_cookie_header_is_untouched_while_no_variant_has_cookies():
    for header in (f"{SESSION}=a;b=2", "a=1;  b=2 ;c", "a=b=c; d=", ""):
        assert cookies.to_platform(header, "") == header


def test_every_name_the_shell_gives_a_variant_is_recognised_as_a_variants_cookie():
    for name in ("AutoGPT-voice", "AutoGPT-a-b-1", f"AutoGPT-{'x' * 24}", "AutoGPT-0"):
        marked = f"{cookies.mark_of(name)}{SESSION}=secret"
        assert cookies.to_platform(marked, "") is None, name
        assert cookies.to_platform(marked, cookies.mark_of(name)) == f"{SESSION}=secret"


def platform_app() -> web.Application:
    async def sign_in(request: web.Request) -> web.Response:
        response = web.Response(text="ok")
        response.headers.add("Set-Cookie", f"{SESSION}=issued; Path=/; HttpOnly")
        response.headers.add("Set-Cookie", "other=1; Path=/")
        return response

    async def seen(request: web.Request) -> web.Response:
        return web.json_response({"cookie": request.headers.get("Cookie")})

    app = web.Application()
    app.router.add_get("/sign-in", sign_in)
    app.router.add_route("*", "/{tail:.*}", seen)
    return app


async def serve(app: web.Application) -> tuple[web.AppRunner, str]:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    return runner, f"http://127.0.0.1:{port}"


async def test_two_installs_in_one_browser_keep_their_sign_ins_apart(monkeypatch):
    """The proxies of the normal app and of a variant, in front of the same
    stand-in platform, and one cookie jar for both as a browser has for
    127.0.0.1."""
    platform, upstream = await serve(platform_app())
    runners = [platform]
    addresses = {}
    for name in ("AutoGPT", "AutoGPT-voice"):
        monkeypatch.setenv(install.VARIABLE, name)
        upstreams = proxy.Upstreams(
            public_url="http://127.0.0.1:1",
            rest=upstream,
            websocket=upstream,
            frontend=upstream,
        )
        runner, addresses[name] = await serve(proxy.build_app(upstreams))
        runners.append(runner)
    jar: dict[str, str] = {}

    async def browse(client: aiohttp.ClientSession, url: str) -> dict[str, str | None]:
        header = "; ".join(f"{name}={value}" for name, value in jar.items())
        async with client.get(
            url, headers={"Cookie": header} if header else {}
        ) as response:
            for set_cookie in response.headers.getall("Set-Cookie", []):
                name, _, value = set_cookie.split(";")[0].partition("=")
                jar[name] = value
            return await response.json() if url.endswith("/seen") else {}

    try:
        async with aiohttp.ClientSession() as client:
            await browse(client, f"{addresses['AutoGPT']}/sign-in")
            assert jar == {SESSION: "issued", "other": "1"}
            jar[SESSION] = "normal"
            await browse(client, f"{addresses['AutoGPT-voice']}/sign-in")
            assert (
                jar[SESSION] == "normal"
            ), "signing in to the variant replaced the normal app's session"
            assert jar[f"AutoGPT-voice.{SESSION}"] == "issued"
            jar[f"AutoGPT-voice.{SESSION}"] = "variant"
            normal = await browse(client, f"{addresses['AutoGPT']}/seen")
            variant = await browse(client, f"{addresses['AutoGPT-voice']}/seen")
            api = await browse(client, f"{addresses['AutoGPT-voice']}/_agpt/api/seen")
    finally:
        for runner in runners:
            await runner.cleanup()
    assert normal["cookie"] == f"{SESSION}=normal; other=1"
    assert variant["cookie"] == f"{SESSION}=variant; other=1"
    assert api["cookie"] == variant["cookie"]


async def test_a_variant_asked_with_only_another_installs_cookies_passes_none_on(
    monkeypatch,
):
    platform, upstream = await serve(platform_app())
    monkeypatch.setenv(install.VARIABLE, "AutoGPT-voice")
    upstreams = proxy.Upstreams(
        public_url="http://127.0.0.1:1",
        rest=upstream,
        websocket=upstream,
        frontend=upstream,
    )
    runner, address = await serve(proxy.build_app(upstreams))
    try:
        async with aiohttp.ClientSession() as client:
            headers = {"Cookie": f"{SESSION}=normal; AutoGPT-lab.{SESSION}=lab"}
            async with client.get(f"{address}/seen", headers=headers) as response:
                assert (await response.json())["cookie"] is None
    finally:
        await runner.cleanup()
        await platform.cleanup()
