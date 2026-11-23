"""Every server keeps an idle connection open longer than the proxy in front of it.

A reverse proxy keeps idle HTTP/1.1 connections to the server behind it and
reuses them. If the server closes one just as the proxy sends a request on it,
the request gets no answer and the proxy answers 502. It retries a GET on a new
connection, but not a POST, which may already have reached the server. A proxy
whose idle timeout is the shorter one always closes first, so the race can't
happen.

This checks each hop the repository configures: the proxies the self-hosting
guide names (and Google's load balancer) in front of the servers Docker Compose
publishes and of the single-container image's nginx, and that nginx in front of
the image's own servers. To fix a failure, raise the server's keep-alive where
the message says, or, if a proxy's default changed, update PROXY_IDLE_SECONDS
from its documentation.
"""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path
from typing import NamedTuple, Optional

import yaml

PLATFORM = Path(__file__).resolve().parents[3] / "autogpt_platform"
NGINX_CONF = PLATFORM / "single-container" / "nginx" / "nginx.conf"
RUN_FRONTEND = PLATFORM / "single-container" / "run-frontend.sh"
COMPOSE_FILES = (
    PLATFORM / "docker-compose.yml",
    PLATFORM / "docker-compose.platform.yml",
)
BACKEND = PLATFORM / "backend" / "backend"
SETTINGS = BACKEND / "util" / "settings.py"
# Where each backend server hands the setting to uvicorn.
REST_SERVER = BACKEND / "api" / "utils" / "uvicorn_options.py"
WEBSOCKET_SERVER = BACKEND / "api" / "ws_api.py"

# How long each proxy keeps an idle connection to the server behind it, by
# default. The self-hosting guide names nginx, Caddy and Traefik; Google's load
# balancer is in front of the hosted backend.
PROXY_IDLE_SECONDS = {
    # https://nginx.org/en/docs/http/ngx_http_upstream_module.html#keepalive_timeout
    "nginx": 60,
    # forwardingTimeouts.idleConnTimeout:
    # https://doc.traefik.io/traefik/reference/routing-configuration/http/load-balancing/serverstransport/
    "Traefik": 90,
    # keepalive, in reverse_proxy's http transport:
    # https://caddyserver.com/docs/caddyfile/directives/reverse_proxy#the-http-transport
    "Caddy": 120,
    # https://docs.cloud.google.com/load-balancing/docs/https/troubleshooting-ext-https-lbs
    "Google's load balancer": 600,
}
# A server starts its idle clock once it has handed a response's last bytes to
# the kernel, and the proxy once it has read them, which can be later.
HEADROOM_SECONDS = 5
# https://nodejs.org/api/http.html#serverkeepalivetimeout
NODE_DEFAULT_SECONDS = 5
# https://www.uvicorn.org/settings/#timeouts
UVICORN_DEFAULT_SECONDS = 5
# https://nginx.org/en/docs/http/ngx_http_core_module.html#keepalive_timeout
NGINX_DEFAULT_SECONDS = 75
NGINX_UPSTREAM_DEFAULT_SECONDS = 60
NGINX_TIME_UNITS = {"ms": 0.001, "": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}


class KeepAlive(NamedTuple):
    seconds: float
    # Where to set it, for the failure message.
    setting: str


class Directive(NamedTuple):
    name: str
    args: list
    block: Optional[list]


def closes_first(server: str, keep_alive: KeepAlive, proxies: dict) -> list:
    """A message if the server closes an idle connection before a proxy does."""
    proxy, idle = max(proxies.items(), key=lambda item: item[1])
    if keep_alive.seconds >= idle + HEADROOM_SECONDS:
        return []
    return [
        f"{server} closes an idle connection after {keep_alive.seconds:g} s, but "
        f"{proxy} in front of it keeps one for {idle:g} s, so {proxy} can send a "
        "request on a connection just as the server closes it, and answer 502. "
        f"Set {keep_alive.setting} to at least {idle + HEADROOM_SECONDS:g} s."
    ]


def next_keep_alive(value: Optional[str], setting: str) -> KeepAlive:
    """What Next's standalone server makes of KEEP_ALIVE_TIMEOUT (milliseconds).

    It keeps Node's default unless the value parses (as parseInt does) to a
    positive number.
    """
    number = re.match(r"\s*([+-]?\d+)", value or "")
    milliseconds = int(number.group(1)) if number else 0
    if milliseconds <= 0:
        return KeepAlive(NODE_DEFAULT_SECONDS, f"{setting} (unset, so Node's default)")
    return KeepAlive(milliseconds / 1000, setting)


def backend_keep_alive(server_file: Path, override: Optional[str]) -> KeepAlive:
    """The idle keep-alive of the uvicorn server that server_file configures."""
    setting = (
        "SERVER_KEEP_ALIVE_TIMEOUT_SECONDS (backend/.env, or its default in "
        "settings.py)"
    )
    default = settings_default("server_keep_alive_timeout_seconds")
    passed = (
        server_file.is_file()
        and "server_keep_alive_timeout_seconds"
        in server_file.read_text(encoding="utf-8")
    )
    if default is None or not passed:
        return KeepAlive(
            UVICORN_DEFAULT_SECONDS,
            f"timeout_keep_alive on the uvicorn server {server_file.name} configures "
            "(it has uvicorn's default now)",
        )
    return KeepAlive(float(override) if override else default, setting)


def settings_default(field: str) -> Optional[float]:
    """A Field(default=...) number in backend/util/settings.py."""
    for node in ast.walk(ast.parse(SETTINGS.read_text(encoding="utf-8"))):
        if (
            isinstance(node, ast.AnnAssign)
            and getattr(node.target, "id", None) == field
            and isinstance(node.value, ast.Call)
        ):
            for keyword in node.value.keywords:
                if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                    return float(keyword.value.value)
    return None


def parse_nginx(text: str) -> list:
    """nginx.conf as a tree of directives. Fails on anything it can't read."""
    token = re.compile(
        r"""\s+|\#[^\n]*|(?P<quoted>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(?P<punct>[{};])"""
        r"""|(?P<word>(?:\$\{[^}]*\}|[^\s{};"'\#])(?:\$\{[^}]*\}|[^\s{};])*)"""
    )
    tokens, position = [], 0
    while position < len(text):
        match = token.match(text, position)
        if match is None:
            raise ValueError(f"nginx.conf: can't read from offset {position}")
        position = match.end()
        if match.group("punct"):
            tokens.append(("punct", match.group("punct")))
        elif match.group("quoted"):
            tokens.append(("word", match.group("quoted")[1:-1]))
        elif match.group("word"):
            tokens.append(("word", match.group("word")))
    tree, _ = _nginx_block(tokens, 0, nested=False)
    return tree


def _nginx_block(tokens: list, index: int, nested: bool) -> tuple:
    directives, words = [], []
    while index < len(tokens):
        kind, value = tokens[index]
        index += 1
        if kind == "word":
            words.append(value)
        elif value == "}" and nested and not words:
            return directives, index
        elif value in ";{" and words:
            block = None
            if value == "{":
                block, index = _nginx_block(tokens, index, nested=True)
            directives.append(Directive(words[0], words[1:], block))
            words = []
        else:
            raise ValueError(f"nginx.conf: unexpected {value!r}")
    if nested or words:
        raise ValueError("nginx.conf: unbalanced block")
    return directives, index


def nginx_seconds(value: str) -> float:
    parts = re.findall(r"(\d+)(ms|[smhd]?)", value)
    if not re.fullmatch(r"(?:\d+(?:ms|[smhd])?)+", value):
        raise ValueError(f"nginx.conf: {value!r} is not a time")
    return sum(int(number) * NGINX_TIME_UNITS[unit] for number, unit in parts)


def nginx_problems(nginx_conf: str, upstream_servers: dict) -> list:
    """The hops in front of nginx and behind it that close in the wrong order."""
    http = next(d for d in parse_nginx(nginx_conf) if d.name == "http")
    problems = []
    timeouts = [d for d in _walk(http.block) if d.name == "keepalive_timeout"]
    if not any(d in http.block for d in timeouts):
        timeouts.append(
            Directive("keepalive_timeout", [str(NGINX_DEFAULT_SECONDS)], None)
        )
    for directive in timeouts:
        seconds = nginx_seconds(directive.args[0])
        if seconds:  # 0 turns keep-alive off: nothing idles
            problems += closes_first(
                "The single-container image's nginx",
                KeepAlive(
                    seconds, "keepalive_timeout in single-container/nginx/nginx.conf"
                ),
                PROXY_IDLE_SECONDS,
            )
    for upstream in (d for d in http.block if d.name == "upstream"):
        name = upstream.args[0]
        settings = {d.name: d.args for d in upstream.block}
        if settings.get("keepalive", ["1"])[0] == "0":
            continue
        if "keepalive" in settings and "keepalive_timeout" not in settings:
            problems.append(
                f"nginx's upstream {name} keeps idle connections, but doesn't say "
                "for how long: set keepalive_timeout in it."
            )
        idle = nginx_seconds(
            settings.get("keepalive_timeout", [str(NGINX_UPSTREAM_DEFAULT_SECONDS)])[0]
        )
        if name not in upstream_servers:
            problems.append(
                f"nginx's upstream {name} isn't in this check: add the server behind "
                f"it to single_container_servers() in {Path(__file__).name}."
            )
            continue
        problems += closes_first(
            f"The server behind nginx's upstream {name}",
            upstream_servers[name],
            {"nginx": idle},
        )
    return problems


def _walk(directives: list):
    for directive in directives:
        if directive.name == "upstream":
            continue
        yield directive
        yield from _walk(directive.block or [])


def run_frontend_keep_alive() -> KeepAlive:
    """KEEP_ALIVE_TIMEOUT in the environment run-frontend.sh starts Next with."""
    text = RUN_FRONTEND.read_text(encoding="utf-8")
    body = re.search(r"^\s*frontend_env=\($(.*?)^\s*\)$", text, re.M | re.S)
    if body is None:
        raise AssertionError("run-frontend.sh: can't find frontend_env=( ... )")
    values = [
        match.group(1)
        for line in body.group(1).splitlines()
        if (
            match := re.fullmatch(
                r"""\s*["']?KEEP_ALIVE_TIMEOUT=([^"'\s]*)["']?\s*""", line
            )
        )
    ]
    return next_keep_alive(
        values[-1] if values else None,
        "KEEP_ALIVE_TIMEOUT (milliseconds) in single-container/run-frontend.sh",
    )


def compose_services() -> dict:
    services = {}
    for path in COMPOSE_FILES:
        services.update(
            yaml.safe_load(path.read_text(encoding="utf-8")).get("services") or {}
        )
    return services


def compose_environment(service: dict) -> dict:
    """A service's environment as shipped: its .env.default files, then environment.

    An operator's own .env overrides are theirs to keep in order, so a local
    one doesn't count here.
    """
    environment = {}
    env_files = service.get("env_file") or []
    for entry in [env_files] if isinstance(env_files, (str, dict)) else env_files:
        path = PLATFORM / (entry["path"] if isinstance(entry, dict) else entry)
        if path.name.endswith(".env.default"):
            for line in path.read_text(encoding="utf-8").splitlines():
                key, equals, value = line.partition("=")
                if equals and not key.lstrip().startswith("#"):
                    environment[key.strip()] = value.strip().strip("\"'")
    values = service.get("environment") or {}
    if isinstance(values, list):
        values = dict(item.partition("=")[::2] for item in values)
    environment.update(
        {key: str(value) for key, value in values.items() if value is not None}
    )
    return environment


def published_services(services: dict) -> set:
    """Services with a port published on every host interface: a proxy's targets."""
    loopback = ("127.0.0.1", "::1", "[::1]", "localhost")
    published = set()
    for name, service in services.items():
        for port in service.get("ports") or []:
            host_ip = (
                port.get("host_ip")
                if isinstance(port, dict)
                else str(port).rpartition(":")[0].rpartition(":")[0]
            )
            if host_ip not in loopback:
                published.add(name)
    return published


def compose_servers(services: dict) -> dict:
    """The keep-alive of each server Docker Compose publishes."""
    frontend = compose_environment(services["frontend"]).get("KEEP_ALIVE_TIMEOUT")
    return {
        "frontend": next_keep_alive(
            frontend,
            "KEEP_ALIVE_TIMEOUT (milliseconds) under the frontend service in "
            "docker-compose.platform.yml",
        ),
        "rest_server": backend_keep_alive(
            REST_SERVER,
            compose_environment(services["rest_server"]).get(
                "SERVER_KEEP_ALIVE_TIMEOUT_SECONDS"
            ),
        ),
        "websocket_server": backend_keep_alive(
            WEBSOCKET_SERVER,
            compose_environment(services["websocket_server"]).get(
                "SERVER_KEEP_ALIVE_TIMEOUT_SECONDS"
            ),
        ),
    }


def single_container_servers() -> dict:
    """The keep-alive of the server behind each upstream of the image's nginx."""
    return {
        "autogpt_frontend": run_frontend_keep_alive(),
        "autogpt_rest": backend_keep_alive(REST_SERVER, None),
        "autogpt_websocket": backend_keep_alive(WEBSOCKET_SERVER, None),
    }


class KeepAliveOrderTest(unittest.TestCase):
    def test_compose_servers_outlast_the_proxies_in_front(self):
        services = compose_services()
        servers = compose_servers(services)
        self.assertEqual(
            published_services(services),
            set(servers),
            "Docker Compose publishes a different set of servers than this check "
            "knows. A proxy in front of a new one needs it to keep idle connections "
            "longer than the proxy does: add it to compose_servers().",
        )
        problems = [
            p
            for name, keep_alive in servers.items()
            for p in closes_first(f"Compose's {name}", keep_alive, PROXY_IDLE_SECONDS)
        ]
        self.assertEqual(problems, [], "\n".join(problems))

    def test_single_container_nginx_closes_in_order(self):
        problems = nginx_problems(
            NGINX_CONF.read_text(encoding="utf-8"), single_container_servers()
        )
        self.assertEqual(problems, [], "\n".join(problems))


class ReaderTest(unittest.TestCase):
    """The readers above, and that the check fails on a configuration it should."""

    CONF = """
        http {
            # keepalive_timeout 1s; is a comment, not a setting
            keepalive_timeout %s;
            map $http_upgrade $connection_upgrade { default upgrade; '' ''; }
            upstream app { server 127.0.0.1:3001; keepalive 32; %s }
            upstream ws { server 127.0.0.1:8001; }
            server { location = /x { return 200 'a { b ; c'; } }
        }
    """

    def test_reads_nginx_times(self):
        self.assertEqual(
            [nginx_seconds(v) for v in ("65", "60s", "1m30s", "500ms")],
            [65, 60, 90, 0.5],
        )
        with self.assertRaises(ValueError):
            nginx_seconds("60 s")

    def test_next_keeps_nodes_default_unless_given_a_positive_number(self):
        self.assertEqual(next_keep_alive("620000", "x").seconds, 620)
        for value in (None, "", "0", "-1", "soon"):
            self.assertEqual(
                next_keep_alive(value, "x").seconds, NODE_DEFAULT_SECONDS, value
            )

    def test_passes_servers_that_outlast_nginx(self):
        long = KeepAlive(620, "x")
        conf = self.CONF % ("620s", "keepalive_timeout 60s;")
        self.assertEqual(nginx_problems(conf, {"app": long, "ws": long}), [])

    def test_flags_each_hop_that_closes_first(self):
        conf = self.CONF % ("65", "")
        problems = nginx_problems(
            conf,
            {
                "app": KeepAlive(NODE_DEFAULT_SECONDS, "KEEP_ALIVE_TIMEOUT"),
                "ws": KeepAlive(620, "x"),
            },
        )
        self.assertEqual(len(problems), 3, problems)
        self.assertIn("after 65 s, but Google's load balancer", problems[0])
        self.assertIn("doesn't say for how long", problems[1])
        self.assertIn("Set KEEP_ALIVE_TIMEOUT to at least 65 s", problems[2])

    def test_finds_the_services_published_on_every_interface(self):
        services = {
            "public": {"ports": ["8006:8006"]},
            "ephemeral": {"ports": ["8002"]},
            "long_syntax": {"ports": [{"target": 80, "published": 8080}]},
            "loopback": {"ports": ["127.0.0.1:5432:5432", "[::1]:6379:6379"]},
            "loopback_long": {"ports": [{"target": 80, "host_ip": "127.0.0.1"}]},
            "internal": {"expose": ["8003"]},
        }
        self.assertEqual(
            published_services(services), {"public", "ephemeral", "long_syntax"}
        )

    def test_flags_an_upstream_it_does_not_know(self):
        conf = self.CONF % ("620s", "keepalive_timeout 60s;")
        self.assertIn(
            "isn't in this check", nginx_problems(conf, {"app": KeepAlive(620, "x")})[0]
        )


if __name__ == "__main__":
    unittest.main()
