"""The Docker Compose stack publishes no internal service port on the host.

Those ports carry calls between the backend's own services; the containers
reach each other over the Compose network, so nothing outside needs them.
Read with a line scan rather than a YAML parser, as this suite runs on the
runner's bare Python.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

PLATFORM_DIR = Path(__file__).parents[2]
INTERNAL_PORTS = {
    "executor": "8002",
    "scheduler_server": "8003",
    "database_manager": "8005",
    "notification_server": "8007",
    "copilot_executor": "8008",
    "platform_linking_manager": "8009",
}
# Every internal service port, including those with no Compose service today.
ANY_INTERNAL = {int(port) for port in INTERNAL_PORTS.values()} | {8010, 8011, 8012}
_TARGET = re.compile(r"\btarget:\s*[\"']?(\d+)")


def _compose_files() -> list[Path]:
    return sorted(
        {*PLATFORM_DIR.glob("*compose*.y*ml"), *PLATFORM_DIR.glob("*/*compose*.y*ml")}
    )


def _services(text: str) -> dict[str, list[str]]:
    """The lines under each service of a Compose file's top-level `services:`."""
    blocks: dict[str, list[str]] = {}
    in_services = False
    current: str | None = None
    for line in text.splitlines():
        if line and not line[0].isspace() and not line.startswith("#"):
            in_services = line.rstrip() == "services:"
            current = None
            continue
        if not in_services:
            continue
        header = re.match(r"^  ([\w.-]+):\s*$", line)
        if header:
            current = header.group(1)
            blocks[current] = []
        elif current is not None:
            blocks[current].append(line)
    return blocks


def _key_items(block: list[str], key: str) -> list[str] | None:
    """The list items under `key:` in a service block, or None if absent."""
    items: list[str] | None = None
    for line in block:
        if re.match(rf"^    {key}:\s*$", line):
            items = []
        elif items is not None:
            item = re.match(r"^      -\s*[\"']?([^\"'#\s]+)", line)
            if item:
                items.append(item.group(1))
            elif line.strip() and not line.lstrip().startswith("#"):
                break
    return items


def _published_ports(text: str) -> set[int]:
    """The container port of every `ports:` entry in a Compose file, in the
    short syntax (`"8005"`, `"18005:8005"`, `"127.0.0.1:18005:8005/tcp"`,
    ranges, inline lists) and the long one (`target: 8005`, also written as an
    inline `{target: 8005}` mapping)."""
    published: set[int] = set()
    ports_indent: int | None = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        is_item = line.lstrip().startswith("-")
        if ports_indent is not None and (
            indent < ports_indent or (indent == ports_indent and not is_item)
        ):
            ports_indent = None
        key = re.match(r"^\s*(?:-\s+)?ports:\s*(.*)$", line)
        if key:
            inline = key.group(1).strip()
            if inline:
                published |= _inline_ports(inline)
            else:
                ports_indent = indent
            continue
        if ports_indent is None:
            continue
        target = _TARGET.search(line)
        entry = re.match(r"^\s*-\s+(\S+)$", line)
        if target:
            published.add(int(target.group(1)))
        elif entry:
            published |= _container_ports(entry.group(1))
    return published


def _inline_ports(inline: str) -> set[int]:
    """A `ports:` list written on one line: `{target: ...}` mappings and short
    entries, in any mix."""
    ports = {int(port) for port in _TARGET.findall(inline)}
    for entry in re.sub(r"\{[^}]*\}", "", inline).strip("[]").split(","):
        ports |= _container_ports(entry)
    return ports


def _container_ports(entry: str) -> set[int]:
    """`[host_ip:][host_port:]container_port[/protocol]`: the container side,
    a port or a range."""
    container = entry.strip().strip("\"'").rsplit(":", 1)[-1].split("/", 1)[0]
    bounds = re.fullmatch(r"(\d+)(?:-(\d+))?", container)
    if not bounds:
        return set()
    first = int(bounds.group(1))
    return set(range(first, int(bounds.group(2) or first) + 1))


class ComposeInternalPortsTest(unittest.TestCase):
    def test_internal_services_are_not_published_to_the_host(self) -> None:
        platform = (PLATFORM_DIR / "docker-compose.platform.yml").read_text(
            encoding="utf-8"
        )
        services = _services(platform)
        for name, port in INTERNAL_PORTS.items():
            with self.subTest(service=name):
                self.assertIn(name, services)
                self.assertIsNone(_key_items(services[name], "ports"))
                self.assertEqual(_key_items(services[name], "expose"), [port])

    def test_no_compose_file_maps_an_internal_port_to_the_host(self) -> None:
        files = _compose_files()
        self.assertIn(PLATFORM_DIR / "docker-compose.platform.yml", files)
        for compose_file in files:
            with self.subTest(file=compose_file.name):
                text = compose_file.read_text(encoding="utf-8")
                self.assertEqual(_published_ports(text) & ANY_INTERNAL, set())

    def test_the_scan_sees_a_published_port_in_any_form(self) -> None:
        """Guards the line scan itself."""
        for entries in (
            '      - "8005:8005"',
            "      - 8005:8005",
            '      - "18005:8005"',
            '      - "127.0.0.1:18005:8005"',
            '      - "[::1]:18005:8005/tcp"',
            '      - "8005"',
            "      - 8000-8010:8000-8010",
            '    - "8005:8005"',  # items level with the key are still its items
            "      - target: 8005\n        published: 18005",
            "      - published: 18005\n        target: 8005",
            "      - {target: 8005, published: 18005}",
        ):
            with self.subTest(entries=entries):
                text = f"services:\n  db:\n    ports:\n{entries}\n"
                self.assertIn(8005, _published_ports(text))
        for inline in (
            '["3000:3000", "8005:8005"]',
            '["3000:3000", {target: 8005, published: "18005"}]',
        ):
            with self.subTest(inline=inline):
                text = f"services:\n  db:\n    ports: {inline}\n"
                self.assertEqual(_published_ports(text), {3000, 8005})

    def test_the_scan_sees_only_published_ports(self) -> None:
        text = (
            "services:\n"
            "  db:\n"
            "    expose:\n"
            '      - "8005"\n'
            "    ports:\n"
            '      - "8005:18005"\n'
            '      - "127.0.0.1:3000:3000"  # the frontend\n'
            "    environment:\n"
            "      - PORT=8005\n"
            "    command: [serve, --port, '8005']\n"
        )
        self.assertEqual(_published_ports(text), {18005, 3000})
        block = ["    ports:", '      - "8005:8005"', "    networks:", "      - x"]
        self.assertEqual(_key_items(block, "ports"), ["8005:8005"])
        self.assertIsNone(_key_items(block, "expose"))


if __name__ == "__main__":
    unittest.main()
