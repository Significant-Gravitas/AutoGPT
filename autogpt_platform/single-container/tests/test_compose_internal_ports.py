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
COMPOSE_FILES = ("docker-compose.yml", "docker-compose.platform.yml")
INTERNAL_PORTS = {
    "executor": "8002",
    "scheduler_server": "8003",
    "database_manager": "8005",
    "notification_server": "8007",
    "copilot_executor": "8008",
    "platform_linking_manager": "8009",
}
# Every internal service port, including those with no Compose service today.
_ANY_INTERNAL = "|".join(sorted({*INTERNAL_PORTS.values(), "8010", "8011", "8012"}))
_PUBLISHED = re.compile(
    rf"""^\s*-\s*["']?(?:[\w.:\[\]]+:)?(?:{_ANY_INTERNAL}):\d+""", re.MULTILINE
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


class ComposeInternalPortsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.platform = (PLATFORM_DIR / "docker-compose.platform.yml").read_text(
            encoding="utf-8"
        )

    def test_internal_services_are_not_published_to_the_host(self) -> None:
        services = _services(self.platform)
        for name, port in INTERNAL_PORTS.items():
            with self.subTest(service=name):
                self.assertIn(name, services)
                self.assertIsNone(_key_items(services[name], "ports"))
                self.assertEqual(_key_items(services[name], "expose"), [port])

    def test_no_compose_file_maps_an_internal_port_to_the_host(self) -> None:
        for compose_file in COMPOSE_FILES:
            text = (PLATFORM_DIR / compose_file).read_text(encoding="utf-8")
            with self.subTest(file=compose_file):
                self.assertEqual(_PUBLISHED.findall(text), [])

    def test_the_scan_sees_a_published_port(self) -> None:
        """Guards the line scan itself: a mapping in any form is found."""
        for mapping in ('"8005:8005"', "8005:8005", '"127.0.0.1:8003:8003"'):
            with self.subTest(mapping=mapping):
                self.assertTrue(_PUBLISHED.search(f"    ports:\n      - {mapping}\n"))
        self.assertFalse(_PUBLISHED.search('      - "18005:18005"\n'))
        block = ["    ports:", '      - "8005:8005"', "    networks:", "      - x"]
        self.assertEqual(_key_items(block, "ports"), ["8005:8005"])
        self.assertIsNone(_key_items(block, "expose"))


if __name__ == "__main__":
    unittest.main()
