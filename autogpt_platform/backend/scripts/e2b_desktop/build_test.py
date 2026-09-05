"""Template resource flags reach the builder without provisioning resources."""

from unittest.mock import patch

import pytest

from scripts.e2b_desktop.build import main


@pytest.mark.parametrize(
    "args,cpu,memory,alias",
    [
        ([], 8, 8192, "autogpt-code-desktop"),
        (
            ["--cpu-count", "4", "--memory-mb", "4096", "--alias", "small"],
            4,
            4096,
            "small",
        ),
    ],
)
def test_build_resources(args, cpu, memory, alias):
    with (
        patch("sys.argv", ["build.py", *args]),
        patch("scripts.e2b_desktop.build.load_dotenv"),
        patch("scripts.e2b_desktop.build.Template.build") as build,
    ):
        main()
    assert build.call_count == 1
    assert build.call_args.kwargs["cpu_count"] == cpu
    assert build.call_args.kwargs["memory_mb"] == memory
    assert build.call_args.kwargs["alias"] == alias
