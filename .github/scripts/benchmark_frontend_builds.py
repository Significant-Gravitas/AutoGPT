from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmark_frontend import Benchmark


def compare_builds(benchmark: Benchmark, frontend: Path, first: str) -> None:
    source = frontend / "src/lib/keyboard.ts"
    original = source.read_bytes()
    routes = []
    inputs = {
        "first_compiler": first,
        "source_probe": str(source.relative_to(frontend)),
        "source_probe_sha256": hashlib.sha256(original).hexdigest(),
        "configuration_sha256": hashlib.sha256(
            (frontend / "next.config.mjs").read_bytes()
        ).hexdigest(),
        "node_options": os.environ.get("NODE_OPTIONS"),
        "browser_sourcemaps": os.environ.get("NEXT_PUBLIC_SOURCEMAPS"),
        "skip_build_checks": os.environ.get("NEXT_SKIP_BUILD_CHECKS"),
        "playwright_flags": os.environ.get("NEXT_PUBLIC_PW_TEST"),
    }
    (benchmark.output / "build-inputs.json").write_text(json.dumps(inputs, indent=2))
    compilers = [first, "turbopack" if first == "webpack" else "webpack"]
    try:
        for compiler in compilers:
            source.write_bytes(original)
            command = ["pnpm", "exec", "next", "build"]
            if compiler == "turbopack":
                command.append("--turbopack")
            for case in ("cold", "unchanged", "edited"):
                reset_output(frontend, keep_cache=case != "cold")
                if case == "edited":
                    source.write_bytes(original + b"\n// CI build-cache probe.\n")
                code = benchmark.run(f"{compiler}-{case}", command)
                if code:
                    break
                try:
                    report = build_report(frontend)
                    benchmark.records[-1].update(report)
                    routes.append(report["routes"])
                except (OSError, ValueError) as error:
                    benchmark.records[-1]["report_error"] = str(error)
                    break
    finally:
        source.write_bytes(original)
    benchmark.canaries["all-six-builds-produced-standalone-and-sourcemaps"] = (
        len(routes) == 6
    )
    benchmark.canaries["same-built-route-inventory"] = len(routes) == 6 and all(
        route_set == routes[0] for route_set in routes
    )


def reset_output(frontend: Path, *, keep_cache: bool) -> None:
    output = frontend / ".next"
    if output.is_symlink():
        raise ValueError("Benchmark output must not be a symlink")
    if not output.exists():
        return
    for path in output.iterdir():
        if path.is_symlink():
            path.unlink()
        elif path.name == "cache" and keep_cache:
            continue
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def build_report(frontend: Path) -> dict:
    output = frontend / ".next"
    if not (output / "standalone/server.js").is_file():
        raise ValueError("Missing standalone server")
    source_maps = list((output / "static").rglob("*.js.map"))
    if not source_maps:
        raise ValueError("Missing browser source maps")
    routes = {}
    for kind in ("app-paths", "pages"):
        manifest = json.loads((output / f"server/{kind}-manifest.json").read_text())
        if not isinstance(manifest, dict):
            raise ValueError(f"Invalid {kind} manifest")
        routes[kind] = sorted(manifest)
    if not routes["app-paths"]:
        raise ValueError("Missing application routes")
    return {
        "routes": routes,
        "browser_source_maps": len(source_maps),
        "cache_bytes": sum(
            path.stat().st_size
            for path in (output / "cache").rglob("*")
            if path.is_file()
        ),
    }
