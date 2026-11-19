import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".github/scripts"))

from benchmark_frontend import FRONTEND, Benchmark  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first", choices=("tsc", "bun"), required=True)
    args = parser.parse_args()
    if os.environ.get("GITHUB_ACTIONS") != "true" or sys.platform != "linux":
        parser.error("Application comparisons run only in hosted Linux CI")
    benchmark = Benchmark(args.output, mode="next-build-types")
    probe = FRONTEND / "src/app/ci-typecheck-probe"
    assert not probe.exists()
    tracked = {
        path: path.read_bytes() if path.exists() else None
        for path in (FRONTEND / "next-env.d.ts", FRONTEND / "tsconfig.json")
    }
    commands = {
        "tsc": ["pnpm", "exec", "tsc", "--noEmit", "--incremental", "false"],
        "bun": [sys.executable, str(ROOT / ".github/scripts/frontend_typecheck.py")],
    }
    order = [args.first, "bun" if args.first == "tsc" else "tsc"]
    valid_page = (
        "export default async function Page({ params }: "
        "{ params: Promise<{ id: string }> }) {\n"
        "  return (await params).id;\n"
        "}\n"
    )
    page = probe / "[id]/page.tsx"
    try:
        page.parent.mkdir(parents=True)
        page.write_text(valid_page, encoding="utf-8")
        benchmark.run("copy-vad-assets", ["pnpm", "copy:vad-assets"], required=True)
        benchmark.run(
            "next-build-with-original-checks",
            ["pnpm", "exec", "next", "build", "--turbopack"],
            required=True,
        )
        build_log = (
            benchmark.output / "next-build-with-original-checks.log"
        ).read_text(encoding="utf-8")
        assert "Linting and checking validity of types" in build_log
        assert "Skipping validation of types" not in build_log
        generated = FRONTEND / ".next/types"
        assert generated.is_dir()
        shutil.copytree(generated, benchmark.output / "build-generated-types")
        for checker in order:
            benchmark.run(
                f"{checker}-valid-build-types", commands[checker], required=True
            )
        for name, content in (
            (
                "sync-page-params",
                "export default function Page({ params }: { params: { id: string } }) {\n"
                "  return params.id;\n}\n",
            ),
            (
                "invalid-page-return",
                "export default function Page() {\n"
                "  return { notAReactNode: true };\n}\n",
            ),
        ):
            page.write_text(content, encoding="utf-8")
            outcomes = {}
            for checker in order:
                label = f"{checker}-{name}"
                code = benchmark.run(label, commands[checker])
                log = (benchmark.output / f"{label}.log").read_text(encoding="utf-8")
                if code in (124, 127) or code < 0:
                    raise RuntimeError(f"{checker} did not complete: {label}")
                if code and "ci-typecheck-probe" not in log:
                    raise RuntimeError(f"Unexpected error while checking {label}")
                outcomes[checker] = code != 0
            equal = outcomes["tsc"] == outcomes["bun"]
            benchmark.canaries[f"equal-{name}-result"] = equal
            if not equal or (name == "invalid-page-return" and not outcomes["tsc"]):
                raise RuntimeError(f"Generated type validation differs for {name}")
            if name == "invalid-page-return":
                benchmark.canaries["reject-invalid-page-return"] = all(
                    outcomes.values()
                )
        page.write_text(valid_page, encoding="utf-8")
        for checker in order:
            benchmark.run(
                f"{checker}-restored-build-types", commands[checker], required=True
            )
    finally:
        if probe.exists():
            shutil.rmtree(probe)
        for path, original in tracked.items():
            if original is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(original)
        benchmark.finish()
    if not all(benchmark.canaries.values()):
        raise SystemExit("Build type-check comparison did not preserve its safeguards")


if __name__ == "__main__":
    main()
