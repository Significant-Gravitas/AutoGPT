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
    benchmark = Benchmark(args.output, mode="next-route-types")
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

    def generate(name):
        generated = FRONTEND / ".next/types"
        if generated.exists():
            shutil.rmtree(generated)
        benchmark.run(name, ["pnpm", "exec", "next", "typegen"], required=True)
        assert (generated / "validator.ts").is_file()
        assert (generated / "routes.d.ts").is_file()

    try:
        generate("generate-original-route-types")
        for checker in order:
            benchmark.run(f"{checker}-valid-routes", commands[checker], required=True)
        page = probe / "[id]/page.tsx"
        page.parent.mkdir(parents=True)
        page.write_text(
            "export default function Page({ params }: { params: { id: string } }) {\n"
            "  return params.id;\n"
            "}\n",
            encoding="utf-8",
        )
        generate("generate-invalid-route-types")
        for checker in order:
            name = f"{checker}-invalid-route"
            code = benchmark.run(name, commands[checker])
            log = (benchmark.output / f"{name}.log").read_text(encoding="utf-8")
            rejected = code not in (0, 124, 127) and "ci-typecheck-probe" in log
            benchmark.canaries[
                f"{checker}-rejects-invalid-next-page-signature"
            ] = rejected
            if not rejected:
                raise RuntimeError(
                    f"{checker} did not reject the injected route signature"
                )
        shutil.rmtree(probe)
        generate("generate-clean-route-types")
        for checker in order:
            benchmark.run(
                f"{checker}-restored-routes", commands[checker], required=True
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
        raise SystemExit("Route type-check comparison did not preserve its safeguards")


if __name__ == "__main__":
    main()
