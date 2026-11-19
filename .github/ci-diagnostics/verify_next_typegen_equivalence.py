import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".github/scripts"))

from benchmark_frontend import FRONTEND, Benchmark  # noqa: E402

BUILD_SHA = "e9f613b386b764dd9f95c556d025ce84d85a4a9e"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-reports", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("GITHUB_ACTIONS") != "true" or sys.platform != "linux":
        parser.error("Application checks run only in hosted Linux CI")
    benchmark = Benchmark(args.output, mode="next-typegen-parity")
    expected_reports = {f"next-build-types-{first}-first" for first in ("tsc", "bun")}
    assert {p.name for p in args.build_reports.iterdir()} == expected_reports
    probe = FRONTEND / "src/app/ci-typecheck-probe"
    assert not probe.exists()
    tracked = {
        path: path.read_bytes() if path.exists() else None
        for path in (FRONTEND / "next-env.d.ts", FRONTEND / "tsconfig.json")
    }
    valid_page = (
        "export default async function Page({ params }: "
        "{ params: Promise<{ id: string }> }) {\n"
        "  return (await params).id;\n"
        "}\n"
    )
    page = probe / "[id]/page.tsx"
    command = [sys.executable, str(ROOT / ".github/scripts/frontend_typecheck.py")]
    try:
        page.parent.mkdir(parents=True)
        page.write_text(valid_page, encoding="utf-8")
        benchmark.run(
            "next-typegen", ["pnpm", "exec", "next", "typegen"], required=True
        )
        generated = FRONTEND / ".next/types"
        names = {p.relative_to(generated) for p in generated.rglob("*") if p.is_file()}
        assert names == {Path("routes.d.ts"), Path("validator.ts")}
        lock_hash = hashlib.sha256(
            (FRONTEND / "pnpm-lock.yaml").read_bytes()
        ).hexdigest()
        for report in args.build_reports.iterdir():
            metadata = json.loads((report / "measurements.json").read_text())
            assert metadata["source_sha"] == BUILD_SHA
            assert metadata["lock_sha256"] == lock_hash
            assert metadata["mode"] == "next-build-types"
            assert all(metadata["canaries"].values())
            original = report / "build-generated-types"
            assert {
                p.relative_to(original) for p in original.rglob("*") if p.is_file()
            } == names
            equal = all(
                (original / name).read_bytes() == (generated / name).read_bytes()
                for name in names
            )
            benchmark.canaries[f"identical-to-{report.name}"] = equal
            if not equal:
                raise RuntimeError(f"Standalone typegen differs from {report.name}")
        shutil.copytree(generated, benchmark.output / "standalone-generated-types")
        benchmark.run("bun-valid-typegen", command, required=True)
        page.write_text(
            "export default function Page() {\n"
            "  return { notAReactNode: true };\n}\n",
            encoding="utf-8",
        )
        code = benchmark.run("bun-invalid-page", command)
        log = (benchmark.output / "bun-invalid-page.log").read_text(encoding="utf-8")
        rejected = (
            code not in (0, 124, 127) and code > 0 and "ci-typecheck-probe" in log
        )
        benchmark.canaries["reject-invalid-page-return"] = rejected
        if not rejected:
            raise RuntimeError("Standalone typegen did not reject the invalid page")
        page.write_text(valid_page, encoding="utf-8")
        benchmark.run("bun-restored-typegen", command, required=True)
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
        raise SystemExit("Typegen equivalence did not preserve its safeguards")


if __name__ == "__main__":
    main()
