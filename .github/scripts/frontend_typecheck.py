import json
import subprocess
from pathlib import Path


def compatibility_config(options: dict, implicit_types: list[str]) -> dict:
    """Keep TypeScript 5's project scope and defaults with the newer checker."""
    return {
        "extends": "./tsconfig.json",
        "compilerOptions": {
            "incremental": False,
            "types": options.get("types", implicit_types),
            "noUncheckedSideEffectImports": options.get(
                "noUncheckedSideEffectImports", False
            ),
            "paths": {
                name: [
                    path if path.startswith(("./", "../", "/")) else f"./{path}"
                    for path in paths
                ]
                for name, paths in options.get("paths", {}).items()
            },
        },
    }


def main() -> int:
    options = json.loads(
        subprocess.check_output(["pnpm", "exec", "tsc", "--showConfig"], text=True)
    )["compilerOptions"]
    implicit_types = json.loads(
        subprocess.check_output(
            [
                "node",
                "-e",
                'const ts = require("typescript"); console.log(JSON.stringify(ts.getAutomaticTypeDirectiveNames({}, ts.sys)));',
            ],
            text=True,
        )
    )
    config = Path("tsconfig.ci-typecheck.json")
    with config.open("x", encoding="utf-8") as stream:
        json.dump(compatibility_config(options, implicit_types), stream, indent=2)
    try:
        return subprocess.run(
            ["bun", "check", "--no-script", "--project", str(config), "--no-pretty"]
        ).returncode
    finally:
        config.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
