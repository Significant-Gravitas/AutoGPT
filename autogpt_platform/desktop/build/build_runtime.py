"""Assemble the desktop runtime for the platform this script runs on.

    python build_runtime.py [--out DIR] [--only step,step] [--skip step,step]

The result is the directory the Electron shell ships as resources/runtime
(layout: runtime/autogpt_desktop/layout.py). Each step is independent and
idempotent, so a failed build resumes with --only.

Needs on PATH: uv, and pnpm (via corepack or standalone) for the frontend.
Everything else, Node included, is downloaded pinned from artifacts.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import backend_patches
import backend_tests
import bundled_tools
import claude_cli
import elf
import erlang_patches
import frontend_role_sql
import lock_export
import next_config
import site_zip
import slim
from artifacts import ARTIFACTS, CLAUDE_CLI_VERSION, PYTHON_VERSION, platform_key
from fetch import download, extract

DESKTOP = Path(__file__).resolve().parents[1]
PLATFORM = DESKTOP.parent
REPO = PLATFORM.parent
LOCK = PLATFORM / "backend" / "poetry.lock"
# What makes the interpreter read site/ (step_relocate) and, after it, the
# archives of pure-Python packages beside it (step_zip). One line: a .pth line
# that starts with `import` is run as it stands, inside a function of
# site.py, where a comprehension or a generator would not see the names the
# line itself defines.
#
# Its last part gives code loaded from an archive the name of the file it
# came from. The import of a .pyc file does that (`_imp._fix_co_filename`);
# zipimport does not, and the bytecode in the archives is compiled under a
# name that is the same wherever the bundle is built (site_zip.py). Left so,
# a traceback names `site-zip/openai/_client.py`, a file that is nowhere, and
# a package that tells its own frames from a caller's by file name (neo4j,
# e2b) takes its own for the caller's. It leans on two private names of
# CPython, and where either is gone it does nothing: the app starts all the
# same and the build is stopped by its gate (bundle_gate.py,
# code_named_for_another_file), which says what to read.
RELOCATION_FILE = "autogpt-desktop.pth"
RELOCATION = (
    "import os, site, sys, zipimport, _imp; "
    "root = os.path.dirname(os.path.abspath(sys.prefix)); "
    f'site.addsitedir(os.path.join(root, "{site_zip.PACKAGES}")); '
    f'zips = os.path.join(root, "{site_zip.ARCHIVES}"); '
    "names = sorted(os.listdir(zips)) if os.path.isdir(zips) else []; "
    "sys.path.extend(map(os.path.join, [zips] * len(names), names)); "
    'load = getattr(zipimport, "_unmarshal_code", None); '
    'rename = getattr(_imp, "_fix_co_filename", None); '
    'load and rename and setattr(zipimport, "_unmarshal_code", '
    "lambda importer, path, *rest, load=load, rename=rename: "
    "(code := load(importer, path, *rest), code and rename(code, path[:-1]))[0])\n"
)
WINDOWS = sys.platform == "win32"
EXE = ".exe" if WINDOWS else ""
# The Prisma engines a Linux bundle carries, whatever the build machine has:
# the CLI picks by the first libssl it finds, and on a machine with OpenSSL 1.1
# and 3 both installed (GitHub's ubuntu-22.04) that is 1.1, which the systems
# the app is for (README: OpenSSL 3) no longer have.
LINUX_ENGINE_TARGETS = {
    "linux-x64": "debian-openssl-3.0.x",
    "linux-arm64": "linux-arm64-openssl-3.0.x",
}
OPENSSL = "libssl.so.3"

# Build-time settings of the frontend, identical to the appliance image
# (single-container/Dockerfile): every browser-facing URL is same-origin and
# the runtime's proxy decides where it lands.
FRONTEND_BUILD_ENV = {
    "NODE_ENV": "production",
    "NEXT_TELEMETRY_DISABLED": "1",
    "NEXT_PUBLIC_AGPT_SERVER_URL": "/_agpt/api",
    "NEXT_PUBLIC_AGPT_WS_SERVER_URL": "/_agpt/ws",
    "NEXT_PUBLIC_FRONTEND_BASE_URL": "",
    "NEXT_PUBLIC_APP_ENV": "local",
    "NEXT_PUBLIC_BEHAVE_AS": "LOCAL",
    "NEXT_PUBLIC_LAUNCHDARKLY_ENABLED": "false",
    "NEXT_PUBLIC_LAUNCHDARKLY_CLIENT_ID": "",
    "NEXT_PUBLIC_POSTHOG_KEY": "",
    "NEXT_PUBLIC_POSTHOG_HOST": "",
    "NEXT_PUBLIC_COOKIEBOT_CBID": "",
    "NEXT_PUBLIC_COOKIEBOT_GEOREGIONS": "",
    "NEXT_PUBLIC_GOOGLE_ADS_ID": "",
    "NEXT_PUBLIC_GOOGLE_ADS_CONVERSION_LABELS": "",
    "NEXT_PUBLIC_GA_MEASUREMENT_ID": "",
    "DATAFAST_API_KEY": "",
    "SENTRY_AUTH_TOKEN": "",
    "SENTRY_DSN": "",
    "DISABLE_SENTRY": "true",
    "OTEL_SDK_DISABLED": "true",
    "NEXT_PUBLIC_FORCE_FLAG_GRAPHITI_MEMORY": "false",
    "NEXT_PUBLIC_SOURCEMAPS": "false",
    "NEXT_PUBLIC_TURNSTILE": "disabled",
    "NEXT_PUBLIC_VAPID_PUBLIC_KEY": "",
    "NEXT_SKIP_BUILD_CHECKS": "true",
    "BETTER_AUTH_SECRET": "build-only-placeholder-not-used-at-runtime",
    "DATABASE_URL": "postgresql://build:build@127.0.0.1:1/postgres",
    "CI": "true",
}


def frontend_build_environment() -> dict[str, str]:
    return {
        **{
            name: value
            for name, value in os.environ.items()
            if name.upper() not in FRONTEND_BUILD_ENV
            and not name.upper().startswith(
                (
                    "SENTRY_",
                    "POSTHOG_",
                    "LANGFUSE_",
                    "LANGSMITH_",
                    "LANGCHAIN_",
                    "OTEL_",
                )
            )
        },
        **FRONTEND_BUILD_ENV,
    }


class Build:
    def __init__(self, out: Path, cache: Path, redis_stand_in: bool = False) -> None:
        self.out = out
        self.cache = cache
        self.redis_stand_in = redis_stand_in
        self.artifacts = ARTIFACTS[platform_key()]
        self.python = out / "python" / ("python.exe" if WINDOWS else "bin/python3")
        self.site_packages = (
            out
            / "python"
            / ("Lib/site-packages" if WINDOWS else "lib/python3.13/site-packages")
        )
        self.node = out / "node" / f"node{EXE}"

    def fetch(self, name: str) -> Path:
        return download(self.artifacts[name], self.cache)

    # --- interpreters -----------------------------------------------------

    def step_python(self) -> None:
        extract(self.fetch("python"), self.out / "python", strip_top_level=True)
        shutil.rmtree(self.out / "site", ignore_errors=True)  # see step_relocate
        # What step_zip made of an earlier build's packages and standard
        # library: the interpreter would read the archive before the files.
        shutil.rmtree(self.out / site_zip.ARCHIVES, ignore_errors=True)
        for archive in (self.out / "python").glob("python3*.zip"):
            archive.unlink()
        # uv treats a standalone build as externally managed; this tree is
        # ours to install into.
        for marker in (self.out / "python").rglob("EXTERNALLY-MANAGED"):
            marker.unlink()

    def step_node(self) -> None:
        source = self.fetch("node")
        target = self.out / "node"
        if WINDOWS:
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, self.node)
            return
        extract(source, self.cache / "node-dist", strip_top_level=True)
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.cache / "node-dist" / "bin" / "node", self.node)
        shutil.copy2(self.cache / "node-dist" / "LICENSE", target / "LICENSE")

    # --- AutoGPT backend --------------------------------------------------

    def step_deps(self) -> None:
        self._restore_site_packages()
        requirements = self.cache / "requirements.txt"
        lines = lock_export.export(LOCK)
        lines.append(f"autogpt-libs @ {(PLATFORM / 'autogpt_libs').as_uri()}")
        requirements.write_text("\n".join(lines) + "\n", encoding="utf-8")
        run(
            [
                "uv",
                "pip",
                "sync",
                "--python",
                str(self.python),
                "--break-system-packages",
                "--reinstall",
                "--link-mode",
                "copy",
                str(requirements),
            ]
        )
        run(["uv", "pip", "check", "--python", str(self.python)])
        # The CLI the locked SDK is built with: from its wheel, or where
        # this platform has none, put there by claude_cli.py.
        cli = claude_cli.ensure(
            self.site_packages / "claude_agent_sdk" / "_bundled",
            self.artifacts.get("claude-cli"),
            CLAUDE_CLI_VERSION,
            self.cache,
        )
        version = claude_cli.declared_version(cli.parents[1])
        print(f"  {cli.relative_to(self.out)}: Claude Code {version}")

    def _restore_site_packages(self) -> None:
        """uv installs into site-packages, and only there does it see what
        is installed already. A bundle that step_relocate has been through
        keeps its packages in site/: put them back, so that running this step
        again changes what the lock changed instead of installing a second
        copy of everything. step_relocate has to run again afterwards;
        step_seal refuses a bundle where it has not."""
        relocated = self.out / "site"
        if not relocated.is_dir():
            return
        site_zip.unpack(self.out)  # step_zip; uv sees files only
        (self.site_packages / RELOCATION_FILE).unlink(missing_ok=True)
        leftovers = [path.name for path in self.site_packages.iterdir()]
        if leftovers:
            raise RuntimeError(
                f"both {relocated} and {self.site_packages} hold packages ({leftovers[:5]}); "
                "run the python step, then deps again"
            )
        self.site_packages.rmdir()
        shutil.move(relocated, self.site_packages)

    def step_backend(self) -> None:
        target = self.out / "backend"
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        source = PLATFORM / "backend"
        shutil.copytree(
            source / "backend",
            target / "backend",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        # Without its tests, except the modules that are only named like one
        # (backend_tests.py).
        left_out = backend_tests.prune(target / "backend")
        print(f"  left out {len(left_out)} test modules of the backend")
        shutil.copytree(source / "migrations", target / "migrations")
        shutil.copy2(source / "schema.prisma", target / "schema.prisma")
        # Before step_compile, which ships the bytecode of what is here now.
        backend_patches.apply(target)
        # backend/util/docs.py finds the docs by walking up to a `docs/platform`
        # directory; markdown only, as in the backend Dockerfile.
        docs = self.out / "docs"
        if docs.exists():
            shutil.rmtree(docs)
        for path in (REPO / "docs").rglob("*"):
            if path.suffix in (".md", ".mdx") and path.is_file():
                destination = docs / path.relative_to(REPO / "docs")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)

    def step_prisma(self) -> None:
        """Generate the Prisma client into the bundled interpreter and keep
        the CLI and engines it downloaded, so nothing is fetched at runtime."""
        prisma_cache = self.cache / f"prisma-{platform_key()}"
        scripts = self.python.parent / "Scripts" if WINDOWS else self.python.parent
        env = {
            **os.environ,
            "PRISMA_BINARY_CACHE_DIR": str(prisma_cache),
            # The Prisma CLI spawns the `prisma-client-py` generator by name
            # and needs a node; both come from the bundle being built.
            "PATH": os.pathsep.join(
                [
                    str(scripts),
                    str(self.python.parent),
                    str(self.node.parent),
                    os.environ["PATH"],
                ]
            ),
        }
        target = LINUX_ENGINE_TARGETS.get(platform_key())
        if target:
            # Read by the CLI's install script and by the CLI itself, in place
            # of the platform they would detect (@prisma/engines).
            env["PRISMA_CLI_BINARY_TARGETS"] = target
        run(
            [str(self.python), "-m", "prisma", "generate"],
            cwd=self.out / "backend",
            env=env,
        )

        target = self.out / "prisma"
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        modules = _single(
            prisma_cache.rglob("node_modules/prisma/build/index.js")
        ).parents[2]
        shutil.copytree(
            modules,
            target / "node_modules",
            ignore=shutil.ignore_patterns(
                ".cache", "*.dll.node", "*.so.node", "*.dylib.node"
            ),
        )
        engines = modules / "@prisma" / "engines"
        for kind in ("query-engine", "schema-engine"):
            shutil.copy2(
                fetched_engine(engines, kind, platform_key()), target / f"{kind}{EXE}"
            )

    # --- infrastructure ---------------------------------------------------

    def step_postgres(self) -> None:
        target = self.out / "postgres"
        if sys.platform.startswith("linux"):
            self._build_postgres(target)
        else:
            extract(self.fetch("postgres"), target, strip_top_level=True)
        shutil.rmtree(target / "include", ignore_errors=True)

    def _build_postgres(self, target: Path) -> None:
        """PostgreSQL, pg_trgm and pgvector from source, with no optional
        libraries: the result depends on glibc alone. The runtime sets
        LD_LIBRARY_PATH to postgres/lib, and PostgreSQL finds its own share/
        and lib/ relative to the binary, so the tree can be moved."""
        jobs = f"-j{os.cpu_count() or 2}"
        source = self.cache / "postgres-src"
        extract(self.fetch("postgres"), source, strip_top_level=True)
        if target.exists():
            shutil.rmtree(target)
        run(
            [
                "./configure",
                f"--prefix={target}",
                "--without-readline",
                "--without-zlib",
                "--without-icu",
            ],
            cwd=source,
        )
        run(["make", jobs], cwd=source)
        run(["make", "install"], cwd=source)
        run(["make", "-C", "contrib/pg_trgm", "install"], cwd=source)

        vector = self.cache / "pgvector-src"
        extract(self.fetch("pgvector"), vector, strip_top_level=True)
        # OPTFLAGS empty: pgvector defaults to -march=native, which would tie
        # the binary to the build machine's CPU.
        pg_config = f"PG_CONFIG={target / 'bin' / 'pg_config'}"
        run(["make", jobs, pg_config, "OPTFLAGS="], cwd=vector)
        run(["make", "install", pg_config, "OPTFLAGS="], cwd=vector)

        for unused in ("lib/pgxs", "share/doc", "share/man"):
            shutil.rmtree(target / unused, ignore_errors=True)
        binaries = [str(path) for path in (target / "bin").iterdir() if path.is_file()]
        libraries = [
            str(path)
            for path in (target / "lib").glob("*.so*")
            if not path.is_symlink()
        ]
        run(["strip", "--strip-unneeded", *binaries, *libraries], check=False)

    def step_valkey(self) -> None:
        target = self.out / "valkey"
        if target.exists():
            shutil.rmtree(target)
        # Windows: valkey-windows.sh (run under MSYS2) leaves its output here.
        prebuilt = self.cache / "valkey-windows"
        if WINDOWS and (prebuilt / "valkey-server.exe").is_file():
            shutil.copytree(prebuilt, target)
            return
        if WINDOWS and not self.redis_stand_in:
            raise RuntimeError(
                f"No Valkey build in {prebuilt}. Run build/valkey-windows.sh under "
                "MSYS2 first. For a local build only, --redis-stand-in bundles the "
                "redis-windows build of Redis instead; do not publish that one."
            )
        source = self.fetch("valkey")
        staging = self.cache / "valkey-dist"
        extract(source, staging, strip_top_level=True)
        target.mkdir(parents=True)
        if WINDOWS:
            shutil.copy2(staging / "redis-server.exe", target / "valkey-server.exe")
            for library in staging.glob("*.dll"):
                shutil.copy2(library, target / library.name)
        elif sys.platform == "darwin":
            run(
                ["make", f"-j{os.cpu_count() or 2}", "BUILD_TLS=no", "valkey-server"],
                cwd=staging,
            )
            shutil.copy2(staging / "src" / "valkey-server", target / "valkey-server")
            shutil.copy2(staging / "COPYING", target / "COPYING")
        else:
            shutil.copy2(staging / "bin" / "valkey-server", target / "valkey-server")

    def step_erlang(self) -> None:
        target = self.out / "erlang"
        extract(self.fetch("erlang"), target, strip_top_level=not WINDOWS)
        if WINDOWS:
            _add_msvc_runtime(target)
            (target / "vc_redist.exe").unlink(missing_ok=True)
        elif sys.platform.startswith("linux"):
            _install_erlang(target)
        for unused in ("doc", "usr/include"):
            shutil.rmtree(target / unused, ignore_errors=True)

    def step_rabbitmq(self) -> None:
        extract(self.fetch("rabbitmq"), self.out / "rabbitmq", strip_top_level=True)

    def step_erlang_patches(self) -> None:
        erlc = self.out / "erlang" / "bin" / f"erlc{EXE}"
        erlang_patches.build(erlc, self.cache, self.out / "erlang-patches")

    # --- frontend ---------------------------------------------------------

    def step_frontend(self) -> None:
        frontend = PLATFORM / "frontend"
        env = frontend_build_environment()
        heap = int(os.environ.get("AUTOGPT_DESKTOP_BUILD_HEAP_MB", "4096"))
        if heap < 512:
            raise ValueError("AUTOGPT_DESKTOP_BUILD_HEAP_MB must be at least 512")
        env["NODE_OPTIONS"] = f"--max-old-space-size={heap}"
        # A flat node_modules: pnpm's default symlink farm does not survive
        # being copied into an installer (Windows junctions store absolute
        # paths, and symlinks need elevation to create).
        pnpm = self._pnpm(env)
        run(
            [
                *pnpm,
                "install",
                "--prod=false",
                "--frozen-lockfile",
                "--config.node-linker=hoisted",
                "--config.confirm-modules-purge=false",
            ],
            cwd=frontend,
            env=env,
        )
        run([*pnpm, "run", "generate:api"], cwd=frontend, env=env)
        shutil.rmtree(frontend / ".next", ignore_errors=True)
        run([*pnpm, "run", "copy:vad-assets"], cwd=frontend, env=env)
        run([*pnpm, "exec", "next", "build"], cwd=frontend, env=env)

        target = self.out / "frontend"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(frontend / ".next" / "standalone", target, symlinks=False)
        shutil.copytree(frontend / ".next" / "static", target / ".next" / "static")
        shutil.copytree(frontend / "public", target / "public")
        # webpack's build cache is several GB and only speeds up a rebuild.
        shutil.rmtree(frontend / ".next" / "cache", ignore_errors=True)
        self.step_frontend_config()

    def step_frontend_config(self) -> None:
        """Keep the Next server from writing its caches into the bundle
        (next_config.py). A step of its own so that it can be applied to a
        frontend that is already built."""
        target = self.out / "frontend"
        shutil.rmtree(target / ".next" / "cache", ignore_errors=True)
        next_config.keep_caches_off_disk(target)

    def _pnpm(self, env: dict[str, str]) -> list[str]:
        if WINDOWS:
            # pnpm on Windows is a .cmd/.ps1 shim, which CreateProcess cannot run.
            return ["cmd", "/c", "pnpm"]
        if shutil.which("pnpm"):
            return ["pnpm"]
        # No pnpm installed: the pinned Node distribution that step_node
        # unpacked carries corepack, which fetches the pnpm version the
        # frontend's package.json names. The shims go in a directory on PATH
        # because the frontend's own scripts call `pnpm` again.
        node_bin = self.cache / "node-dist" / "bin"
        shims = self.cache / "corepack-bin"
        shims.mkdir(parents=True, exist_ok=True)
        env["PATH"] = os.pathsep.join([str(shims), str(node_bin), env["PATH"]])
        env["COREPACK_ENABLE_DOWNLOAD_PROMPT"] = "0"
        run(
            [
                str(node_bin / "corepack"),
                "enable",
                "--install-directory",
                str(shims),
                "pnpm",
            ],
            env=env,
        )
        return [str(shims / "pnpm")]

    # --- glue -------------------------------------------------------------

    def step_assets(self) -> None:
        assets = self.out / "assets"
        (assets / "python").mkdir(parents=True, exist_ok=True)
        appliance = PLATFORM / "single-container"
        shutil.copy2(PLATFORM / "db" / "init" / "00-init.sql", assets / "00-init.sql")
        shutil.copy2(appliance / "runtime_config.py", assets / "runtime_config.py")
        shutil.copy2(
            appliance / "python" / "sitecustomize.py",
            assets / "python" / "sitecustomize.py",
        )
        (assets / "frontend-role.sql").write_text(
            frontend_role_sql.extract(appliance / "bootstrap.sh"), encoding="utf-8"
        )
        shutil.copy2(PLATFORM / "LICENSE.md", self.out / "LICENSE.md")

        package = self.out / "autogpt_desktop"
        if package.exists():
            shutil.rmtree(package)
        shutil.copytree(
            DESKTOP / "runtime" / "autogpt_desktop",
            package,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        # -B: the runtime must never write bytecode into the bundle. It is
        # read-only when installed system-wide, and on macOS a file changed
        # inside the app breaks its code signature.
        arguments = ["-B", "-m", "autogpt_desktop", "serve"]
        manifest = {
            "win32": {"command": "python/python.exe", "args": arguments},
            "default": {"command": "python/bin/python3", "args": arguments},
        }
        (self.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # --- size and startup ------------------------------------------------

    def step_prune(self) -> None:
        """Drop what the runtime never loads (slim.py has the rules).
        Erlang/OTP ships every application it has (a GUI toolkit, SNMP,
        CORBA-era protocols); RabbitMQ needs the handful in ERLANG_APPS."""
        for application in (self.out / "erlang" / "lib").iterdir():
            name = application.name.rsplit("-", 1)[0]
            if name not in ERLANG_APPS:
                shutil.rmtree(application)
                continue
            for unused in ("doc", "examples", "src", "c_src", "emacs"):
                shutil.rmtree(application / unused, ignore_errors=True)
        # The engines the runtime uses were copied to prisma/ by step_prisma;
        # the copies npm left inside node_modules are dead weight.
        for engine in (self.out / "prisma" / "node_modules").rglob("*-engine-*"):
            if engine.is_file() and engine.stat().st_size > 1_000_000:
                engine.unlink()
        for cache in (self.out / "python").rglob("__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)
        packages = self.packages()
        python = self.out / "python"
        backend = self.out / "backend"
        removed = {
            # First: what imports a test directory is read from the sources.
            "tests of third-party packages": slim.unused_test_directories(packages),
            "Google API descriptions the backend does not name": (
                slim.unused_discovery_documents(packages, [backend, packages])
            ),
            "typing stubs, C sources and link libraries": [
                *slim.build_only_files(packages),
                *slim.build_only_files(python),
            ],
            "pywin32's editor, examples and help": slim.pywin32_extras(packages),
            "Tcl/Tk, IDLE, pip's installer, console scripts": slim.interpreter_extras(
                python
            ),
            "Erlang's documentation, headers and debug builds": (
                slim.erlang_extras(self.out / "erlang")
            ),
            "RabbitMQ plugins the server does not start": (
                slim.unused_rabbitmq_plugins(self.out / "rabbitmq")
            ),
            "RabbitMQ's other command-line tools": slim.unused_rabbitmq_tools(
                self.out / "rabbitmq"
            ),
            "PostgreSQL's headers and build files": slim.postgres_extras(
                self.out / "postgres"
            ),
        }
        for what, paths in removed.items():
            print(f"  {slim.remove(paths):6} files: {what}")

    def packages(self) -> Path:
        """Where the third-party packages are now: site/ once step_relocate
        has run, the interpreter's site-packages before."""
        relocated = self.out / "site"
        return relocated if relocated.is_dir() else self.site_packages

    def step_relocate(self) -> None:
        """Move third-party packages from python/Lib/site-packages to site/.

        Windows limits a path to 260 characters and the bundle installs under
        the user's profile. Some SDKs generate module names over 150
        characters long; 20 characters saved on every path keeps the longest
        one inside the limit for any Windows user name. A .pth file makes the
        interpreter treat site/ exactly like site-packages (it runs the moved
        packages' own .pth files too, which pywin32 depends on)."""
        target = self.out / "site"
        if target.exists() and self._relocated_already(target):
            print("  the packages are in site/ already")
        else:
            shutil.rmtree(target, ignore_errors=True)
            shutil.move(self.site_packages, target)
            self.site_packages.mkdir()
        (self.site_packages / RELOCATION_FILE).write_text(RELOCATION, encoding="utf-8")

    def _relocated_already(self, relocated: Path) -> bool:
        """Whether this step has been through the bundle before: the file
        it leaves in site-packages is there. Moving site-packages again
        would then put that one file, and whatever was installed beside it
        since, in the place of every package the bundle has."""
        if not (self.site_packages / RELOCATION_FILE).is_file():
            return False  # a fresh install; a site/ beside it is an older one
        added = sorted(
            path.name
            for path in self.site_packages.iterdir()
            if path.name != RELOCATION_FILE
        )
        if added:
            raise RuntimeError(
                f"{relocated} holds the bundle's packages, and {added[:5]} were installed "
                f"into {self.site_packages} afterwards: relocating would replace the first "
                "with the second. Remove them and install through the lock instead "
                "(the deps step, then relocate)."
            )
        return True

    def step_tools(self) -> None:
        """Programs the backend runs by name, in tools/bin (bundled_tools.py)."""
        ffmpeg = bundled_tools.install_ffmpeg(self.out, self.cache)
        print(
            f"  {ffmpeg.relative_to(self.out)}: {bundled_tools.inspect(ffmpeg).version}"
        )

    def step_compile(self) -> None:
        """Ship bytecode. uv installs none, and the installed bundle is
        read-only, so without this every service would recompile every module
        on every start (measured: 51s to ready without, 34s with).
        `unchecked-hash` trusts the .pyc without stat-ing its source, which
        suits files an installer may give fresh timestamps. Forced, and run
        with -B: earlier steps and this interpreter's own startup leave
        ordinary timestamp-checked .pyc files behind, which an installed app
        would find stale and try to rewrite."""
        library = self.python.parent / "Lib" if WINDOWS else self.out / "python" / "lib"
        run(
            [
                str(self.python),
                "-B",
                "-m",
                "compileall",
                "-q",
                "-f",
                "-j",
                "0",
                "--invalidation-mode",
                "unchecked-hash",
                str(library),
                str(self.out / "site"),
                str(self.out / "backend" / "backend"),
                str(self.out / "autogpt_desktop"),
            ],
            check=False,  # a few vendored test fixtures are intentionally invalid
        )

    def step_zip(self) -> None:
        """A few files in place of the pure-Python packages (site_zip.py).
        The standard library stays as files; a bundle that has it zipped,
        from a build before this was known to be wrong, is put back."""
        major, minor = PYTHON_VERSION.split(".")[:2]
        if site_zip.unpack_stdlib(self.out / "python", (int(major), int(minor))):
            print("  the standard library is files again; compiling it")
            self.step_compile()
        before, _ = site_zip.count(self.out)
        if site_zip.plan(self.out / site_zip.PACKAGES).zipped:
            # How every module about to be zipped imports while it is still a
            # file: what step_verify compares the zip with.
            self._gate("--imports", str(self._import_record()), "--record")
        chosen = site_zip.pack(self.out, self.python)
        after, _ = site_zip.count(self.out)
        print(
            f"  {len(chosen.zipped)} packages into {site_zip.ARCHIVES}/; {before} files before, {after} now"
        )
        naming = sorted(
            unit
            for unit, why in chosen.kept.items()
            if why.startswith(site_zip.NAMES_OWN)
        )
        if naming:
            print(f"  kept as files because they name their own: {', '.join(naming)}")

    def step_verify(self) -> None:
        """The bundle's own interpreter imports the whole backend, loads
        every block, and reads what the zipped packages keep as data
        (bundle_gate.py). What pruning or zipping broke shows here, not on a
        user's machine."""
        self._gate("--imports", str(self._import_record()))

    def _import_record(self) -> Path:
        return self.cache / f"zip-imports-{platform_key()}.json"

    def _gate(self, *arguments: str) -> None:
        gate = Path(__file__).with_name("bundle_gate.py")
        command = [str(self.python), "-B", str(gate), str(self.out), *arguments]
        print(f"  $ {' '.join(command)}", flush=True)
        if subprocess.run(command, stdin=subprocess.DEVNULL).returncode != 0:
            raise RuntimeError("the bundle did not pass its gates; see above")

    # --- last ------------------------------------------------------------

    def step_seal(self) -> None:
        """Leave the bundle as an installed app must find it, and refuse to
        call it assembled otherwise. A bundle that has been run from by an
        older build of the runtime has what that run wrote into it; a bundle
        that would write into itself again must not be packaged."""
        shutil.rmtree(self.out / "frontend" / ".next" / "cache", ignore_errors=True)
        modules = self.out / "prisma" / "node_modules"
        for stray in downloaded_engines(modules):
            stray.unlink()
        shutil.rmtree(
            modules / "@prisma" / "engines" / "node_modules" / ".cache",
            ignore_errors=True,
        )
        next_config.check(self.out / "frontend")
        unread = unread_by_prisma_cli(self.out / "prisma" / "node_modules")
        if unread:
            raise RuntimeError(
                f"the bundled Prisma CLI no longer reads {', '.join(unread)}, which is how "
                "the runtime keeps it from downloading engines on the user's machine. Read "
                "how this version finds its engines (@prisma/engines ensureBinariesExist, "
                "@prisma/fetch-engine download) and update "
                "desktop/runtime/autogpt_desktop/migrations.py."
            )
        for kind in ("query-engine", "schema-engine"):
            engine = self.out / "prisma" / f"{kind}{EXE}"
            if not engine.is_file():
                raise RuntimeError(
                    f"the bundle has no prisma/{kind}{EXE}; run the prisma step"
                )
            if sys.platform.startswith("linux"):
                problem = wrong_openssl(elf.needed(engine))
                if problem:
                    raise RuntimeError(
                        f"prisma/{kind} {problem}; run the prisma step again"
                    )
        bundled_tools.check(self.out)
        backend_patches.check(self.out / "backend")
        if [path.name for path in self.site_packages.iterdir()] != [RELOCATION_FILE]:
            raise RuntimeError(
                f"third-party packages are in {self.site_packages}, not in site/; "
                "run the deps step and then the relocate step"
            )
        claude_cli.check(
            self.out / "site" / "claude_agent_sdk" / "_bundled",
            LOCK,
            self.cache,
            CLAUDE_CLI_VERSION,
        )
        left = [str(path.relative_to(self.out)) for path in never_shipped(self.out)]
        if left:
            raise RuntimeError(
                f"the bundle holds what is never shipped: {', '.join(left[:5])}. An older "
                "build left it; run the prune step, or build into a fresh --out."
            )
        site_zip.check_bytecode(self.out)
        zipped = site_zip.zipped_stdlib(self.out / "python")
        if zipped:
            raise RuntimeError(
                f"{zipped[0].relative_to(self.out)}: the standard library must not be zipped. With "
                "it the interpreter also takes the search path of a Python installed on the "
                "machine from the Windows registry (build/site_zip.py). Run "
                "`--only compile,zip,verify,seal`: the zip step puts it back as files."
            )
        over = over_budget(self.out)
        if over:
            raise RuntimeError(
                "the bundle is over its budget: "
                + "; ".join(over)
                + ". What grew: read the "
                "table above beside the same table in the log of the last build that passed."
            )


def never_shipped(out: Path) -> list[Path]:
    """What a run of the app, or a build older than the step that removes
    it, leaves in a bundle. electron-builder packs the directory as it is."""
    engines = out / "prisma" / "node_modules" / "@prisma" / "engines"
    candidates = [
        out / "frontend" / ".next" / "cache",
        out / "erlang" / "doc",
        out / "postgres" / "include",
        engines / "node_modules" / ".cache",
        *(out / "prisma").rglob("*.dll.node"),
        *(out / "prisma").rglob("*.so.node"),
        *(out / "prisma").rglob("*.dylib.node"),
    ]
    return [path for path in candidates if path.exists()]


def over_budget(out: Path) -> list[str]:
    """Which of the bundle's budgets it breaks, in words. The three numbers
    are what installing costs: the Windows installer writes every file twice
    and has it scanned, a path past 260 characters cannot be installed at
    all, and every byte is downloaded again by an update."""
    files, size = site_zip.count(out)
    length, longest = site_zip.longest_path(out)
    found = []
    if files > MAX_FILES:
        found.append(f"{files} files, and the budget is {MAX_FILES}")
    if WINDOWS and length > MAX_RELATIVE_PATH:
        found.append(
            f"{longest} is {length} characters long, and the budget is {MAX_RELATIVE_PATH}"
        )
    if size > MAX_BYTES:
        found.append(f"{size / 1e6:.0f} MB, and the budget is {MAX_BYTES / 1e6:.0f} MB")
    print(f"  {files} files, {size / 1e6:.0f} MB, longest path {length}")
    # On every build, not only one that fails: what grew is found by reading
    # this beside the same lines of the last build that passed.
    print("\n".join(where_it_is(out)))
    return found


def where_it_is(out: Path, largest: int = 10) -> list[str]:
    """The bundle's files and megabytes by directory: each directory of the
    bundle, then the largest of what site/ holds (the packages that are not
    zipped), which is where a dependency that grew or was added shows."""
    lines = [_tally_line(f"{path.name}/", path) for path in _directories(out)]
    packages = sorted(
        _directories(out / site_zip.PACKAGES),
        key=lambda path: site_zip.count(path)[0],
        reverse=True,
    )
    shown = [
        _tally_line(f"{site_zip.PACKAGES}/{path.name}/", path)
        for path in packages[:largest]
    ]
    return [
        *lines,
        f"  the {len(shown)} directories of {site_zip.PACKAGES}/ with most files:",
        *shown,
    ]


def _directories(parent: Path) -> list[Path]:
    return (
        sorted(path for path in parent.iterdir() if path.is_dir())
        if parent.is_dir()
        else []
    )


def _tally_line(label: str, directory: Path) -> str:
    files, size = site_zip.count(directory)
    return f"    {label:<34} {files:>6} files {size / 1e6:>7.0f} MB"


def fetched_engine(engines: Path, kind: str, platform: str) -> Path:
    """The executable engine the Prisma CLI downloaded for this platform
    (not the .node library builds): query-engine-windows.exe,
    query-engine-darwin-arm64, and on Linux the one named for OpenSSL 3 and
    no other, even when the download cache holds others."""
    target = LINUX_ENGINE_TARGETS.get(platform)
    if not target:
        return _single(
            path
            for path in engines.glob(f"{kind}-*")
            if not path.name.endswith((".node", ".gz", ".sha256", ".tmp"))
        )
    engine = engines / f"{kind}-{target}"
    if not engine.is_file():
        raise RuntimeError(
            f"the Prisma CLI did not fetch {engine.name} into {engines}. It is told which "
            "to fetch through PRISMA_CLI_BINARY_TARGETS; if this version of the CLI no "
            "longer reads that, update step_prisma."
        )
    return engine


def wrong_openssl(needed: list[str]) -> str | None:
    """What is wrong with the libraries a Linux Prisma engine is linked
    against, if it is not OpenSSL 3 alone."""
    linked = sorted(name for name in needed if name.startswith("libssl.so"))
    if linked == [OPENSSL]:
        return None
    found = ", ".join(linked) or "no libssl at all"
    return f"is linked against {found}, and the bundle is for systems with {OPENSSL}"


def downloaded_engines(node_modules: Path) -> list[Path]:
    """Engines the Prisma CLI fetched into its own package while it ran (the
    build copies none there: step_prisma, step_prune), and its download
    cache."""
    engines = node_modules / "@prisma" / "engines"
    patterns = ("*.node", "*-engine-*", "*.gz", "*.sha256", "*.tmp")
    return sorted(
        path for pattern in patterns for path in engines.glob(pattern) if path.is_file()
    )


def unread_by_prisma_cli(node_modules: Path) -> list[str]:
    """Of the variables the runtime steers the Prisma CLI with, those the
    CLI's engine handling no longer mentions."""
    sys.path.insert(0, str(DESKTOP / "runtime"))
    from autogpt_desktop.migrations import READ_BY_THE_CLI

    code = "".join(
        path.read_text(encoding="utf-8", errors="replace")
        for package in ("engines", "fetch-engine")
        for path in sorted((node_modules / "@prisma" / package / "dist").rglob("*.js"))
    )
    return [name for name in READ_BY_THE_CLI if name not in code]


# The budgets step_seal holds the bundle to (`over_budget`).
#
# A path: Windows refuses one over 260 characters. The installer unpacks
# into %TEMP%\ns<5>.tmp\7z-out\resources\runtime\ before it copies, which
# for a user name of 20 characters is 73 characters, and installs into
# %LOCALAPPDATA%\Programs\<name>\resources\runtime\, which is 67 for the
# normal app. 130 inside the bundle leaves room for a user name of some 70
# characters, or a long variant name.
MAX_RELATIVE_PATH = 130
# Measured on Windows x64 (README.md, "Size"). The macOS and Linux bundles
# keep the standard library as files, and have not been measured since the
# pruning. Before it CI printed 2.2 GiB for Linux and 1.9 GiB for macOS and
# Windows (`du -sh`; run 37154118481), and Windows lost 450 MB of its 1.96 GB,
# part of it files only Windows has: Linux would be about 1.9 to 2.0 GB.
# Their budgets leave room over that until a build has printed their numbers
# (the `[seal]` line), and are then to be set from them.
# Windows: 23,616 with the standard library as files (1,087 of them).
MAX_FILES = 24_500 if WINDOWS else 30_000
MAX_BYTES = 1_600_000_000 if WINDOWS else 2_300_000_000

# OTP applications RabbitMQ and its Elixir-based CLI load.
ERLANG_APPS = {
    "asn1",
    "compiler",
    "crypto",
    "eldap",
    "erts",
    "inets",
    "kernel",
    "mnesia",
    "os_mon",
    "public_key",
    "runtime_tools",
    "sasl",
    "ssl",
    "stdlib",
    "syntax_tools",
    "tools",
    "xmerl",
}

STEPS = (
    "python",
    "node",
    "deps",
    "backend",
    "prisma",
    "postgres",
    "valkey",
    "erlang",
    "rabbitmq",
    "erlang_patches",
    "frontend",
    "frontend_config",
    "assets",
    "prune",
    "relocate",
    "tools",
    "compile",
    "zip",
    "verify",
    "seal",
)


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> None:
    print(f"  $ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=check)


def _single(paths) -> Path:
    found = list(paths)
    if len(found) != 1:
        raise RuntimeError(f"expected exactly one match, found {found}")
    return found[0]


def _add_msvc_runtime(erlang: Path) -> None:
    """Erlang's Windows binaries link the MSVC runtime dynamically and ship
    an installer for it instead of the DLLs. Put the DLLs beside the
    binaries so a machine without the redistributable can still run them."""
    system32 = Path(os.environ["SYSTEMROOT"]) / "System32"
    for bin_dir in (erlang / "bin", *erlang.glob("erts-*/bin")):
        for name in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"):
            shutil.copy2(system32 / name, bin_dir / name)


def _install_erlang(erlang: Path) -> None:
    """The hex.pm Linux build is an uninstalled release: `Install` creates
    bin/ and the boot scripts. It records the build path, but the launcher
    it writes asks `dyn_erl --realpath` where it really is on every start
    and prefers that, so the tree can be moved afterwards."""
    run(["./Install", "-sasl", str(erlang)], cwd=erlang)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DESKTOP / "build" / "runtime")
    parser.add_argument("--cache", type=Path, default=DESKTOP / "build" / ".cache")
    parser.add_argument("--only", default="")
    parser.add_argument("--skip", default="")
    parser.add_argument(
        "--redis-stand-in",
        action="store_true",
        help="Windows, local builds only: bundle Redis when no Valkey build is in the cache",
    )
    args = parser.parse_args()
    only = [name for name in args.only.split(",") if name]
    skip = {name for name in args.skip.split(",") if name}
    unknown = (set(only) | skip) - set(STEPS)
    if unknown:
        parser.error(f"unknown steps: {sorted(unknown)}")

    build = Build(args.out.resolve(), args.cache.resolve(), args.redis_stand_in)
    build.out.mkdir(parents=True, exist_ok=True)
    for name in only or STEPS:
        if name in skip:
            continue
        print(f"[{name}]", flush=True)
        getattr(build, f"step_{name}")()
    print(f"runtime assembled for {platform_key()} at {build.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
