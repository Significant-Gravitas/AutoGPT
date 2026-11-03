# Backend CI shard benchmark

The backend workflow supports `paths` (the existing layout) and `weighted` (an experiment). Automatic PR and merge-group runs continue to use `paths` until a complete hosted comparison validates the weighted layout.

Both layouts execute the full suite on Python 3.12 and 3.13. Automatic path mode retains eight fixture groups, then balances whole files within each group across four isolated runners; billing-threads has three files and uses three runners. This makes 31 partitions per Python version, 62 jobs total. Every partition has its own services. Data partitions retain the guarded `postgres` database and Redis restart permission. Weighted mode remains an experiment that redistributes files across groups before partitioning; it has not passed full runtime validation.

## Selection and evidence

`.github/scripts/backend-test-durations.json` uses per-file collection, setup, call and teardown timings from Python 3.12 in passing run `41578935067` on PR head `797627a`. Its metadata records the checked-out merge commit. New files receive a fallback weight. Missing weights never exclude tests.

Pytest keeps authority over filename patterns and ignore hooks. The plugin filters partition ownership before import and deselects explicit file arguments belonging to another partition. A collected file missing from the assignment fails the run. Every partition computes the same deterministic assignment from the same files and measured weights. A real pytest subprocess test runs all 31 partitions with nested groups, explicit files, custom filename patterns, ignored files, new files and CI helpers, and requires every parameterized case exactly once. The aggregate report gate requires all 62 artifacts and rejects duplicate JUnit case identities as well as missing reports, failures and unapproved skips.

Each test artifact includes JUnit, coverage, and `shard-metrics-<group>-<partition>.json`. Metrics record:

- The checked-out Git commit, Python version, CPU model/count, and runner image.
- The collected node IDs and per-file case counts.
- Per-module collection duration, including module import, and setup/call/teardown duration.
- Pytest configuration-to-session-finish time and exit status.

Use the Actions job/step timestamps for total test-step and job duration; plugin timing starts after initial pytest configuration imports. Do not treat the sum of individual test durations as the whole CI time.

## Run the comparison

Dispatch both modes against one unchanged branch revision:

```sh
gh workflow run platform-backend-ci.yml --ref <branch> -f test_layout=paths -f benchmark=true
gh workflow run platform-backend-ci.yml --ref <branch> -f test_layout=weighted -f benchmark=true
```

Both dispatches use the same source, dependency setup, service images, runner class, coverage instrumentation, and Python matrix. `benchmark=true` keeps coverage artifacts and validation while suppressing duplicate Codecov uploads. It does not reduce the test matrix or disable the skip-policy gate.

Require all 62 jobs and report validation to pass in both runs. Verify the recorded source SHA and runtime/runner image match, and compare collected inventories, including parameterized cases, with no missing or duplicate tests. Record any nondeterministic parameter IDs explicitly rather than dropping all parameter IDs from the comparison.

Compare the slowest test step and job in each Python leg, total runner time, setup/finalization, and run wall time including scheduling delay. Inspect CPU model differences and repeat the pair if the result is close or inconsistent. A passing single pair is an initial measurement, not a p95 guarantee. A backend improvement alone does not establish end-to-end merge-queue latency.

## Refresh weights

After downloading one complete passing run's artifacts into a directory:

```sh
python .github/scripts/update_backend_test_durations.py <reports-directory>
```

The updater includes measured module collection costs and refuses failed, duplicate, mixed-revision, mixed-layout, or incomplete shard sets. Review the generated file, run the helper tests, and commit the calibration. Do not mix reports from multiple runs in one directory.

Partitioning increases the number of standard GitHub runners. It does not cache test results, disable test isolation or change Python support. Measure scheduling delay and total runner time as well as job duration: added parallelism can move the bottleneck to runner availability. Setup caching is measured separately.

## Rust tooling comparison

`backend-tool-benchmark.yml` measures installation, formatting, import sorting, and type checking on hosted Python 3.12 and 3.13. It runs on manual dispatch, so unchanged experiments do not compete with required checks on every PR update. It does not replace the normal CI checks or change the shard layout.

The paired measurements use the same checkout and runner:

- Poetry installs the existing lock into `.venv`. The official Poetry export plugin exports that lock for `uv pip sync` into a second environment, including the editable shared library and pinned Git dependency. Poetry then registers only the root package in that environment, after checking the environment path: the package's sdist-only declaration cannot be installed with uv's editable-wheel path. Include this registration in installation timing. The benchmark compares package versions, source URLs, commit IDs, and editable status; only the virtualenv bootstrap package `pip` is excluded. It records dependency checks and runs the shard-selection helper tests under both interpreters. Runtime compatibility still requires the full backend suite.
- Black versus Ruff formatting, and isort versus Ruff import sorting, receive the same tracked Python file list, excluding vendored test fixtures. Output is retained; formatter differences require a deliberate migration, not suppression of failing checks.
- Pyright versus ty use the same installed dependencies, generated Prisma stubs, Python version, and backend source root. Existing Pyright ignore paths become ty diagnostic overrides. Their rule sets differ, so timing alone does not establish equivalent checking.

Artifacts retain every command, duration, exit status, source revision, package inventory, file list, and diagnostic log. The job succeeds when it collects the evidence; candidate failures are recorded and are not approvals to switch tools. The export tool uses a separate download cache; include export time and the bootstrap Actions steps when estimating installation savings. The warm install sample reuses an installed environment. Repeat on fresh runners before treating a small difference as a win.

Candidates checked against upstream documentation:

- [uv's locked-environment installation](https://docs.astral.sh/uv/pip/compile/) preserves the Poetry lock via the [official export plugin](https://github.com/python-poetry/poetry-plugin-export); this is not a project-wide lockfile migration.
- [Ruff documents formatting differences from Black](https://docs.astral.sh/ruff/formatter/black/). Ruff already checks backend code and formats the shared library; the remaining replacement opportunity is backend Black/isort.
- [ty](https://docs.astral.sh/ty/coming-from-mypy-or-pyright/) and [Pyrefly](https://pyrefly.org/en/docs/migrating-from-pyright/) are Rust type-checker alternatives. The first hosted probe uses ty; neither has been accepted as a replacement for Pyright.
- [Oxlint](https://oxc.rs/docs/guide/usage/linter/migrate-from-eslint.html) already handles the frontend's design-system rules. Replacing its remaining ESLint checks requires preserving the React Compiler, keyboard/IME, Storybook, and TanStack rules.
- [pytest-rs](https://github.com/Yasu-umi/pytest-rs) is alpha and requires CPython 3.13+. Its distribution conflicts with the installed pytest package and bundled plugin packages. It cannot replace the current 3.12–3.13 matrix without a separate migration and plugin/coverage conformance proof. [Karva](https://github.com/MatthewMckee4/karva) explicitly has a narrower scope than pytest.
- [ast-grep](https://ast-grep.github.io/) can implement structural architecture checks in Rust, but the existing Python AST rules contain alias handling and exceptions that require matched fixtures before a rewrite can replace them.
