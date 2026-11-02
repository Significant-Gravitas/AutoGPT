# Backend CI shard benchmark

The backend workflow supports `paths` (the existing layout) and `weighted` (an experiment). Automatic PR and merge-group runs continue to use `paths` until a complete hosted comparison validates the weighted layout.

Both layouts execute the full suite on Python 3.12 and 3.13 with eight isolated runners per version. The data shard stays intact because its tests restart Redis and use the guarded `postgres` database. Weighted mode assigns the remaining files longest-first to the least-loaded worker. The other seven job names remain stable for check and report compatibility; in weighted mode they identify workers rather than directories.

## Selection and evidence

`.github/scripts/backend-test-durations.json` seeds weights from the passing run at `508ec74559`, using the median per-file JUnit duration across both Python versions. Those initial values include setup, test calls, and teardown, but not collection. New files receive a fallback weight. Missing weights never exclude tests.

Pytest keeps authority over collection rules. The plugin filters file ownership before import and preserves normal ignore hooks; unexpected collectors fall back to the remainder worker. Tests under `backend/data` always stay on the data worker. Every worker computes the same deterministic assignment from the same files and weights. A subprocess test exercises all eight workers, custom filename patterns, ignored files, new files, and the CI-helper directory, and checks every case appears exactly once.

Each test artifact includes JUnit, coverage, and `shard-metrics-<worker>.json`. Metrics record:

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

Require all sixteen jobs and report validation to pass in both runs. Verify the recorded source SHA and runtime/runner image match, and compare collected inventories, including parameterized cases, with no missing or duplicate tests. Record any nondeterministic parameter IDs explicitly rather than dropping all parameter IDs from the comparison.

Compare the slowest test step and job in each Python leg, total runner time, setup/finalization, and run wall time including scheduling delay. Inspect CPU model differences and repeat the pair if the result is close or inconsistent. A passing single pair is an initial measurement, not a p95 guarantee. A backend improvement alone does not establish end-to-end merge-queue latency.

## Refresh weights

After downloading one complete passing run's artifacts into a directory:

```sh
python .github/scripts/update_backend_test_durations.py <reports-directory>
```

The updater includes measured module collection costs and refuses failed, duplicate, mixed-revision, mixed-layout, or incomplete shard sets. Review the generated file, run the helper tests, and commit the calibration. Do not mix reports from multiple runs in one directory.

This experiment does not cache test results, disable test isolation, change Python support, or increase runner count. Setup caching is a separate experiment so its effect can be measured independently.
