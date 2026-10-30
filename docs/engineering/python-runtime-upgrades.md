# Backend Python upgrade plan

## Immediate CI change

The backend supports Python 3.12 and 3.13. Python 3.11 is retired from the
test/type-check matrices, skip policy, package metadata, and backend setup and
deployment jobs. Containers already use 3.13. The standalone shared library
retains its independent Python requirement.

Run dependency resolution, integration tests, and benchmarks on hosted CI.
Keep local work to source edits and lightweight static checks: the October 30
benchmark setup exhausted the Windows host disk.

The [October 30 benchmark](https://github.com/Significant-Gravitas/AutoGPT/actions/runs/41072022879)
passed the complete original four-group suite on 3.11, 3.12, and 3.13. Its
slowest jobs were 27m01s, 17m08s, and 19m28s respectively. These are single
runs on different hosted CPUs, not a controlled interpreter speed comparison.
The eight-group layout must pass on both retained versions before merging;
measure its critical path separately from runner wait and merge-queue time.

## Python 3.14

Follow the Python project's [3.14 porting guide](https://docs.python.org/3.14/whatsnew/3.14.html#porting-to-python-3-14)
and [typing changes](https://docs.python.org/3.14/whatsnew/3.14.html#typing).

- Update the OpenAI dependency through its upstream migration guidance. The
  benchmark's pinned 1.109.1 writes `__discriminator__` onto a union; 3.14
  changed unions and forbids that mutation. Re-run the Codex usage and billing
  tests against the real SDK parsing path.
- Resolve native wheels and the protobuf/grpcio-status constraints together.
  The experimental lock used conditional versions; it is not the production
  lock and does not establish compatibility of the complete dependency set.
- Audit annotation introspection, Pydantic schemas, and union handling. Use
  documented annotation APIs rather than private typing implementation details.
- Validate process startup, pickling, readiness, and shutdown. Python changed
  the Unix default to forkserver; `backend/util/process.py` already explicitly
  selects forkserver or spawn. The benchmark's readiness and child-import
  failures still need reproduction; the default change alone is not a diagnosis.
- Require all shards, unchanged skip validation, and complete coverage. The
  3.14 copilot timeout is unresolved; timing an incomplete run is not a win.

## Python 3.15

Follow the Python project's [3.15 porting guide](https://docs.python.org/3.15/whatsnew/3.15.html#porting-to-python-3-15)
and [UTF-8 default change](https://docs.python.org/3.15/whatsnew/3.15.html#other-language-changes).

- Resolve the `youtube-transcript-api` dependency first: the benchmark's pinned
  1.2.4 declares Python `<3.15`. Do not override upstream compatibility metadata.
- Audit text-file encodings on Linux and Windows, SQLite positional arguments,
  resource-limit sentinels, and base64 validation. In particular, check token
  and key readers in `backend/data/user.py`, `backend/util/service_auth.py`, and
  `backend/integrations/codex/auth_bundle.py` against the changed padding rules.
- Repeat the 3.14 checks and full hosted suite on 3.15. Keep ordinary CPython
  builds as the baseline; free-threading, JIT, and lazy imports are separate
  experiments.

## Adoption gate

The intended support range is 3.13–3.15 after both new versions pass. Change
the Python range and regenerated lock, CI matrices, skip policy, coverage
count, deployment/setup jobs, lint target, and contributor instructions in
the same support change. Validate container startup and representative
workflow execution before changing production images. Compare repeated
matched-source runs and full test inventories before claiming a speedup.
