# CI throughput work

Keep GitHub-hosted runners. The [Boardy CI article](https://x.com/owengretzinger/status/2116291924219060319) is a source of experiments, not an AutoGPT benchmark. Its paid runners and larger machines are outside this work's scope. No tests are deleted to meet a duration target.

## Affected platform projects

`platform-ci.yml` selects backend, frontend and full-stack suites for PR and merge-group changes. The existing suite workflows remain independently dispatchable and run in full every night on the repository's default branch, currently `master`. Their existing `dev`/`master` push triggers also remain.

The selector uses `.github/ci-projects.json`. A frontend change runs frontend and full-stack checks; a backend or shared-library change runs backend and full-stack checks. Prisma schema, backend container, integration catalog and compose changes preserve their cross-project dependencies. Unknown platform directories, CI configuration and other unclassified files select every suite. Documentation, classic and issue-template changes do not select platform suites; their own workflows remain responsible for them.

PR selection compares the merge base with the PR head. Merge groups compare their base and head commits. Checkouts that read history use a full, blobless fetch. Deleted paths and both sides of renames are included. Missing revisions and unsupported event types fail instead of returning an empty plan.

The safety check separately reads Git's diff and applies a conservative policy that does not consult the selector's graph or file list. The final job repeats this check and requires every selected suite to succeed. A missing, cancelled or skipped selected suite fails. The CLI tests deliberately remove a backend change from the graph, produce an empty selection, supply successful selection plus skipped suite results, and require rejection. They also cover renames, deletions, base-only changes, schema dependencies, failed selection and nightly/full dispatches.

This is project selection, not individual test selection. The full backend matrix and all four frontend shards still execute whenever their project is affected. The existing advisory journeys policy is preserved; smoke, API types and all required tests still block.

## Hosted frontend comparison

`frontend-tool-benchmark.yml` runs when its benchmark files change in a PR, or by manual dispatch. It uses the exact PR head, the existing pnpm lock, Node 24.22.1 and standard GitHub-hosted runners. Candidate tools live in a temporary directory, without changing the project lock or installed TypeScript used by Next.js.

- Compare existing Next/ESLint plus custom Oxlint checks with Oxlint 1.90.0 and a migrated configuration. Preserve the current custom plugins and overrides in the migration input. Retain all warnings about unsupported rules. Inject invalid keyboard handling, conditional hooks and props mutation in a compiler-enabled directory, and require both linters to report the intended rule before considering a replacement. Passing these probes alone does not prove every rule has equivalent semantics.
- Compare TypeScript 5.9 with TypeScript 7.0.2 and Bun 1.4.4. Extend the existing configuration with the original implicit type packages and side-effect-import default, and normalize relative paths for the newer compilers. Retain the effective configuration and all diagnostics. Every checker must detect an injected assignment error in a project source file. Keep the existing compiler and Next.js API-schema check authoritative until migration diagnostics are resolved.
- Compare Prettier with Oxfmt 0.75.0, using its configuration migrator, including the Tailwind plugin settings. Retain migration warnings and formatting differences; never rewrite application files during the benchmark.
- Run each Vitest shard twice on one machine, with isolation enabled and disabled. Keep the forks pool, CPU-based worker limit, coverage and test configuration identical; disable the result cache and alternate ordering across shards. Retain JUnit and Cobertura for both layouts, and require identical passing test inventories. No coverage is sent to Codecov from diagnostic runs. Shared workers are not adopted until repeated and reordered runs validate state cleanup.
- Time installation from the existing pnpm store cache, then separately time restoring installed `node_modules`. Key the experiment by OS, architecture, runner image family, exact Node version and dependency configuration. Trusted `dev` and `master` pushes seed this cache when dependency inputs change. Production jobs continue using their existing dependency setup until restore benefit and cache scope are demonstrated.

Artifacts record commands, exit statuses, durations, lock/source hashes, runner hardware, negative probes and compatibility logs. A diagnostic workflow succeeding means its measurements were collected, not that a candidate passed. Compare whole-job and queue durations from Actions as well as command times. Repeat close results before adoption, and distinguish a faster tool from a faster PR critical path.

References: [Oxlint migration](https://oxc.rs/docs/guide/usage/linter/migrate-from-eslint.html), [Bun type checking](https://bun.sh/docs/runtime/check), [Vitest isolation](https://vitest.dev/guide/improving-performance.html), [Oxfmt migration](https://oxc.rs/docs/guide/usage/formatter/migrate-from-prettier.html), [GitHub cache scopes](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching). Backend comparisons are documented in [backend-ci-sharding.md](backend-ci-sharding.md).
