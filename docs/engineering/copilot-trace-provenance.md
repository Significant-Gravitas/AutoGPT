# Copilot trace provenance

The backend uses `SENTRY_RELEASE` as its deployed release identifier. Supply the
immutable source commit or release tag at build or deploy time; the backend
Dockerfile accepts `--build-arg SENTRY_RELEASE=<commit>`, and a deployment can
override it with the runtime environment variable. Set it on every backend
service, including copilot workers. No additional Langfuse variable is required.

Sentry already reports this value in its native `release` field. Copilot now
also propagates the same trimmed value as Langfuse `metadata.release` on both
SDK and baseline turns and on separate title generations. Existing OpenRouter
Broadcast attribution uses `revision` for the same value. This change does not
rename that field or alter Langfuse's independently configured native release
field. Missing or whitespace-only configuration omits the stamp; it does not
guess from a local checkout or report the package version as a deployed build.

Main-turn metadata also includes `base_prompt_sha256`: the SHA-256 of the exact
UTF-8 base prompt returned by `_build_system_prompt`. That is the compiled
Langfuse prompt actually served (including a cached copy on refresh failure),
or the static fallback. It excludes runtime supplements, expert identity,
conversation content, and the Claude Code preset. Use it to distinguish base
prompt revisions within a release, not as a fingerprint of every model input.
Title generations carry the release only.

For error-discovery calibration, filter by `metadata.release` and, when the
base prompt matters, `metadata.base_prompt_sha256`. A follow-up turn in a
long-lived chat gets the release of the worker that executes that turn; the
release is not saved on the chat. Existing traces are not backfilled.

After deployment, send a fresh SDK turn, a baseline turn, and a first message
that triggers title generation. Confirm their Langfuse metadata names the
deployed commit, and compare it to a Sentry event's native release from the
same backend build. Change a test Langfuse base prompt and confirm that a new
turn has a different hash with the same release. Local regression tests cover
the turn wiring and capture real OTLP exports, including nested spans, error
exits, blank configuration, and context cleanup; they do not establish that a
particular production deployment has supplied the variable or ingested events.
