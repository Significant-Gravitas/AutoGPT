# Capy Reviews
<!-- MANUAL: file_description -->
Run Capy's pull request reviewer on a GitHub PR and read the findings it produced.
<!-- END MANUAL -->

## Capy Get Review Round

### What it is
Gets a Capy review round's status and its findings, each with severity, confidence, category, file and line.

### How it works
<!-- MANUAL: how_it_works -->
Calls `GET /api/v1/reviews/rounds/{requestId}` and returns the round's status and findings. `is_settled` is true once the round has completed, failed or gone stale; `high_severity_count` counts high-severity issues, not notes.

Only a `completed` round has reviewed the code. A round that failed or went stale settles with no findings, so treat it as unreviewed, not clean.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| request_id | The request_id Capy Start Review returned | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| round | The review round | ReviewRound |
| status | pending, running, completed, failed or stale | str |
| is_settled | True once the round has completed, failed or gone stale | bool |
| findings | Findings with severity, confidence, category, file and line | List[ReviewFinding] |
| high_severity_count | Number of high-severity issues. Zero means a clean review only when status is completed; a failed or stale round reviewed nothing. | int |

### Possible use case
<!-- MANUAL: use_case -->
**Merge Blocking**: Allow a merge only when `status` is `completed` and `high_severity_count` is zero, so a failed or stale round never passes as clean.

**Findings Report**: Send the confirmed issues, with file and line, to the author.

**Settle Polling**: Poll until `is_settled` is true before acting on the review.
<!-- END MANUAL -->

---

## Capy Start Review

### What it is
Starts a Capy code review on a GitHub pull request. Capy's review agent reads the diff in a real checkout and posts findings as inline comments on the pull request. Bills the review to your Capy organization.

### How it works
<!-- MANUAL: how_it_works -->
Calls `POST /api/v1/reviews` for a repository and PR number that the organization's Capy GitHub installation covers. A round is keyed to the PR's exact head and base commits, so asking again for the same code returns the existing round with `adopted` true; `force_refresh` re-runs a completed round. Findings at or above the repository's threshold post to the pull request as inline comments, which is why the block pauses for approval when the run requires it.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo | The repository as owner/name | str | Yes |
| pr_number | The pull request number | int | Yes |
| tier | Review depth and cost: low, medium or high. Empty uses the repository's default. | "" \| "low" \| "medium" \| "high" | No |
| idempotency_key | Retry key: the same key returns the same round instead of starting another. Also the request_id you read the round back with. Empty lets Capy key the round on the PR's exact commits. | str | No |
| source_thread_id | A Capy thread that should receive the verdict and triage the findings itself | str | No |
| force_refresh | Re-run a round that already completed on the same commits | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| review | The started (or adopted) round | ReviewStarted |
| request_id | Pass to Capy Get Review Round to read the findings | str |
| adopted | True when a round already existed for these exact commits and nothing new was started | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Release Gate**: Review the release pull request before it merges.

**Agent Self-Review**: Pass `source_thread_id` so the thread that opened the PR triages the findings itself.

**Deep Review on Demand**: Run a `high` tier review on a risky change.
<!-- END MANUAL -->

---
