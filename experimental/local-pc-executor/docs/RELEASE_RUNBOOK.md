# Release Runbook

> One-time setup + per-release steps for publishing
> `autogpt-local-executor` to PyPI, Homebrew, and Scoop.
>
> All of the wire-up (`release.yml`, `scripts/update_packaging.sh`,
> formula and manifest stubs) is already in the repo. This doc is the
> "who does what" so the first publisher doesn't have to reverse-
> engineer it.

---

## First-time setup (one-time, ~30 min)

### 1. PyPI trusted publisher

Trusted-publisher OIDC means the release workflow can publish without
a long-lived `PYPI_API_TOKEN` secret in the repo. Set it up once:

1. Sign in to https://pypi.org with an account that owns (or will
   own) the `autogpt-local-executor` package name.
2. https://pypi.org/manage/account/publishing/ → **Add a new pending
   publisher**.
3. Fill in:
   - **PyPI project name:** `autogpt-local-executor`
   - **Owner:** `Significant-Gravitas`
   - **Repository name:** `autogpt-local-executor`
   - **Workflow filename:** `release.yml`
   - **Environment name:** `pypi`
4. Save. The "pending" state resolves automatically on first publish.
5. In the GitHub repo settings → Environments → create a `pypi`
   environment. No secrets needed, but enable "Required reviewers" if
   you want a human-in-the-loop on every publish (recommended for the
   first few releases — disable later if cadence picks up).

### 2. Homebrew tap

Operator decision: which repo holds the tap?

- Recommended: `Significant-Gravitas/homebrew-tap` (one tap for all
  Significant-Gravitas formulae; `autogpt-local-executor` lives at
  `Formula/autogpt-local-executor.rb`).
- Alternative: dedicated tap repo `Significant-Gravitas/homebrew-autogpt-local-executor`.

Create the tap repo (whichever you pick), drop in an empty
`Formula/` directory, commit, push. End-user install will be
`brew install Significant-Gravitas/tap/autogpt-local-executor` after
the first publish.

### 3. Scoop bucket

Same decision shape:

- Recommended: `Significant-Gravitas/scoop-bucket`
  (`bucket/autogpt-local-executor.json`).
- Alternative: dedicated bucket.

End-user install: `scoop bucket add autogpt https://github.com/Significant-Gravitas/scoop-bucket` then `scoop install autogpt/autogpt-local-executor`.

---

## Per-release steps

### Pre-flight

- All CI green on the release commit (`.github/workflows/ci.yml`
  matrix passes on mac/Linux/Windows × py3.11/py3.12).
- `pyproject.toml` `version` field bumped to the target tag — the
  release workflow's `verify pyproject version matches tag` step
  will fail loudly if not.
- `CHANGELOG.md` (if/when added) updated.

### 1. Tag + push

```bash
git checkout experimental/local-pc-executor  # or whatever the release branch is
git pull
git tag v0.0.1
git push origin v0.0.1
```

`release.yml` triggers automatically on the tag push. Watch
`https://github.com/Significant-Gravitas/autogpt-local-executor/actions`.

### 2. Verify PyPI publish

The `release.yml` workflow does three jobs sequentially:

- `build` — builds sdist + wheel, runs `twine check`. Should pass
  cleanly; pre-flight catches the common breakage.
- `publish-pypi` — uploads via trusted-publisher OIDC. If the "pending
  publisher" was set up correctly, the first run resolves it to a
  real publisher in the PyPI dashboard.
- `github-release` — creates the GitHub release with the dists
  attached + `pipx install` instructions in the notes.

Once green, https://pypi.org/project/autogpt-local-executor/ shows
the version.

### 3. Refresh packaging stubs

The Homebrew formula + Scoop manifest in `packaging/` still have
`PLACEHOLDER-` URLs and hashes. The script fetches the dists from
PyPI and patches them in place:

```bash
scripts/update_packaging.sh v0.0.1
```

Output:

```
Fetching sdist: https://files.pythonhosted.org/.../autogpt_local_executor-0.0.1.tar.gz
Fetching wheel: https://files.pythonhosted.org/.../autogpt_local_executor-0.0.1-py3-none-any.whl
sdist sha256 = 7f3b8a91...
wheel sha256 = 9c81f4a2...
Updated:
  /repo/packaging/homebrew/autogpt-local-executor.rb
  /repo/packaging/scoop/autogpt-local-executor.json

Next:
  1. Review the diffs.
  2. Copy them into the tap / bucket repos:
     - Significant-Gravitas/homebrew-tap → Formula/autogpt-local-executor.rb
     - Significant-Gravitas/scoop-bucket → bucket/autogpt-local-executor.json
  3. Open PRs against those repos.
```

### 4. Tap + bucket PRs

```bash
# Homebrew
cd ~/code/Significant-Gravitas/homebrew-tap
cp /path/to/autogpt-local-executor/packaging/homebrew/autogpt-local-executor.rb Formula/
git checkout -b bump-autogpt-local-executor-v0.0.1
git add Formula/autogpt-local-executor.rb
git commit -m "autogpt-local-executor v0.0.1"
git push -u origin bump-autogpt-local-executor-v0.0.1
gh pr create --title "autogpt-local-executor v0.0.1" --body "..."

# Scoop — same pattern in the bucket repo.
```

Both can be self-reviewed and merged immediately for v0.0.1; once the
release cadence picks up, an "auto-update" workflow in the tap/bucket
repos (pattern: https://github.com/scoopinstaller/scoop-bucket
checker) becomes worth the time.

### 5. Smoke test

Either side of merge to the tap/bucket:

```bash
# macOS / Linux
pipx install autogpt-local-executor==0.0.1
autogpt-shim --version  # confirms install
autogpt-shim doctor      # confirms permissions / TCC consent
# Don't `autogpt-shim start` yet — coordinate with the platform team
# to make sure a test session is ready to receive the connection.

# Homebrew (after tap merge)
brew install Significant-Gravitas/tap/autogpt-local-executor
autogpt-shim --version

# Windows / Scoop (after bucket merge)
scoop bucket add autogpt https://github.com/Significant-Gravitas/scoop-bucket
scoop install autogpt/autogpt-local-executor
autogpt-shim --version
```

### 6. Announce

When everything is green:

- Update the platform PR (#13050) description / status if it's still
  open.
- Post in the relevant Slack channel: "v0.0.1 of autogpt-local-executor
  is on PyPI / brew / scoop. Gated behind LD flag
  `local-pc-executor`. Install instructions at <repo README>."

---

## What goes wrong (recoveries)

- **`build` job fails on `verify pyproject version matches tag`** —
  you tagged before bumping. Delete the tag (`git tag -d v0.0.1 &&
  git push --delete origin v0.0.1`), bump `pyproject.toml`, commit,
  re-tag, push.
- **`publish-pypi` job fails with "OIDC trusted publisher not
  configured"** — the pending publisher at pypi.org/manage/account/
  publishing/ doesn't match the workflow filename or environment
  name. Compare the workflow's `id-token: write` permission +
  environment name (`pypi`) against the pending publisher form.
- **`publish-pypi` fails with "package already exists"** — a prior
  run uploaded the dists for this version. Either bump the version
  and re-release, or delete the version from PyPI (only possible if
  it was published in the last few minutes; otherwise yank-only).
- **Smoke test pipx install fails** — confirm the wheel uploaded
  correctly: https://pypi.org/project/autogpt-local-executor/0.0.1/
  Files tab. If the wheel is missing but the sdist is there, the
  build job had a problem isolated to the wheel job.

---

## Future: auto-bump tap + bucket

Currently steps 3+4 are manual. A future enhancement: a separate
workflow in the tap / bucket repos that polls the autogpt-local-executor
GitHub releases API on a daily cron, runs `update_packaging.sh`
inside the tap/bucket, and auto-opens a PR. Pattern:
https://github.com/dawidd6/action-homebrew-bump-formula for Homebrew;
hand-rolled for Scoop. Not worth the complexity until release
cadence justifies it (probably >1 release per month).
