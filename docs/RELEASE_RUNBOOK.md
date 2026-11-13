# Release Runbook

> One-time setup + per-release steps for publishing
> `autogpt-local-executor` to PyPI, Homebrew, and Scoop.
>
> Releases are built by `.github/workflows/release.yml` in
> `Significant-Gravitas/autogpt-local-executor`. The copy of this
> package in the AutoGPT monorepo (`local_executor/`) has no release
> workflow yet, so run these steps in `autogpt-local-executor`.
> `scripts/update_packaging.sh` and the formula and manifest stubs in
> `packaging/` are in both. This doc is the "who does what" so the
> first publisher doesn't have to reverse-engineer it.

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
- Every change made to the monorepo copy (`local_executor/` in
  AutoGPT) is in this repository too. Fixes usually land here first and
  reach the monorepo with `git subtree pull`, but a change made only in
  the monorepo has to come back the other way before you tag. Don't
  `git subtree push` it onto the release branch (`experimental/local-pc-executor`
  until there is another): the branch it writes matches `local_executor/`,
  which has none of this repository's `.github/workflows/` (`release.yml`
  would be gone before the tag) and has the monorepo's own `README.md`,
  `AGENTS.md`, `CLAUDE.md` and `pyproject.toml`. Push it to a branch of its
  own, from an AutoGPT checkout:

  ```bash
  git subtree push --prefix=local_executor \
    git@github.com:Significant-Gravitas/autogpt-local-executor.git \
    sync/from-monorepo  # a branch only this step writes
  ```

  Then merge that into the release branch, from a checkout of this
  repository:

  ```bash
  RELEASE_BRANCH=experimental/local-pc-executor  # the branch step 1 tags
  git checkout "$RELEASE_BRANCH" && git pull
  git fetch origin sync/from-monorepo
  git merge --no-ff --no-commit FETCH_HEAD  # "Already up to date": nothing to bring back
  git restore --source=HEAD --staged --worktree -- .github README.md pyproject.toml
  git rm -rqf --ignore-unmatch -- AGENTS.md CLAUDE.md
  git commit --no-edit
  ```

  `--no-ff` stops the merge before a fast-forward could take the
  monorepo's tree whole. `git restore` puts back this repository's
  `.github/`, `README.md` and `pyproject.toml`, conflicted or not, and
  `git rm` leaves out the monorepo's `AGENTS.md` and `CLAUDE.md`, which
  this repository doesn't have (only the first sync brings them in, so
  they aren't named to `git restore`, which refuses a path it can't find).
  A change the monorepo made to one of those files for both copies (a new
  dependency in `pyproject.toml`, say) has to be made here by hand before
  the commit: `git diff HEAD...FETCH_HEAD -- README.md pyproject.toml`
  shows what the monorepo changed. Then check the release branch and push
  it:

  ```bash
  git diff --stat FETCH_HEAD HEAD
  ls .github/workflows/release.yml
  git push origin "$RELEASE_BRANCH"
  ```

  The diff should list only the files the monorepo keeps its own way
  (`.github/workflows/` as its two workflows, `AGENTS.md`, `CLAUDE.md`,
  `README.md` and `pyproject.toml`), and any commit of this repository's
  the monorepo hasn't pulled in yet.

### 1. Tag + push

On the release branch, once the pre-flight's merge is pushed:

```bash
git checkout experimental/local-pc-executor  # the release branch, as above
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
PyPI and patches them in place. Running it again replaces the values
from the previous run:

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

WARNING: /repo/packaging/homebrew/autogpt-local-executor.rb has no resource blocks.
  Without them, brew install leaves out pydantic, websockets, httpx,
  keyring and the rest, and autogpt-shim fails to start.
  Do step 2 below before you copy the formula to the tap.

Next:
  1. Review the diffs.
  2. Generate the formula's resource blocks (needs Homebrew). Do this
     on every release, because the dependencies change:
       brew update-python-resources /repo/packaging/homebrew/autogpt-local-executor.rb
  3. Copy the files into the tap / bucket repos:
     - Significant-Gravitas/homebrew-tap → Formula/autogpt-local-executor.rb
     - Significant-Gravitas/scoop-bucket → bucket/autogpt-local-executor.json
  4. Open PRs against those repos.
```

Then generate the formula's `resource` blocks. This is required.
Homebrew installs the package with `pip --no-deps`, so a formula
without them leaves out pydantic, websockets, httpx, keyring and the
rest, and `autogpt-shim` fails to start. It needs Homebrew, so run it
on a Mac or on Linux with Homebrew:

```bash
brew update-python-resources packaging/homebrew/autogpt-local-executor.rb
```

Homebrew builds every resource from source, and pydantic-core needs
Rust, so also add `depends_on "rust" => :build` to the formula. Do
this on every release, because the dependencies change. Don't copy the
formula to the tap until it has resource blocks.

### 4. Tap + bucket PRs

```bash
# Homebrew (the formula must already have its resource blocks; see step 3)
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
pipx install git+https://github.com/Significant-Gravitas/autogpt-local-executor.git
autogpt-shim --version  # confirms install
autogpt-shim doctor      # confirms permissions / TCC consent
autogpt-shim auth
autogpt-shim start       # opens the persistent outbound machine connection

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
  is on PyPI / brew / scoop. Gated behind the PostHog flag
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
- **Homebrew's `autogpt-shim` fails with `ModuleNotFoundError`** for
  pydantic or another dependency — the formula reached the tap without
  its resource blocks. Generate them (step 3), add a `revision` line to
  the formula so Homebrew rebuilds it, and open a new tap PR.
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
