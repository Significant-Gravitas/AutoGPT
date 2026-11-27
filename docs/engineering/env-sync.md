# Syncing selected development settings

`scripts/env_sync.py` copies chosen keys from env files on your other machines
into a local one, over SSH. It never prints a value. Without `--key`, it syncs
the three settings the Microsoft app shares between the backend and the Teams
bot: `MICROSOFT_CLIENT_ID`, `MICROSOFT_CLIENT_SECRET` and
`MICROSOFT_TENANT_ID`.

From `autogpt_platform/backend`:

```bash
poetry run python -m scripts.env_sync \
  --source 'mac:/path/to/backend/.env' \
  --source 'desktop:C:\path\to\backend\.env' \
  --target .env
```

This is a dry run: it reports each key as `matches`, `different`,
`missing at target` or `missing at source`. Add `--apply` to write the keys
missing at the target. Replacing a `different` value also requires `--replace`.
Sources with conflicting values are rejected: choose one explicitly. Empty or
missing source keys never clear the target. Use repeated `--key KEY_NAME` to
select another explicit allowlist. Name the SSH hosts that run Windows with
repeated `--windows-host HOST`; without it, `desktop` is the only one.

Applying checks the target again under a lock just before writing, so a value
another run added since the report still needs `--replace`. It then copies the
existing target to `~/.local/state/autogpt-env-sync/`
(`$XDG_STATE_HOME/autogpt-env-sync/` when that is set), a directory only you
can read and outside any repository, so `git add` cannot stage the copy. It
prints the copy's path. The script never deletes these copies, and each holds
every setting the target had, so remove old ones you no longer need. Finally it
replaces the target atomically, keeping unrelated settings.

Tracked files, symlinks and template env files are rejected, by the dry run
too. Only local
targets are writable; remote environments stay unchanged. Choose a canonical
source before enabling any scheduled sync.

A sync that is killed mid-write can leave `<target>.sync-lock`, and possibly an
`.env-sync-*` file, beside the target. This repository's `.gitignore` covers
both; for a target in another repository, add the two patterns to its
`.gitignore`. Later runs stop and name the lock; delete it once no sync is
running.
