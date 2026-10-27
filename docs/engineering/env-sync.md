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

This is a dry run. Add `--apply` to write missing values. Replacing a different
nonempty target value also requires `--replace`. Sources with conflicting values
are rejected: choose one explicitly. Empty or missing source keys never clear
the target. Use repeated `--key KEY_NAME` to select another explicit allowlist.
Use `--windows-host HOST` for additional Windows SSH aliases (`desktop` is one
by default).

Applying creates a private backup and replaces the target atomically, keeping
unrelated settings. Tracked files, symlinks and template env files are
rejected. Only local targets are writable; remote environments stay unchanged.
Choose a canonical source before enabling any scheduled sync.
