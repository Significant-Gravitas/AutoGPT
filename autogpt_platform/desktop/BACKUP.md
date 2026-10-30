# Back up and restore desktop data

Make a backup before upgrading an installation with work you want to retain. Quit AutoGPT completely, including its tray or menu-bar app, and wait for it to finish closing. Use the runtime from the version that last opened this data folder. A backup is a cold snapshot: the database, broker and other services must be stopped.

The archive includes the entire data folder: PostgreSQL, queued work, Valkey, workspaces, media, configuration, encryption keys, credentials, caches, logs and pending cancellation records. Only the child-process PID registry is omitted because it describes processes on the old launch. Exporting an agent is not a substitute for this backup. State outside the data folder, such as operating-system keychains, a separate browser profile and external files linked from a workspace, is not included.

**Archives are not encrypted. They contain private credentials and the keys needed to decrypt stored credentials.** Keep them in private storage or an encrypted drive. Anyone who can read one can recover that installation's data. The tool never uploads an archive.

## Create a backup

Choose a new filename outside the data folder. Its parent directory must already exist on a filesystem supporting hard links, such as NTFS, APFS or ext4. The tool refuses to overwrite an existing backup. Allow the size of the entire data folder plus 64 MiB of free space; compression may reduce the final archive, but the space check does not depend on that.

Use the installed app's `resources/runtime` directory. On macOS it is normally `/Applications/AutoGPT.app/Contents/Resources/runtime`. The default data folders are:

| System | Data folder |
| --- | --- |
| Windows | `%LOCALAPPDATA%\AutoGPT` |
| macOS | `~/Library/Application Support/AutoGPT` |
| Linux | `${XDG_DATA_HOME:-~/.local/share}/AutoGPT` |

An install variant uses its own `AutoGPT-<variant>` directory and requires `--install AutoGPT-<variant>`. If you moved the data folder, use its actual path.

Windows PowerShell, after setting `$runtime` to the installed runtime:

```powershell
$runtime = 'C:\path\to\AutoGPT\resources\runtime'
Push-Location $runtime
try {
    & "$runtime\python\python.exe" -B -m autogpt_desktop.backup create `
        --data-dir "$env:LOCALAPPDATA\AutoGPT" `
        --file "$env:USERPROFILE\Documents\AutoGPT-before-upgrade.agptbackup"
} finally {
    Pop-Location
}
```

macOS:

```sh
cd /Applications/AutoGPT.app/Contents/Resources/runtime
./python/bin/python3 -B -m autogpt_desktop.backup create \
  --data-dir "$HOME/Library/Application Support/AutoGPT" \
  --file "$HOME/Documents/AutoGPT-before-upgrade.agptbackup"
```

Linux uses the same Python command from its installed runtime directory, with the Linux data path. Development runs can supply `--runtime /path/to/built/runtime` while importing the source module.

The command refuses a running runtime, desktop shell, active database, browser profile lock, incomplete initial setup, unclean PostgreSQL shutdown, links/junctions inside the data folder, insufficient space or a data tree that changes during copying. The shell and maintenance command publish ownership records before checking each other, preventing a new app launch from writing into a backup or restore in progress. A malformed ownership record blocks the operation rather than assuming its owner is gone. It never follows a link into unlisted external files. Runtime-created RabbitMQ path aliases live outside the data folder and are rebuilt at startup. If a stale process or profile marker blocks backup, start and fully quit the current app version; do not remove a marker while services might still be running.

## Restore a backup

Quit AutoGPT completely again. Use `restore` with the same paths and add `--confirm-restore`:

```powershell
& "$runtime\python\python.exe" -B -m autogpt_desktop.backup restore `
    --data-dir "$env:LOCALAPPDATA\AutoGPT" `
    --file "$env:USERPROFILE\Documents\AutoGPT-before-upgrade.agptbackup" `
    --confirm-restore
```

Run that command from `$runtime`, as in the create example. On macOS/Linux, replace the executable and paths using the corresponding example above.

Restore supports the original data path and install identity on the same operating system, architecture and PostgreSQL major version. It does not move a physical database between machines or platforms. Use the app version that created the backup, or a compatible newer version. The manifest records the supplied runtime's migration files; it does not independently read the stopped database's applied migration table. A runtime supplied from a different version can make the compatibility check conservatively refuse a restore. On app startup, applied database migration names and checksums are checked against the installed runtime before schema changes; a runtime missing or changing an applied migration refuses startup.

The tool checks every archive path and checksum, extracts into a new sibling directory, verifies the staged database has a clean shutdown, then replaces the data directory. The existing directory is retained beside it as `AutoGPT.before-restore-<id>` and its exact path is printed. Restore requires room for the full staged copy plus 64 MiB in addition to the existing data. It does not delete current data or old backups to make space. Files are flushed before replacement; POSIX directory updates are flushed too. Windows does not provide the same directory-flush API through Python. The previous directory is retained on every platform.

After restoring, start AutoGPT and check that you can sign in, decrypt a saved test credential, open saved work and see the expected run history. Do not remove the retained directory until those checks succeed. Restore only trusted backups: checksums detect corruption, not an archive deliberately modified by someone who can also rewrite its manifest.

Before replacing the data directory, restore writes a sibling `.autogpt-restore-<id>.json` journal containing the current, previous and staged directory paths. If the command is interrupted or cannot safely roll back, it retains that journal and all remaining copies. The shell and runtime refuse to initialize or change the data until recovery is resolved, even if another program recreated an empty directory at the original path. Keep every recorded directory. With all app processes closed, use the journal to identify and verify the complete copy you intend to recover; put that verified copy at the original path while retaining the other copies, and remove the journal only after that recovery is complete. Never clear the journal merely because a directory exists at the original path. An ordinary failed replacement that can restore the original directory rolls back and clears the journal itself.

## Upgrade and downgrade behavior

There is no automatic backup or retention policy yet. Make the cold backup before installing the update, then verify your saved data after the new version starts. Keep both the backup and the corresponding installer until you have checked the upgrade.

Installing an older binary does not undo database migrations. Versions with this guard refuse a database whose completed migrations they do not recognize. Binaries released before this guard cannot enforce it. To return to an earlier schema, fully quit the app, install the matching earlier version, and restore a complete backup made before the upgrade. Work created after that snapshot remains in the retained directory; it is not automatically merged into the restored database.

## Development verification

The focused tests cover archive integrity, complete file recovery, ownership and path conflicts, insufficient space, live-service refusal, schema compatibility, interrupted replacement and rollback failures. `runtime/tests/test_backup_postgres.py` is opt-in: set `AUTOGPT_TEST_POSTGRES_RUNTIME` to an existing built runtime and run that test with the bundled Python and the source runtime on `PYTHONPATH`. It creates a disposable PostgreSQL cluster on a dynamically allocated loopback port, saves an encrypted synthetic credential and workspace, backs up, applies a newer migration, proves the older migration history is refused, then restores and decrypts the original credential. It stops only its own cluster and removes its own disposable data. This proves physical PostgreSQL recovery; it does not replace installed-app upgrade, account sign-in or broker recovery testing.
