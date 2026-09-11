---
name: backup-restore-check
description: Judge whether a backup would actually restore, rather than whether the backup job succeeded. Use when asked about backups, disaster recovery, whether a restore would work, or to review a backup strategy for self-hosted services.
triggers:
  - are my backups ok
  - check my backup setup
  - would this restore
  - disaster recovery
---

# Checking a backup is restorable

A backup job exiting zero proves the job ran. It does not prove the data is
usable, complete, or recoverable by anyone other than the person who set it
up. Work through these.

## The question that comes first

**When was a restore last actually performed, end to end, onto something
other than the source machine?**

If the answer is "never", that is the finding, and it outranks everything
else on this page. Say it plainly. An untested backup is a hypothesis.

Follow with: how long did it take, and was the result checked, or just
assumed because files appeared.

## What is not covered

Ask what is *excluded*, not what is included. People back up
`/var/lib/docker/volumes` and miss:

- **Bind-mounted data outside the volume root.** Very common.
- **The compose files and `.env` themselves.** Restoring data without the
  stack definition means rebuilding from memory.
- **Databases inside containers.** A filesystem copy of a running Postgres
  or MySQL data directory is a crash-consistent snapshot at best and
  corrupt at worst. These need `pg_dump` / `mysqldump`, or a filesystem
  snapshot taken with the engine quiesced.
- **Secrets and certificates.** Recoverable in principle, painful in
  practice.
- **The reverse proxy config and DNS**, without which nothing is reachable
  even once restored.

## Where the copies live

Apply the 3-2-1 test honestly and say which leg is missing:

- Three copies, on two kinds of media, one off-site.
- A second disk in the same machine is one copy, not two. A NAS in the same
  building survives disk failure but not fire or theft.
- **A backup the source machine can delete is not a backup.** If the host
  holds credentials that can erase the remote copy, ransomware or a bad
  script takes both. Look for append-only, immutable, or pull-based
  arrangements.

## Would it be noticed if it stopped

- Is a failure reported anywhere a human sees, or does it only appear in a
  log nobody reads?
- Silent success is worse than loud failure: check whether the job would
  report success on an empty or partial run.
- When did the last one run, and how big was it compared with the one
  before? A sudden shrink means something stopped being included.

## Retention against the actual threat

Restoring last night's backup does not help with corruption or an accidental
deletion noticed a week later. Ask how far back they can go, and whether
retention is long enough to cover the gap between a mistake and noticing.

## Writing it up

Lead with the single thing most likely to lose data. Then the others,
consequence first. Give the exact verification command where one exists
(`restic check --read-data-subset`, `pg_restore --list`, a test restore into
a scratch container), and say what a good result looks like.

Finish with the smallest useful next step. "Restore last night's database
dump into a throwaway container and run one query" is a thirty-minute task
that converts a hypothesis into a fact.

## Never

Do not run backup or restore commands, and never suggest one that writes
over existing data without saying so first and giving the undo.
