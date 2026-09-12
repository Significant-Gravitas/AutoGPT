---
name: compose-review
description: Review a Docker Compose file or service definition for the faults that actually cause outages in self-hosted setups - missing healthchecks, unpinned tags, data loss on bind mounts, accidental exposure, no restart policy. Use when asked to check a compose file, review a stack, or say why a container keeps restarting.
triggers:
  - review my compose file
  - check this docker-compose
  - why does my container keep restarting
  - is this stack safe
version: "1"
---

# Reviewing a Docker Compose stack

Read for what bites in practice. Ignore style. Go looking for these, in
rough order of how often they ruin someone's evening.

## 1. Data that will not survive

- **Bind mount to a path that does not exist yet.** Docker creates it as a
  root-owned directory. A container running as a non-root user then cannot
  write to it, or a file-shaped mount silently becomes a folder. Name the
  exact path.
- **Named volume vs bind mount confusion.** A named volume that the user
  thinks lives in their backup path does not. Ask where they back up from.
- **Database directory on a network share.** SQLite and Postgres on NFS or
  SMB corrupt under lock contention. This is not theoretical.
- **No volume at all** on a service that holds state. `docker compose down`
  and it is gone.

## 2. Start order that only works by luck

`depends_on` without `condition: service_healthy` waits for the container
to *start*, not to be *ready*. An app that connects on boot will die
against a database still running its init. If the dependency has no
`healthcheck`, say so and give one: the point is that `depends_on` is
useless without it.

The tell: "it works if I start it twice" or "it works after a reboot but
not on first run".

## 3. Tags that make rollback impossible

`latest`, or no tag, means the version that is running is unknown and
unrepeatable. When an update breaks something there is nothing to go back
to. Recommend a pinned minor (`postgres:16.4`) over both `latest` and a
bare major.

## 4. Exposure the author did not intend

- `ports:` publishes on **all interfaces** by default. `"8080:8080"` is
  reachable from the LAN, and from the internet if the router forwards it.
  If it only needs to be reachable by another container, it needs no
  `ports:` at all, just a shared network.
- To bind locally: `"127.0.0.1:8080:8080"`.
- Flag any admin interface, database port, or unauthenticated service that
  is published. Name the service and what an attacker on the LAN reaches.
- Note that published ports usually bypass the host firewall, because
  Docker writes its own iptables rules. People are routinely surprised.

## 5. Restart and failure behaviour

- No `restart:` policy means the service stays down after a host reboot.
  `unless-stopped` is the usual right answer.
- `restart: always` on a container that fails instantly is a crash loop
  that fills the disk with logs unless logging is capped.
- No log limits: add `max-size` and `max-file`, or a chatty container fills
  the root filesystem. This is one of the most common homelab outages.

## 6. Secrets in the file

Passwords inline in `environment:` end up in git and in `docker inspect`.
Point at an `.env` file or a secrets mechanism, and note that if it was
ever committed, rotating is now part of the fix.

## Writing the review

Lead with what will actually break, not with a list of everything. For each
finding: the service, the line, and the concrete consequence, in that
order. "`db` has no healthcheck but `app` depends on it, so app dies on
first boot" beats "consider adding healthchecks".

Then give the corrected snippet. Whole file only if most of it changes.

Say what you could not judge from the file alone: their backup path, what
is behind a reverse proxy, whether a published port is firewalled upstream.
A compose file does not show any of that.

## Never

Do not run anything, do not connect to their host, and do not assume a
directory layout you were not shown.
