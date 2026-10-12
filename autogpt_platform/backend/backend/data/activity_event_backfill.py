"""One-off backfill: seed file activity events from existing workspace files.

Agent-created workspace files predate the activity-event log, so without
this the Home "Recent work" feed starts empty even for users whose agents
have been producing files for weeks. Derives a ``file.created`` event per
live agent-created file, stamped with the file's own creation time.

Safe to re-run: files that already have a FILE event are skipped.

Run with:
    poetry run python -m backend.data.activity_event_backfill [--dry-run]
"""

import argparse
import asyncio
import logging
import re

import prisma.models
import prisma.types

from backend.data import db
from backend.util.json import SafeJson

logger = logging.getLogger(__name__)

_BATCH_SIZE = 500
_SESSION_PATH = re.compile(r"^/sessions/([^/]+)/")


async def backfill_file_events(dry_run: bool = False) -> int:
    created_total = 0
    skip = 0
    while True:
        files = await prisma.models.UserWorkspaceFile.prisma().find_many(
            where={"isDeleted": False},
            include={"Workspace": True},
            # id as tiebreaker: createdAt is not unique, and skip-based pages
            # over a non-deterministic order can drop rows on the boundary.
            order=[{"createdAt": "asc"}, {"id": "asc"}],
            take=_BATCH_SIZE,
            skip=skip,
        )
        if not files:
            return created_total
        skip += len(files)
        created_total += await _backfill_batch(files, dry_run=dry_run)


async def _backfill_batch(
    files: list[prisma.models.UserWorkspaceFile], dry_run: bool
) -> int:
    agent_files = [
        file
        for file in files
        if file.Workspace and dict(file.metadata or {}).get("origin") == "agent-created"
    ]
    if not agent_files:
        return 0

    existing = await prisma.models.ActivityEvent.prisma().find_many(
        where={"category": "FILE", "objectId": {"in": [f.id for f in agent_files]}}
    )
    seen_file_ids = {event.objectId for event in existing}

    session_ids = [
        match.group(1)
        for file in agent_files
        if (match := _SESSION_PATH.match(file.path))
    ]
    sessions = await prisma.models.ChatSession.prisma().find_many(
        where={"id": {"in": session_ids}}
    )
    session_by_id = {session.id: session for session in sessions}

    rows: list[prisma.types.ActivityEventCreateWithoutRelationsInput] = []
    for file in agent_files:
        if file.id in seen_file_ids or not file.Workspace:
            continue
        match = _SESSION_PATH.match(file.path)
        session = session_by_id.get(match.group(1)) if match else None
        if session and session.userId != file.Workspace.userId:
            session = None
        rows.append(
            {
                "createdAt": file.createdAt,
                "userId": file.Workspace.userId,
                "organizationId": session.organizationId if session else None,
                "expertId": session.expertId if session else None,
                "sessionId": session.id if session else None,
                "category": "FILE",
                "eventType": "file.created",
                "objectId": file.id,
                "title": file.name,
                "data": SafeJson(
                    {
                        "path": file.path,
                        "mime_type": file.mimeType,
                        "size_bytes": int(file.sizeBytes),
                        "backfilled": True,
                    }
                ),
            }
        )

    if dry_run:
        for row in rows:
            logger.info(
                "Would create event for file %s (%s)", row["title"], row["userId"]
            )
        return len(rows)
    if rows:
        await prisma.models.ActivityEvent.prisma().create_many(data=rows)
    return len(rows)


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    await db.connect()
    try:
        count = await backfill_file_events(dry_run=args.dry_run)
        logger.info(
            "%s %d file event(s)",
            "Would backfill" if args.dry_run else "Backfilled",
            count,
        )
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(_main())
