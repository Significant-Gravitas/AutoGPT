"""The description an update to a Marketplace listing keeps when it sends none."""

import prisma.bases
import prisma.enums


class VersionHistoryRow(prisma.bases.BaseStoreListingVersion):
    """The columns of a listing version that a new submission reads: its number,
    review status and description. A query through this model selects only
    these columns."""

    id: str
    version: int
    description: str
    submissionStatus: prisma.enums.SubmissionStatus


def kept_description(
    history: list[VersionHistoryRow], active_version_id: str | None, sent: str
) -> str:
    """The description an update keeps when the one it sends is blank.

    The live version's description comes first, then the newest one that no
    review rejected. Superseded versions count, since they hold text the
    creator wrote and never withdrew. Blank means empty once stripped. With no
    such description, the update keeps what it sent. ``history`` runs newest
    first.
    """
    live = [v for v in history if v.id == active_version_id]
    unrejected = [
        v
        for v in history
        if v.submissionStatus != prisma.enums.SubmissionStatus.REJECTED
    ]
    return next(
        (v.description for v in [*live, *unrejected] if v.description.strip()),
        sent,
    )
