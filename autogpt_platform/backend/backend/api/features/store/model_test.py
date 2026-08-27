import datetime
from unittest.mock import MagicMock

import pytest

from . import model as store_model


def _listing_version(owning_org_id: str | None) -> MagicMock:
    """A StoreListingVersion stand-in with its StoreListing included."""
    listing = MagicMock()
    listing.id = "listing-1"
    listing.owningUserId = "user-1"
    listing.owningOrgId = owning_org_id
    listing.slug = "my-agent"

    version = MagicMock()
    version.StoreListing = listing
    version.id = "listing-version-1"
    version.version = 3
    version.agentGraphId = "graph-1"
    version.agentGraphVersion = 2
    version.name = "My Agent"
    version.subHeading = "Does things"
    version.description = "A longer description"
    version.instructions = None
    version.categories = []
    version.imageUrls = []
    version.videoUrl = None
    version.agentOutputDemoUrl = None
    version.submittedAt = datetime.datetime(2025, 6, 1, tzinfo=datetime.timezone.utc)
    version.changesSummary = "Initial"
    version.submissionStatus = "PENDING"
    version.reviewedAt = None
    version.reviewerId = None
    version.reviewComments = None
    version.internalComments = None
    return version


@pytest.mark.parametrize("owning_org_id", ["org-42", None])
def test_from_listing_version_surfaces_owning_org(owning_org_id: str | None) -> None:
    """The listing's owning org is what drives the org badge on submissions,
    so it has to survive the mapping — including the pre-backfill NULL case."""
    submission = store_model.StoreSubmission.from_listing_version(
        _listing_version(owning_org_id)
    )

    assert submission.organization_id == owning_org_id


def test_admin_view_from_listing_version_surfaces_owning_org() -> None:
    """The admin view re-uses the same mapping, so it must carry the org too."""
    submission = store_model.StoreSubmissionAdminView.from_listing_version(
        _listing_version("org-42")
    )

    assert submission.organization_id == "org-42"
