"""The shipped expert roster, for tests that assert on its content.

The roster lives in the skills catalog (``experts/*.yml`` in
``Significant-Gravitas/skills-catalog``), a separate repository. It is read
from ``SKILLS_CATALOG_PATH`` or a cached download of the configured ref, so a
run that cannot reach it skips the roster-content tests instead of erroring.
"""

import httpx
import pytest

from backend.api.features.experts.roster import RosterError, load_roster
from backend.api.features.experts.roster_types import RosterEntry
from backend.api.features.store.skill_catalog_checkout import catalog_checkout
from backend.api.features.store.skill_catalog_release import CatalogError


@pytest.fixture(scope="session")
def real_roster() -> list[RosterEntry]:
    try:
        return load_roster(catalog_checkout().root)
    except (RosterError, CatalogError, OSError, httpx.HTTPError) as exc:
        pytest.skip(f"skills catalog roster unavailable: {exc}")


@pytest.fixture(scope="session")
def roster_by_name(real_roster: list[RosterEntry]) -> dict[str, RosterEntry]:
    return {entry["name"]: entry for entry in real_roster}
