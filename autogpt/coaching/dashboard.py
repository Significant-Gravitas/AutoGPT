"""Coach dashboard aggregation."""
from __future__ import annotations

from datetime import datetime

from autogpt.coaching.models import CoachDashboard
from autogpt.coaching.storage import get_client_statuses


def build_dashboard() -> CoachDashboard:
    """Build the coach dashboard from the latest session per client."""
    clients = get_client_statuses()
    return CoachDashboard(
        generated_at=datetime.utcnow(),
        clients=clients,
    )
