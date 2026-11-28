"""Expert reads the copilot executor gets through the database manager survive
the trip as the models they were, not as plain dicts."""

import json
from datetime import UTC, date, datetime

from backend.util.service import _build_return_adapter

from . import experts_db
from .models import ExpertActivity, ExpertActivityDay, ExpertRun


def test_expert_runs_survive_the_rpc_round_trip() -> None:
    runs = [
        ExpertRun(
            execution_id="exec-1",
            graph_id="graph-1",
            agent_name="Competitor watch",
            library_agent_id="lib-1",
            status="completed",
            output_type="table",
            output_key="changes",
            needs_review=False,
            source="scheduled",
            started_at=datetime(2026, 11, 20, 9, tzinfo=UTC),
            ended_at=datetime(2026, 11, 20, 9, 1, tzinfo=UTC),
            link="/library/agents/lib-1?activeItem=exec-1",
        )
    ]
    adapter = _build_return_adapter(
        experts_db.list_expert_runs, list[ExpertRun], "list_expert_runs"
    )
    assert adapter is not None

    delivered = adapter.validate_python(
        json.loads(json.dumps([run.model_dump(mode="json") for run in runs]))
    )

    assert delivered == runs
    assert isinstance(delivered[0], ExpertRun)


def test_expert_activity_survives_the_rpc_round_trip() -> None:
    activity = ExpertActivity(
        timezone="Europe/Amsterdam",
        days=[ExpertActivityDay(day=date(2026, 11, 20), sessions=2, runs=5)],
    )
    adapter = _build_return_adapter(
        experts_db.get_expert_activity, ExpertActivity, "get_expert_activity"
    )
    assert adapter is not None

    delivered = adapter.validate_python(json.loads(activity.model_dump_json()))

    assert delivered == activity
    assert isinstance(delivered.days[0], ExpertActivityDay)
