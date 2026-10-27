from .batch_outcome import BatchPass, _applied_result
from .schemas import IngestionDrainStatus


def test_applied_result_preserves_recall_write_outcomes():
    result = _applied_result(
        BatchPass(
            user_id="u1", expert_id=None, pass_id="p1", job_id="j1", phase_models={}
        ),
        {
            "consolidated_count": 1,
            "dropped_forgotten": 2,
            "pending_writes": 3,
            "failed_writes": 4,
        },
        IngestionDrainStatus.skipped,
        "done",
    )

    assert result.consolidated_count == 1
    assert (result.dropped_forgotten, result.pending_writes, result.failed_writes) == (
        2,
        3,
        4,
    )
