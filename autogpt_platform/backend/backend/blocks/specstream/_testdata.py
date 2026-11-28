"""Fixed API values shared by the block test data and the block tests."""

from datetime import UTC, datetime

from ._models import (
    ApiCell,
    ApiRow,
    Commerce,
    Execution,
    Provenance,
    Quote,
    Reading,
    RowsPage,
    RunStatus,
)

TEST_REQUEST = "Seed-stage fintech startups in France with their website and founders"
TEST_AST_HASH = "a" * 64
TEST_QUOTE_HASH = "b" * 64
TEST_FIND_ID = "fnd_test_001"
TEST_OBSERVED_AT = datetime(2026, 11, 3, 9, 30, tzinfo=UTC)

TEST_READING = Reading(
    status="runnable",
    ast_hash=TEST_AST_HASH,
    canonical_sentence="Fintech startups in France at seed stage",
)

TEST_QUOTE = Quote(
    quote_id="qte_test_001",
    quote_hash=TEST_QUOTE_HASH,
    maximum_cents=375,
    expires_at=datetime(2026, 11, 3, 10, 0, tzinfo=UTC),
    scope={"requested_units": 25},
)

TEST_EXECUTION = Execution(
    reservation_id="rsv_test_001",
    find_id=TEST_FIND_ID,
    maximum_cents=375,
    state="running",
)

TEST_RUNNING = RunStatus(
    find_id=TEST_FIND_ID, status="running", requested_count=25, accepted_count=7
)

TEST_COMPLETED = RunStatus(
    find_id=TEST_FIND_ID,
    status="completed",
    requested_count=25,
    accepted_count=25,
    stop_reason="complete_target",
    commerce=Commerce(
        amount_authorized_cents=375, amount_captured_cents=375, delivered_rows=25
    ),
)

TEST_ROWS_PAGE = RowsPage(
    find_id=TEST_FIND_ID,
    status="completed",
    accepted_count=25,
    rows=[
        ApiRow(
            entity_id="ent_1",
            ordinal=0,
            cells=[
                ApiCell(
                    column_id="identity",
                    label="Name",
                    value="Acme Pay",
                    display="Acme Pay",
                    provenance=Provenance(
                        source_url="https://acme.example/about",
                        observed_at=TEST_OBSERVED_AT,
                    ),
                ),
                ApiCell(
                    column_id="website",
                    label="Website",
                    value="https://acme.example",
                    display="acme.example",
                    provenance=Provenance(
                        source_url="https://acme.example",
                        observed_at=TEST_OBSERVED_AT,
                    ),
                ),
                ApiCell(
                    column_id="detail:w1",
                    label="Founders",
                    value="Ada Roy",
                    display="Ada Roy",
                ),
                ApiCell(
                    column_id="matches",
                    label="Matches",
                    value=True,
                    display="Why it matches",
                ),
            ],
            worded_evidence={
                "details": [
                    {
                        "detail_id": "w1",
                        "value": "Ada Roy",
                        "witness": {
                            "source_url": "https://acme.example/team",
                            "observed_at": "2026-11-03T09:31:00+00:00",
                        },
                    }
                ]
            },
        )
    ],
    next_cursor="cur_2",
)
