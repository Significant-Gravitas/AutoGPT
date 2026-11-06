"""Tests for how SpecStream rows and their sources are read."""

from backend.blocks.specstream._models import ApiCell, ApiRow, DatasetRow, Provenance


def test_a_cell_source_missing_a_part_is_filled_from_its_witness():
    row = ApiRow(
        entity_id="ent_1",
        cells=[
            ApiCell(
                column_id="detail:w1",
                label="Founders",
                value="Ada Roy",
                provenance=Provenance(source_url="https://acme.example/own"),
            )
        ],
        worded_evidence={
            "details": [
                {
                    "detail_id": "w1",
                    "witness": {
                        "source_url": "https://acme.example/team",
                        "observed_at": "2026-11-03T09:31:00+00:00",
                    },
                }
            ]
        },
    )

    cell = DatasetRow.from_api(row).cells[0]

    assert cell.source_url == "https://acme.example/own"
    assert cell.read_at == "2026-11-03T09:31:00+00:00"
