from pathlib import Path

import yaml

BACKEND_DIR = Path(__file__).resolve().parents[1]


def test_test_database_matches_ci_without_mounting_development_data():
    compose = yaml.safe_load((BACKEND_DIR / "docker-compose.test.yaml").read_text())
    database = compose["services"]["db"]

    assert database["image"] == "pgvector/pgvector:pg15"
    assert "extends" not in database
    assert database["environment"]["POSTGRES_DB"] == "agpt_test"
    assert database["volumes"] == [
        "../db/init/00-init.sql:/docker-entrypoint-initdb.d/00-init.sql:ro",
        "test-db-data:/var/lib/postgresql/data",
    ]
    assert "vector" not in compose["services"]
