"""One-off Graphiti / FalkorDB migrations.

Migrations here are Cypher operations on the per-user FalkorDB graphs.
They are NOT Prisma migrations and do NOT participate in
``poetry run prisma migrate``. Run each script manually via
``poetry run python -m backend.copilot.graphiti.migrations.<name>``.

Each migration MUST be idempotent — re-running it on an already-migrated
database is a no-op.
"""
