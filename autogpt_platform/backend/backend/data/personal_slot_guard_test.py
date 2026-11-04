"""Every tenant table sends a personal organization's rows to its default team.

A table merged in later with both ``organizationId`` and ``teamId`` must carry
migration 20261104140000's ``a0_normalize_personal_team`` trigger, or be listed
in ``NOT_NORMALIZED`` with the reason, or personal rows can land at org-home
again, where reads restricted to the default team don't find them.
"""

import pytest

from backend.data.db import prisma
from backend.util.test import SpinTestServer

TRIGGER = "a0_normalize_personal_team"
# Tables with both tenant columns that keep their team as written.
NOT_NORMALIZED = {
    # History, recorded in the scope the action happened in.
    "AuditLog",
    # Its teamId marks a credential a team owns, never a personal slot.
    "IntegrationCredential",
}


@pytest.mark.asyncio(loop_scope="session")
async def test_every_table_with_tenant_columns_carries_the_normalizer(
    server: SpinTestServer,
):
    rows = await prisma.query_raw(
        """
        SELECT team.table_name
        FROM information_schema.columns AS team
        JOIN information_schema.columns AS org
          ON org.table_schema = team.table_schema
         AND org.table_name = team.table_name
         AND org.column_name = 'organizationId'
        WHERE team.table_schema = current_schema() AND team.column_name = 'teamId'
        """
    )
    tenant_tables = {row["table_name"] for row in rows}
    triggered = await prisma.query_raw(
        """
        SELECT cls.relname
        FROM pg_trigger AS trg
        JOIN pg_class AS cls ON cls.oid = trg.tgrelid
        JOIN pg_namespace AS ns ON ns.oid = cls.relnamespace
        WHERE ns.nspname = current_schema() AND trg.tgname = $1
          AND NOT trg.tgisinternal AND trg.tgenabled = 'O'
        """,
        TRIGGER,
    )

    assert NOT_NORMALIZED <= tenant_tables
    assert {row["relname"] for row in triggered} == tenant_tables - NOT_NORMALIZED
