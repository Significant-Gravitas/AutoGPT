from __future__ import annotations

from .....interfaces.db.table.nosql.base import BaseNoSQLTable


class UsersInformationsTable(BaseNoSQLTable):
    table_name = "users_informations"
    primary_key = "user_id"
