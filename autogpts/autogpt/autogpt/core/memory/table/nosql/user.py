from __future__ import annotations


from .base import BaseNoSQLTable


class UsersInformationsTable(BaseNoSQLTable):
    table_name = "users_informations"
    primary_key = "user_id"
