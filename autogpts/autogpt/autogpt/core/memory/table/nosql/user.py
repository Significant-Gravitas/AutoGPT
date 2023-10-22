from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from .base import BaseNoSQLTable


class UsersInformationsTable(BaseNoSQLTable):
    table_name = "users_informations"
    primary_key = "user_id"
