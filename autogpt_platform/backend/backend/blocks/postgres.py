from enum import Enum
from typing import Any, List, Literal, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel, SecretStr

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import (
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
    UserPasswordCredentials,
)
from backend.integrations.providers import ProviderName

PostgresCredentials = UserPasswordCredentials
PostgresCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.POSTGRES],
    Literal["user_password"],
]


def PostgresCredentialsField() -> PostgresCredentialsInput:
    """Creates a Postgres credentials input on a block."""
    return CredentialsField(
        description="The Postgres integration requires a username and password.",
    )


TEST_POSTGRES_CREDENTIALS = UserPasswordCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="postgres",
    username=SecretStr("mock-postgres-username"),
    password=SecretStr("mock-postgres-password"),
    title="Mock Postgres credentials",
)

TEST_POSTGRES_CREDENTIALS_INPUT = {
    "provider": TEST_POSTGRES_CREDENTIALS.provider,
    "id": TEST_POSTGRES_CREDENTIALS.id,
    "type": TEST_POSTGRES_CREDENTIALS.type,
    "title": TEST_POSTGRES_CREDENTIALS.title,
}


class CommandType(str, Enum):
    TRUNCATE = "TRUNCATE"
    DELETE = "DELETE"
    DROP = "DROP"


class ConditionOperator(str, Enum):
    EQUALS = "="
    NOT_EQUALS = "<>"
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUALS = ">="
    LESS_EQUALS = "<="
    LIKE = "LIKE"
    IN = "IN"


class Condition(BaseModel):
    column: str
    operator: ConditionOperator
    value: Any


class CombineCondition(str, Enum):
    AND = "AND"
    OR = "OR"


class PostgresDeleteBlock(Block):
    class Input(BlockSchema):
        credentials: PostgresCredentialsInput = PostgresCredentialsField()
        host: str = SchemaField(description="Database host", advanced=False)
        port: int = SchemaField(description="Database port", advanced=False)
        database: str = SchemaField(description="Database name", default="postgres",advanced=False)
        schema_: str = SchemaField(description="Schema name", default="public",advanced=False)
        table: str = SchemaField(description="Table name")
        command: CommandType = SchemaField(
            description="Command type to execute",
            default=CommandType.DELETE,
            advanced=False
        )
        conditions: List[Condition] = SchemaField(
            description="Conditions for DELETE command",
            default=[],
            advanced=False
        )
        combine_conditions: CombineCondition = SchemaField(
            description="How to combine multiple conditions",
            default=CombineCondition.AND,
            advanced=False
        )
        restart_sequences: bool = SchemaField(
            description="Restart any auto-incrementing counters associated with the table after truncate",
            default=False
        )
        cascade: bool = SchemaField(
            description="This automatically truncates any tables that reference the target table via foreign keys, Only used for Truncate and Drop",
            default=False
        )

    class Output(BlockSchema):
        success: bool = SchemaField(description="Operation succeeded")
        rows_affected: Optional[int] = SchemaField(description="Number of rows affected")
        error: str = SchemaField(description="Error message if operation failed")

    def __init__(self):
        super().__init__(
            id="81b103ad-0fa9-47d3-a18f-2ea96579e3bb",
            description="Delete, truncate or drop data from a PostgreSQL table",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=PostgresDeleteBlock.Input,
            output_schema=PostgresDeleteBlock.Output,
            test_credentials=TEST_POSTGRES_CREDENTIALS,
            test_input={
                "credentials": TEST_POSTGRES_CREDENTIALS_INPUT,
                "host": "localhost",
                "port": 5432,
                "database": "test_db",
                "schema_": "public",
                "table": "users",
                "command": CommandType.DELETE,
                "conditions": [
                    {"column": "id", "operator": ConditionOperator.EQUALS, "value": 1}
                ]
            },
            test_output=[
                ("success", True),
                ("rows_affected", 1)
            ],
        )

    def run(
        self, input_data: Input, *, credentials: PostgresCredentials, **kwargs
    ) -> BlockOutput:
        conn = None
        try:
            conn = psycopg2.connect(
                host=input_data.host,
                port=input_data.port,
                database=input_data.database,
                user=credentials.username.get_secret_value(),
                password=credentials.password.get_secret_value()
            )

            with conn.cursor() as cursor:
                rows_affected = 0

                if input_data.command == CommandType.TRUNCATE:
                    sql = f"TRUNCATE TABLE {input_data.schema_}.{input_data.table}"
                    if input_data.restart_sequences:
                        sql += " RESTART IDENTITY"
                    if input_data.cascade:
                        sql += " CASCADE"
                    cursor.execute(sql)

                elif input_data.command == CommandType.DELETE:
                    if input_data.conditions:
                        where_clauses = []
                        values = []

                        for condition in input_data.conditions:
                            if condition.operator == ConditionOperator.IN:
                                placeholders = ", ".join(["%s"] * len(condition.value))
                                where_clauses.append(f"{condition.column} IN ({placeholders})")
                                values.extend(condition.value)
                            else:
                                where_clauses.append(f"{condition.column} {condition.operator.value} %s")
                                values.append(condition.value)

                        where_clause = f" {input_data.combine_conditions.value} ".join(where_clauses)
                        sql = f"DELETE FROM {input_data.schema_}.{input_data.table} WHERE {where_clause}"
                        cursor.execute(sql, values)
                    else:
                        sql = f"DELETE FROM {input_data.schema_}.{input_data.table}"
                        cursor.execute(sql)

                    rows_affected = cursor.rowcount

                elif input_data.command == CommandType.DROP:
                    sql = f"DROP TABLE {input_data.schema_}.{input_data.table}"
                    if input_data.cascade:
                        sql += " CASCADE"
                    cursor.execute(sql)

                conn.commit()
                yield "success", True
                yield "rows_affected", rows_affected

        except Exception as e:
            if conn:
                conn.rollback()
            yield "error", str(e)

        finally:
            if conn:
                conn.close() # Just for extra safety

class PostgresExecuteQueryBlock(Block):
    class Input(BlockSchema):
        credentials: PostgresCredentialsInput = PostgresCredentialsField()
        host: str = SchemaField(description="Database host", advanced=False)
        port: int = SchemaField(description="Database port", advanced=False)
        database: str = SchemaField(description="Database name", default="postgres", advanced=False)
        schema_: str = SchemaField(description="Schema name", default="public", advanced=False)
        query: str = SchemaField(description="SQL query to execute")
        parameters: List[Any] = SchemaField(description="Query parameters", default=[], advanced=False)

    class Output(BlockSchema):
        success: bool = SchemaField(description="Operation succeeded")
        result: Any = SchemaField(description="Query results or affected rows")
        error: str = SchemaField(description="Error message if operation failed")

    def __init__(self):
        super().__init__(
            id="c5d18dc8-ee3c-4366-ba99-a3996b7a4e78",
            description="Executes an SQL query on a PostgreSQL database.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=PostgresExecuteQueryBlock.Input,
            output_schema=PostgresExecuteQueryBlock.Output,
            test_credentials=TEST_POSTGRES_CREDENTIALS,
            test_input={
                "credentials": TEST_POSTGRES_CREDENTIALS_INPUT,
                "host": "localhost",
                "port": 5432,
                "database": "test_db",
                "schema_": "public",
                "query": "SELECT * FROM users WHERE id = %s",
                "parameters": [1]
            },
            test_output=[
                ("success", True),
                ("result", [{"id": 1, "name": "Test User"}])
            ],
        )

    def run(
        self, input_data: Input, *, credentials: PostgresCredentials, **kwargs
    ) -> BlockOutput:
        conn = None
        try:
            conn = psycopg2.connect(
                host=input_data.host,
                port=input_data.port,
                database=input_data.database,
                user=credentials.username.get_secret_value(),
                password=credentials.password.get_secret_value()
            )

            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Using RealDictCursor to return data as dict, otherwise cursor return data as tuple
                cursor.execute(input_data.query, input_data.parameters)
                if cursor.description:
                    result = cursor.fetchall()
                    result = [dict(row) for row in result]
                else:
                    # Query doesn't return data (INSERT, UPDATE, DELETE)
                    result = cursor.rowcount # Number of rows affected by executing this query

                conn.commit()
                yield "success", True
                yield "result", result

        except Exception as e:
            if conn:
                conn.rollback()
            yield "error", str(e)

        finally:
            if conn:
                conn.close()

class PostgresInsertBlock(Block):
    class Input(BlockSchema):
        credentials: PostgresCredentialsInput = PostgresCredentialsField()
        host: str = SchemaField(description="Database host", advanced=False)
        port: int = SchemaField(description="Database port", advanced=False)
        database: str = SchemaField(description="Database name", default="postgres", advanced=False)
        schema_: str = SchemaField(description="Schema name", default="public", advanced=False)
        table: str = SchemaField(description="Table name")
        data: List[dict] = SchemaField(description="Data to insert", default=[])
        return_inserted_rows: bool = SchemaField(description="Return inserted rows", default=False)

    class Output(BlockSchema):
        success: bool = SchemaField(description="Operation succeeded")
        inserted_rows: List[dict] = SchemaField(description="Inserted rows if requested")
        rows_affected: int = SchemaField(description="Number of rows affected")
        error: str = SchemaField(description="Error message if operation failed")

    def __init__(self):
        super().__init__(
            id="82a6c2d5-4c6f-4e3a-aba2-feae15c03cbe",
            description="Inserts rows into a PostgreSQL table",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=PostgresInsertBlock.Input,
            output_schema=PostgresInsertBlock.Output,
            test_credentials=TEST_POSTGRES_CREDENTIALS,
            test_input={
                "credentials": TEST_POSTGRES_CREDENTIALS_INPUT,
                "host": "localhost",
                "port": 5432,
                "database": "test_db",
                "schema_": "public",
                "table": "users",
                "data": [{"name": "Test User", "email": "test@example.com"}],
                "return_inserted_rows": True
            },
            test_output=[
                ("success", True),
                ("rows_affected", 1),
                ("inserted_rows", [{"id": 1, "name": "Test User", "email": "test@example.com"}])
            ],
        )

    def run(
        self, input_data: Input, *, credentials: PostgresCredentials, **kwargs
    ) -> BlockOutput:
        conn = None
        try:
            conn = psycopg2.connect(
                host=input_data.host,
                port=input_data.port,
                database=input_data.database,
                user=credentials.username.get_secret_value(),
                password=credentials.password.get_secret_value()
            )

            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                if not input_data.data:
                    yield "success", True
                    yield "rows_affected", 0
                    yield "inserted_rows", []
                    return

                columns = list(input_data.data[0].keys())
                cols_str = ", ".join(columns)
                placeholders = ", ".join(["%s"] * len(columns))
                sql = f"INSERT INTO {input_data.schema_}.{input_data.table} ({cols_str}) VALUES ({placeholders})"

                if input_data.return_inserted_rows:
                    sql += " RETURNING *"

                inserted_rows = []
                rows_affected = 0

                for row in input_data.data:
                    values = [row[col] for col in columns]
                    cursor.execute(sql, values)
                    rows_affected += cursor.rowcount

                    if input_data.return_inserted_rows:
                        inserted_rows.extend([dict(row) for row in cursor.fetchall()])

                conn.commit()
                yield "success", True
                yield "rows_affected", rows_affected
                yield "inserted_rows", inserted_rows

        except Exception as e:
            if conn:
                conn.rollback()
            yield "success", False
            yield "error", str(e)

        finally:
            if conn:
                conn.close()

class PostgresInsertOrUpdateBlock(Block):
    class Input(BlockSchema):
        credentials: PostgresCredentialsInput = PostgresCredentialsField()
        host: str = SchemaField(description="Database host", advanced=False)
        port: int = SchemaField(description="Database port", advanced=False)
        database: str = SchemaField(description="Database name", default="postgres", advanced=False)
        schema_: str = SchemaField(description="Schema name", default="public", advanced=False)
        table: str = SchemaField(description="Table name")
        data: List[dict] = SchemaField(description="Data to insert or update", default=[])
        key_columns: List[str] = SchemaField(description="Columns to use as unique constraint", default=[])
        return_affected_rows: bool = SchemaField(description="Return affected rows", default=False)

    class Output(BlockSchema):
        success: bool = SchemaField(description="Operation succeeded")
        affected_rows: List[dict] = SchemaField(description="Affected rows if requested")
        rows_affected: int = SchemaField(description="Number of rows affected")
        error: str = SchemaField(description="Error message if operation failed")

    def __init__(self):
        super().__init__(
            id="fa8e0ce3-5b8c-49e2-a3b7-dca21f5c4a72",
            description="Inserts or updates rows in a PostgreSQL table using ON CONFLICT",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=PostgresInsertOrUpdateBlock.Input,
            output_schema=PostgresInsertOrUpdateBlock.Output,
            test_credentials=TEST_POSTGRES_CREDENTIALS,
            test_input={
                "credentials": TEST_POSTGRES_CREDENTIALS_INPUT,
                "host": "localhost",
                "port": 5432,
                "database": "test_db",
                "schema_": "public",
                "table": "users",
                "data": [{"id": 1, "name": "Updated User", "email": "updated@example.com"}],
                "key_columns": ["id"],
                "return_affected_rows": True
            },
            test_output=[
                ("success", True),
                ("rows_affected", 1),
                ("affected_rows", [{"id": 1, "name": "Updated User", "email": "updated@example.com"}])
            ],
        )

    def run(
        self, input_data: Input, *, credentials: PostgresCredentials, **kwargs
    ) -> BlockOutput:
        conn = None
        try:
            conn = psycopg2.connect(
                host=input_data.host,
                port=input_data.port,
                database=input_data.database,
                user=credentials.username.get_secret_value(),
                password=credentials.password.get_secret_value()
            )

            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                if not input_data.data or not input_data.key_columns:
                    yield "success", True
                    yield "rows_affected", 0
                    yield "affected_rows", []
                    return

                affected_rows = []
                rows_affected = 0

                for row in input_data.data:
                    columns = list(row.keys())
                    cols_str = ", ".join(columns)
                    placeholders = ", ".join(["%s"] * len(columns))
                    conflict_cols = ", ".join(input_data.key_columns)
                    update_cols = ", ".join(
                        f"{col} = EXCLUDED.{col}" for col in columns if col not in input_data.key_columns
                    )

                    sql = (
                        f"INSERT INTO {input_data.schema_}.{input_data.table} ({cols_str}) "
                        f"VALUES ({placeholders}) ON CONFLICT ({conflict_cols}) DO UPDATE SET {update_cols}"
                    )

                    if input_data.return_affected_rows:
                        sql += " RETURNING *"

                    values = [row[col] for col in columns]
                    cursor.execute(sql, values)
                    rows_affected += cursor.rowcount

                    if input_data.return_affected_rows:
                        affected_rows.extend([dict(row) for row in cursor.fetchall()])

                conn.commit()
                yield "success", True
                yield "rows_affected", rows_affected
                yield "affected_rows", affected_rows

        except Exception as e:
            if conn:
                conn.rollback()
            yield "success", False
            yield "error", str(e)

        finally:
            if conn:
                conn.close()

class PostgresSelectBlock(Block):
    class Input(BlockSchema):
        credentials: PostgresCredentialsInput = PostgresCredentialsField()
        host: str = SchemaField(description="Database host", advanced=False)
        port: int = SchemaField(description="Database port", advanced=False)
        database: str = SchemaField(description="Database name", default="postgres", advanced=False)
        schema_: str = SchemaField(description="Schema name", default="public", advanced=False)
        table: str = SchemaField(description="Table name")
        columns: List[str] = SchemaField(description="Columns to select (empty for all columns)", default=[])
        conditions: List[Condition] = SchemaField(description="Conditions for WHERE clause", default=[], advanced=False)
        combine_conditions: CombineCondition = SchemaField(
            description="How to combine multiple conditions",
            default=CombineCondition.AND,
            advanced=False
        )
        limit: Optional[int] = SchemaField(description="Maximum number of rows to return", default=None)

    class Output(BlockSchema):
        success: bool = SchemaField(description="Operation succeeded")
        rows: List[dict] = SchemaField(description="Selected rows")
        error: str = SchemaField(description="Error message if operation failed")

    def __init__(self):
        super().__init__(
            id="e7c92ea5-1d2a-4e9c-bb89-376dfcbea342",
            description="Selects rows from a PostgreSQL table",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=PostgresSelectBlock.Input,
            output_schema=PostgresSelectBlock.Output,
            test_credentials=TEST_POSTGRES_CREDENTIALS,
            test_input={
                "credentials": TEST_POSTGRES_CREDENTIALS_INPUT,
                "host": "localhost",
                "port": 5432,
                "database": "test_db",
                "schema_": "public",
                "table": "users",
                "columns": ["id", "name", "email"],
                "conditions": [
                    {"column": "id", "operator": ConditionOperator.GREATER_THAN, "value": 0}
                ],
                "limit": 100
            },
            test_output=[
                ("success", True),
                ("rows", [{"id": 1, "name": "Test User", "email": "test@example.com"}])
            ],
        )

    def run(
        self, input_data: Input, *, credentials: PostgresCredentials, **kwargs
    ) -> BlockOutput:
        conn = None
        try:
            conn = psycopg2.connect(
                host=input_data.host,
                port=input_data.port,
                database=input_data.database,
                user=credentials.username.get_secret_value(),
                password=credentials.password.get_secret_value()
            )

            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cols = ", ".join(input_data.columns) if input_data.columns else "*"
                sql = f"SELECT {cols} FROM {input_data.schema_}.{input_data.table}"

                values = []
                if input_data.conditions:
                    where_clauses = []

                    for condition in input_data.conditions:
                        if condition.operator == ConditionOperator.IN:
                            placeholders = ", ".join(["%s"] * len(condition.value))
                            where_clauses.append(f"{condition.column} IN ({placeholders})")
                            values.extend(condition.value)
                        else:
                            where_clauses.append(f"{condition.column} {condition.operator.value} %s")
                            values.append(condition.value)

                    where_clause = f" {input_data.combine_conditions.value} ".join(where_clauses)
                    sql += f" WHERE {where_clause}"

                if input_data.limit is not None:
                    sql += f" LIMIT {input_data.limit}"

                cursor.execute(sql, values)
                rows = [dict(row) for row in cursor.fetchall()]

                yield "success", True
                yield "rows", rows

        except Exception as e:
            if conn:
                conn.rollback()
            yield "success", False
            yield "error", str(e)

        finally:
            if conn:
                conn.close()

class PostgresUpdateBlock(Block):
    class Input(BlockSchema):
        credentials: PostgresCredentialsInput = PostgresCredentialsField()
        host: str = SchemaField(description="Database host", advanced=False)
        port: int = SchemaField(description="Database port", advanced=False)
        database: str = SchemaField(description="Database name", default="postgres", advanced=False)
        schema_: str = SchemaField(description="Schema name", default="public", advanced=False)
        table: str = SchemaField(description="Table name")
        set_data: dict = SchemaField(description="Column-value pairs to update", default={})
        conditions: List[Condition] = SchemaField(description="Conditions for WHERE clause", default=[], advanced=False)
        combine_conditions: CombineCondition = SchemaField(
            description="How to combine multiple conditions",
            default=CombineCondition.AND,
            advanced=False
        )
        return_updated_rows: bool = SchemaField(description="Return updated rows", default=False)

    class Output(BlockSchema):
        success: bool = SchemaField(description="Operation succeeded")
        rows_affected: int = SchemaField(description="Number of rows affected")
        updated_rows: List[dict] = SchemaField(description="Updated rows if requested")
        error: str = SchemaField(description="Error message if operation failed")

    def __init__(self):
        super().__init__(
            id="a4e3d8c2-7f1b-49d0-8bc6-e479ea3d5752",
            description="Updates rows in a PostgreSQL table",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=PostgresUpdateBlock.Input,
            output_schema=PostgresUpdateBlock.Output,
            test_credentials=TEST_POSTGRES_CREDENTIALS,
            test_input={
                "credentials": TEST_POSTGRES_CREDENTIALS_INPUT,
                "host": "localhost",
                "port": 5432,
                "database": "test_db",
                "schema_": "public",
                "table": "users",
                "set_data": {"name": "Updated User", "email": "updated@example.com"},
                "conditions": [
                    {"column": "id", "operator": ConditionOperator.EQUALS, "value": 1}
                ],
                "return_updated_rows": True
            },
            test_output=[
                ("success", True),
                ("rows_affected", 1),
                ("updated_rows", [{"id": 1, "name": "Updated User", "email": "updated@example.com"}])
            ],
        )

    def run(
        self, input_data: Input, *, credentials: PostgresCredentials, **kwargs
    ) -> BlockOutput:
        conn = None
        try:
            conn = psycopg2.connect(
                host=input_data.host,
                port=input_data.port,
                database=input_data.database,
                user=credentials.username.get_secret_value(),
                password=credentials.password.get_secret_value()
            )

            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                if not input_data.set_data:
                    yield "success", True
                    yield "rows_affected", 0
                    yield "updated_rows", []
                    return

                set_clause = ", ".join(f"{k} = %s" for k in input_data.set_data.keys())
                sql = f"UPDATE {input_data.schema_}.{input_data.table} SET {set_clause}"

                values = list(input_data.set_data.values())

                if input_data.conditions:
                    where_clauses = []

                    for condition in input_data.conditions:
                        if condition.operator == ConditionOperator.IN:
                            placeholders = ", ".join(["%s"] * len(condition.value))
                            where_clauses.append(f"{condition.column} IN ({placeholders})")
                            values.extend(condition.value)
                        else:
                            where_clauses.append(f"{condition.column} {condition.operator.value} %s")
                            values.append(condition.value)

                    where_clause = f" {input_data.combine_conditions.value} ".join(where_clauses)
                    sql += f" WHERE {where_clause}"

                if input_data.return_updated_rows:
                    sql += " RETURNING *"

                cursor.execute(sql, values)
                rows_affected = cursor.rowcount

                updated_rows = []
                if input_data.return_updated_rows:
                    updated_rows = [dict(row) for row in cursor.fetchall()]

                conn.commit()
                yield "success", True
                yield "rows_affected", rows_affected
                yield "updated_rows", updated_rows

        except Exception as e:
            if conn:
                conn.rollback()
            yield "success", False
            yield "error", str(e)

        finally:
            if conn:
                conn.close()
