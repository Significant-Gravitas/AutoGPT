import psycopg2
from psycopg2 import sql
from typing import Any
from autogpt.llm_utils import get_ada_embedding
from autogpt.logs import logger
from autogpt.memory.base import MemoryProviderSingleton


class PostgresMLMemory(MemoryProviderSingleton):
    def __init__(self, cfg) -> None:
        postgresml_host = cfg.postgresml_host
        postgresml_port = cfg.postgresml_port
        postgresml_username = cfg.postgresml_username
        postgresml_password = cfg.postgresml_password
        postgresml_database = cfg.postgresml_database
        self.postgresml_tablename = cfg.postgresml_tablename

        self.conn_params = {
            "dbname": postgresml_database,
            "user": postgresml_username,
            "password": postgresml_password,
            "host": postgresml_host,
            "port": postgresml_port,
        }
        conn = psycopg2.connect(**self.conn_params)
        # Insert the vector and text into the database
        create_table = (
            "CREATE TABLE IF NOT EXISTS %s (data TEXT, embedding vector(1536))"
            % self.postgresml_tablename
        )
        cur = conn.cursor()
        cur.execute(create_table)
        # Commit the transaction
        conn.commit()
        # Close the cursor and the connection
        cur.close()
        conn.close()

    def add(self, data):
        vector = get_ada_embedding(data)
        conn = psycopg2.connect(**self.conn_params)
        cur = conn.cursor()

        # Insert the vector and text into the database
        insert_statement = sql.SQL(
            "INSERT INTO {} (data, embedding) VALUES ({}, {})"
        ).format(
            sql.Identifier(self.postgresml_tablename),
            sql.Literal(data),
            sql.Literal(vector),
        )

        cur.execute(insert_statement)
        index_statement = (
            "CREATE INDEX ON %s USING ivfflat (embedding vector_cosine_ops)"
            % self.postgresml_tablename
        )
        cur.execute(index_statement)

        # Commit the transaction
        conn.commit()

        # Close the cursor and the connection
        cur.close()
        conn.close()

    def get(self, data: str) -> list[Any] | None:
        """
        Gets the data from the memory that is most relevant to the given data.

        Args:
            data: The data to compare to.

        Returns: The most relevant data.
        """
        return self.get_relevant(data, 1)

    def clear(self) -> str:
        """
        Drops the PostgreSML table.

        Returns: A message indicating that the memory has been cleared.
        """
        drop_table = "DROP TABLE IF EXISTS %s" % self.postgresml_tablename
        conn = psycopg2.connect(**self.conn_params)
        cur = conn.cursor()
        cur.execute(drop_table)
        cur.close()
        conn.close()
        return "Obliviated"

    def get_relevant(self, data: str, num_relevant: int = 5) -> list[Any] | None:
        """
        Returns all the data in the memory that is relevant to the given data.
        Args:
            data: The data to compare to.
            num_relevant: The number of relevant data to return.

        Returns: A list of the most relevant data.
        """
        query_embedding = get_ada_embedding(data)
        query_embedding = ",".join(str(v) for v in query_embedding)
        query_statement = (
            "SELECT data FROM %s ORDER BY %s.embedding <-> ARRAY[%s]::vector LIMIT %d;"
            % (
                self.postgresml_tablename,
                self.postgresml_tablename,
                query_embedding,
                num_relevant,
            )
        )

        conn = psycopg2.connect(**self.conn_params)
        cur = conn.cursor()
        cur.execute(query_statement)
        result = cur.fetchall()
        cur.close()
        conn.close()
        return [r[0] for r in result]

    def get_stats(self):
        """
        Return: Stats of PostgresML table
        """
        conn = psycopg2.connect(**self.conn_params)
        cur = conn.cursor()
        pgstats_statement = (
            "SELECT * FROM pg_stats WHERE tablename = '%s'" % self.postgresml_tablename
        )
        cur.execute(pgstats_statement)
        result = cur.fetchall()
        cur.close()
        conn.close()
        return result
