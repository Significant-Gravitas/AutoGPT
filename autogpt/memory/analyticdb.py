""" AnalyticDB memory storage provider."""
import uuid

import psycopg2cffi as psycopg2

from autogpt.llm import get_ada_embedding
from autogpt.memory.base import MemoryProviderSingleton


class AnalyticDBMemory(MemoryProviderSingleton):
    """AnalyticDB memory storage provider."""

    def __init__(self, cfg):
        """Construct an AnalyticDB memory storage connection.

        Args:
            cfg (Config): Auto-GPT global config.
        """
        self.relation_name = cfg.pg_relation
        self.user = cfg.pg_user
        self.password = cfg.pg_password
        self.database = cfg.pg_database
        self.host = cfg.pg_host
        self.port = cfg.pg_port
        self.vec_num = uuid.uuid4().hex

        self.conn = psycopg2.connect(
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
        )
        self._initialize_db()

    def _initialize_db(self):
        with self.conn.cursor() as cur:
            self._create_table(cur)
            self._create_embedding_index(cur)
            self.conn.commit()

    def _create_table(self, cur: psycopg2.extensions.cursor):
        cur.execute(
            f"""
              CREATE TABLE IF NOT EXISTS {self.relation_name} (
                id TEXT PRIMARY KEY DEFAULT uuid_generate_v4()::TEXT,
                raw_text TEXT,
                embedding real[]
            );
            """
        )

    def _create_embedding_index(self, cur: psycopg2.extensions.cursor):
        cur.execute(
            f"""
            SELECT * FROM pg_indexes WHERE tablename='{self.relation_name}';
            """
        )
        index_exists = any(
            index[2] == f"{self.relation_name}_embedding_idx"
            for index in cur.fetchall()
        )
        if not index_exists:
            cur.execute(
                f"""
                CREATE INDEX {self.relation_name}_embedding_idx
                ON {self.relation_name}
                USING ann(embedding)
                WITH (
                    distancemeasure=L2,
                    dim=1536,
                    pq_segments=64,
                    hnsw_m=100,
                    pq_centers=2048
                );
                """
            )

    def add(self, data) -> str:
        """Add a embedding of data into memory.

        Args:
            data (str): The raw text to construct embedding index.

        Returns:
            str: log.
        """
        embedding = get_ada_embedding(data)

        self._upsert_data(data, embedding)
        _text = (
            "Inserting data into memory at primary key: "
            f"{self.vec_num}:\n data: {data}"
        )
        return _text

    def _upsert_data(self, data, embedding):
        self.vec_num = uuid.uuid4().hex

        upsert_data = (self.vec_num, data, embedding)

        with self.conn.cursor() as cur:
            # Construct the SQL query and data
            query = f"""
                    INSERT INTO {self.relation_name} (id, raw_text, embedding)
                    VALUES (%s::text, %s::text, %s::real[])
                    ON CONFLICT (id) DO UPDATE SET
                        raw_text = EXCLUDED.raw_text,
                        embedding = EXCLUDED.embedding;
            """

            # Execute the query
            cur.execute(query, upsert_data)

            # Commit the transaction
            self.conn.commit()

    def get(self, data):
        """Return the most relevant data in memory.
        Args:
            data: The data to compare to.
        """
        return self.get_relevant(data, 1)

    def clear(self) -> str:
        """Drop the index in memory. For AnalyticDB is TRUNCATE RELATION.

        Returns:
            str: log.
        """

        query = f""" TRUNCATE {self.relation_name};"""

        # Execute the query
        with self.conn.cursor() as cur:
            cur.execute(query)

        # Commit the transaction
        self.conn.commit()
        return "Obliviated"

    def get_relevant(self, data: str, num_relevant: int = 5):
        """Return the top-k relevant data in memory.
        Args:
            data: The data to compare to.
            num_relevant (int, optional): The max number of relevant data.
                Defaults to 5.

        Returns:
            list: The top-k relevant data.
        """
        # search the embedding and return the most relevant text.
        embedding = get_ada_embedding(data)

        # Convert the embedded_query to PostgresSQL compatible format
        embedding_query_pg = "{" + ",".join(map(str, embedding)) + "}"

        # Create SQL query
        query_sql = f"""
        SELECT raw_text FROM {self.relation_name}
        ORDER BY embedding <-> '{embedding_query_pg}'::real[]
        LIMIT {num_relevant};
        """
        # Execute the query
        with self.conn.cursor() as cur:
            cur.execute(query_sql)
            results = cur.fetchall()

        return [item[0] for item in results]

    def get_stats(self) -> str:
        """
        Returns: The stats of the milvus cache.
        """
        query_sql = f"""
        SELECT count(*) FROM {self.relation_name};
        """
        with self.conn.cursor() as cur:
            cur.execute(query_sql)
            result = cur.fetchone()
        return f"Entities num: {result[0]}"
