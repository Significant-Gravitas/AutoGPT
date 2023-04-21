from hashlib import sha1
from typing import List, Optional

from clickhouse_connect import get_client
from colorama import Fore, Style

from autogpt.llm_utils import create_embedding_with_ada
from autogpt.logs import logger
from autogpt.memory.base import MemoryProviderSingleton


class MyScaleMemory(MemoryProviderSingleton):
    def __init__(self, cfg, init):
        self.client = get_client(
            host=cfg.myscale_host,
            port=cfg.myscale_port,
            username=cfg.myscale_username,
            password=cfg.myscale_password,
            secure=True,
        )
        dimension = 1536
        metric = "cosine"
        self.BS = "\\"
        self.must_escape = ("\\", "'")
        self.table_name = f"{cfg.myscale_database}.auto_gpt"

        try:
            self.client.query("SELECT 1")
        except Exception as e:
            logger.typewriter_log(
                "FAILED TO CONNECT TO MYSCALE",
                Fore.RED,
                Style.BRIGHT + str(e) + Style.RESET_ALL,
            )
            logger.double_check(
                "Please ensure you have setup and configured MyScale properly for use."
                + f"You can check out {Fore.CYAN + Style.BRIGHT}"
                "https://github.com/Torantulino/Auto-GPT#myscale-setup"
                f"{Style.RESET_ALL} to ensure you've set up everything correctly."
            )
            exit(1)
        schema_ = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name}(
                id String,
                vector Array(Float32),
                raw_text String,
                CONSTRAINT cons_vec_len CHECK length(vector) = {dimension},
                VECTOR INDEX vidx vector TYPE {cfg.myscale_index_type}('metric_type={metric}')
            ) ENGINE = MergeTree ORDER BY id
        """
        if init:
            self.clear()
        self.client.command(schema_)

    def escape_str(self, value: str) -> str:
        return "".join(f"{self.BS}{c}" if c in self.must_escape else c for c in value)

    def _build_istr(self, transac):
        _data = []
        for n in transac:
            n = ",".join([f"'{self.escape_str(str(_n))}'" for _n in n])
            _data.append(f"({n})")
        i_str = f"""
                INSERT INTO TABLE {self.table_name}
                VALUES
                {','.join(_data)}
                """
        return i_str

    def _insert(self, transac):
        _i_str = self._build_istr(transac)
        self.client.command(_i_str)

    def add(self, data):
        vector = create_embedding_with_ada(data)
        doc_id = sha1(data.encode("utf-8")).hexdigest()
        self._insert([[sha1(data.encode("utf-8")).hexdigest(), vector, data]])
        _text = f"Inserting data into memory at index: {doc_id}:\n data: {data}"
        return _text

    def get(self, data):
        return self.get_relevant(data, 1)

    def clear(self):
        self.client.command(f"DROP TABLE IF EXISTS {self.table_name}")
        return "Obliviated"

    def _build_qstr(self, q_emb: List[float], topk: int) -> str:
        q_emb = ",".join(map(str, q_emb))
        q_str = f"""
            SELECT raw_text
            FROM {self.table_name}
            ORDER BY distance(vector, [{q_emb}]) AS dist
            LIMIT {topk}
            """
        return q_str

    def get_relevant(self, data, num_relevant=5):
        """
        Returns all the data in the memory that is relevant to the given data.
        :param data: The data to compare to.
        :param num_relevant: The number of relevant data to return. Defaults to 5
        """
        query_embedding = create_embedding_with_ada(data)
        q_str = self._build_qstr(query_embedding, num_relevant)
        return [
            str(item["raw_text"]) for item in self.client.query(q_str).named_results()
        ]

    def get_stats(self):
        cnt = [
            r
            for r in self.client.query(
                f"SELECT COUNT(*) AS cnt FROM {self.table_name}"
            ).named_results()
        ][0]["cnt"]
        return f"Entities num: {cnt}"
