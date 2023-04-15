import os
import sys

from autogpt.memory.sqlitemem import SqliteMemory


def MockConfig():
    return type(
        "MockConfig",
        (object,),
        {
            "debug_mode": False,
            "continuous_mode": False,
            "speak_mode": False,
            "memory_index": "test",
            "memory_backend": "sqlite",
        },
    )


class TestSqliteMemory(unittest.TestCase):
    def setUp(self):
        self.cfg = MockConfig()
        self.memory = SqliteMemory(self.cfg)
        self.memory.clear()

    def tearDown(self):
        self.memory.clear()

    def test_add(self):
        text = "Sample text"
        self.memory.add(text)

        with self.memory.sql_conn as conn:
            conn.row_factory = lambda cur, row: row[0]
            res = conn.execute(f"SELECT data FROM {self.cfg.memory_index};").fetchall()
        self.assertIn(text, res)

    def test_clear(self):
        text = "Sample text"
        self.memory.add(text)
        self.memory.clear()

        with self.memory.sql_conn as conn:
            res = conn.execute(f"SELECT * FROM {self.cfg.memory_index}").fetchall()
        self.assertEqual(res, [])

    def test_get(self):
        text = "Sample text"
        self.memory.add(text)

        result = self.memory.get(text)
        self.assertEqual(result, [text])

    def test_get_relevant(self):
        text1 = "Sample text 1"
        text2 = "Sample text 2"
        self.memory.add(text1)
        self.memory.add(text2)

        result = self.memory.get_relevant(text1, 1)
        self.assertEqual(result, [text1])

    def test_get_stats(self):
        text = "Sample text"
        self.memory.add(text)

        stats = self.memory.get_stats()
        self.assertEqual(
            stats, {"num_memories": 1, "index": self.cfg.memory_index, "embedder": "ada"}
        )


if __name__ == "__main__":
    unittest.main()
