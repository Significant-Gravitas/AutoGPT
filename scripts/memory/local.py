import sqlite3
import numpy as np
from memory.base import MemoryProviderSingleton, get_ada_embedding

EMBED_DIM = 1536

class LocalCache(MemoryProviderSingleton):
    def __init__(self, cfg) -> None:
        self.filename = f"{cfg.memory_index}.db"
        self.conn = sqlite3.connect(self.filename)
        self._initialize_database()

    def _initialize_database(self):
        cursor = self.conn.cursor()
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS texts (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
            '''
        )
        self.conn.commit()

    def add(self, text: str):
        if 'Command Error:' in text:
            return ""
        
        embedding = get_ada_embedding(text)
        vector = np.array(embedding).astype(np.float32).tobytes()

        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO texts (text, embedding) VALUES (?, ?)", (text, vector))
        self.conn.commit()

        return text

    def clear(self) -> str:
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM texts")
        self.conn.commit()

        return "Obliviated"

    def get_relevant(self, text: str, k: int) -> List[Any]:
        embedding = get_ada_embedding(text)
        target_vector = np.array(embedding).astype(np.float32)

        cursor = self.conn.cursor()
        cursor.execute("SELECT text, embedding FROM texts")
        rows = cursor.fetchall()

        texts = []
        embeddings = []

        for row in rows:
            texts.append(row[0])
            embeddings.append(np.frombuffer(row[1], dtype=np.float32))

        matrix = np.stack(embeddings)
        scores = np.dot(matrix, target_vector)
        top_k_indices = np.argsort(scores)[-k:][::-1]

        return [texts[i] for i in top_k_indices]

    def get_stats(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM texts")
        count = cursor.fetchone()[0]

        return count, EMBED_DIM

