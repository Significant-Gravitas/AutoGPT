"""SQLite memory provider."""
import shutil
import sqlite3
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import zarr
from colorama import Fore, Style
from sklearn.neighbors import NearestNeighbors

from autogpt.logger import logger
from autogpt.memory.base import MemoryProviderSingleton
from autogpt.memory.base import get_ada_embedding as get_embedding 

CHUNK_SIZE = 1000
EMBED_DIM = 1536


class SqliteMemory(MemoryProviderSingleton):
    """
    SQLite memory provider.
    Based on: https://github.com/wawawario2/long_term_memory
    """

    def __init__(self, cfg):
        """
        Initializes the SQLite memory provider.

        Args:
            cfg: The config object.

        Returns: None
        """
        # Store the index (table name) and the embedder name
        self.index = cfg.sqlite_index
        self.embedder_name = "ada"

        # Set database and embeddings to None
        self.sql_conn = None
        self.embeddings = None

        # Memory index cannot start with digits or contains any dashes, dots or spaces
        if self.index[0].isdigit() or any(c in self.index for c in "-. "):
            logger.error(
                "Invalid memory index: "
                f"{self.index} cannot start with digits or contain any dashes, dots or spaces. Got {self.index!r}. "
            )
            exit(1)

        # Construct the paths to the database file and embeddings directory
        parent_dir = Path("memory")
        self.database_path = parent_dir / "memory.db"
        self.embeddings_path = (
            parent_dir / f"embeddings_{self.index}_{self.embedder_name}.zarr"
        )

        # If parent directory doesn't exist, create it
        parent_dir.mkdir(parents=True, exist_ok=True)

        # Check if database exists
        if not self.database_path.exists():
            # Check if embeddings directory exists
            if self.embeddings_path.exists():
                # Database doesn't exist, but embeddings for this index do
                logger.error(
                    "Inconsistent state detected: "
                    f"{self.embeddings_path!r} exists but {self.database_path!r} does not. "
                    f"Delete {Fore.CYAN + Style.BRIGHT}{self.embeddings_path!r}{Style.RESET_ALL} and try again."
                )
                exit(1)

            # Neither database nor embeddings exist, create them
            self._create_table()
            self._create_embeddings()
        # Database exists, check if embeddings exist
        elif not self.embeddings_path.exists():
            # Check if another embeddings directory for this index exists, but using a different embedder
            for file in parent_dir.glob(f"embeddings_{self.index}_*.zarr"):
                # Found another embeddings directory using a different embedder, convert it
                old_embedder_name = file.stem.split("_")[-1]
                self._convert_embeddings(file, old_embedder_name)
                break

            # Embeddings don't exist, check if a table for this index exists and is empty
            conn = sqlite3.connect(self.database_path)
            with conn:
                res = conn.execute(
                    f"SELECT name FROM sqlite_master WHERE type='table' AND name='{self.index}';"
                )
                if res.fetchone():
                    # Table exists, but embeddings don't
                    logger.error(
                        "Inconsistent state detected: "
                        f"{self.database_path} exists but {self.embeddings_path!r} does not. "
                        f"Delete {Fore.CYAN + Style.BRIGHT}{self.database_path!r}{Style.RESET_ALL} and try again."
                    )

                    # If user wants to regenerate embeddings, do so
                    print("Would you like to regenerate the embeddings? [Y/n] ", end="")
                    if input().lower() != "n":
                        self._regenerate_embeddings()
                    # Otherwise, wipe the memory
                    else:
                        self.clear()

            # Close the connection
            conn.close()

        # Create a connection to the database and load the embeddings array
        self.sql_conn = sqlite3.connect(self.database_path)
        self.embeddings = zarr.open(self.embeddings_path, mode="a")

        # Create table if it doesn't exist
        self._create_table()

        # Check if the row count in the database matches the number of embeddings
        self._verify_integrity()

    def _verify_integrity(self):
        """
        Verifies the integrity of the database and embeddings.

        Returns: None
        """
        # Get the row count in the database
        with self.sql_conn as conn:
            res = conn.execute(f"SELECT COUNT(*) FROM {self.index};")
            db_count = res.fetchone()[0]

        # Get the number of embeddings
        embed_count = self.embeddings.shape[0]

        # If the counts don't match, regenerate the embeddings
        if db_count != embed_count:
            logger.error(
                "Inconsistent state detected: "
                f"Row count in {self.database_path} ({db_count}) does not match the number of embeddings "
                f"({embed_count}). Remove {Fore.CYAN + Style.BRIGHT}{self.embeddings_path!r}{Style.RESET_ALL} and try again."
            )
            exit(1)

    def _create_table(self):
        """
        Creates a table in the database.

        Returns: None
        """
        conn = self.sql_conn
        if self.sql_conn is None:
            conn = sqlite3.connect(self.database_path)

        with conn:
            conn.execute(
                f"""
            CREATE TABLE IF NOT EXISTS {self.index} (
                id INTEGER PRIMARY KEY,
                data TEXT NOT NULL UNIQUE,
                timestamp TEXT NOT NULL
            );
            """
            )

        if self.sql_conn is None:
            conn.close()

    def _create_embeddings(self):
        """
        Creates the embeddings array.

        Returns: None
        """
        zarr.open(
            self.embeddings_path,
            mode="w",
            shape=(0, EMBED_DIM),
            chunks=(CHUNK_SIZE, EMBED_DIM),
            dtype="float32",
        )

    def _regenerate_embeddings(self):
        """
        Regenerates the embeddings.

        Returns: None
        """
        # Delete the embeddings for index (if they exist)
        self.embeddings = None
        shutil.rmtree(self.embeddings_path, ignore_errors=True)

        # Create and load the empty embeddings
        self._create_embeddings()
        self.embeddings = zarr.open(self.embeddings_path, mode="a")

        # For each row in the database, generate the embedding and store it
        with self.sql_conn as conn:
            res = conn.execute(f"SELECT id, text FROM {self.index};")
            for row in res:
                # Get the index (id) and text
                index, text = row

                # Generate the embedding
                embedding = get_embedding(text)
                embedding = np.expand_dims(embedding, axis=0)

                # Store the embedding
                self.embeddings[index] = embedding

    def _convert_embeddings(
        self, old_embeddings: Path, prev_embedder: str, soft_delete: bool = True
    ) -> None:
        """
        Converts the embeddings from a different embedder to the current one.

        Args:
            old_embeddings: The path to the previous embeddings
            prev_embedder: The name of the previous embedder

        Returns: None
        """

        # Ask user if they want to regenerate embeddings (may come with additional costs for embedders like "ada")
        print(
            f"WARNING: Embeddings for index '{self.index}' were generated "
            f"with a different embedder ({prev_embedder}) than the current one ({self.embedder_name}). "
            "Regenerating embeddings (may come with additional costs if using embedders like 'ada'), "
            "do you want to regenerate them [y/N]?",
            end="",
        )
        if input().lower() != "y":
            logger.error(
                "Embeddings for this index were generated with a different embedder. "
                "Either regenerate them, use a different embedder or remove the old embeddings file."
            )
            exit(1)

        # Create the new embeddings array
        self._create_embeddings()

        # Get the data from the database
        with self.sql_conn as conn:
            res = conn.execute(f"SELECT id, data, timestamp FROM {self.index};")
            rows = res.fetchall()

        # For each data point, get the embedding and add it to the new embeddings array
        for index, data, timestamp in rows:
            # embedding = get_embedding(data)
            embedding = get_embedding(data)
            embedding = np.expand_dims(embedding, axis=0)
            self.embeddings[index] = embedding

        # Rename the old embeddings
        if soft_delete:
            old_embeddings.rename(old_embeddings.with_suffix(".zarr.old"))
        else:
            shutil.rmtree(old_embeddings, ignore_errors=True)

    def add(self, data: str) -> str:
        """
        Adds a data point to the memory.

        Args:
            data: The data to add.

        Returns: Message indicating that the data has been added.
        """
        # Create the data embedding
        embedding = get_embedding(data)
        embedding = np.expand_dims(embedding, axis=0)

        # Get the embedding index (index of the next vector is the same as the number of vectors)
        embedding_index = self.embeddings.shape[0]

        # Add data to the database
        with self.sql_conn as conn:
            try:
                conn.execute(
                    f"INSERT INTO {self.index} (id, data, timestamp) VALUES(?, ?, CURRENT_TIMESTAMP);",
                    (embedding_index, data),
                )
            except sqlite3.IntegrityError as err:
                if "UNIQUE constraint failed:" in str(err):
                    return "Trying to add duplicate data to memory, ignoring."
                raise err

        # Store the embedding
        self.embeddings.append(embedding)
        return f"Inserting data into memory at index: {embedding_index}:\ndata: {data}"

    def get(self, data: str) -> Optional[List[Any]]:
        """
        Gets the data from the memory that is most relevant to the given data.

        Args:
            data: The data to compare to.

        Returns: The most relevant data.
        """
        return self.get_relevant(data, 1)

    def clear(self) -> str:
        """
        Clears the memory.

        Returns: A message indicating that the memory has been cleared.
        """
        # Wipe the embeddings for index
        self.embeddings = None
        shutil.rmtree(self.embeddings_path, ignore_errors=True)
        self._create_embeddings()
        self.embeddings = zarr.open(self.embeddings_path, mode="a")

        # Wipe the table for index
        with self.sql_conn as conn:
            conn.execute(f"DROP TABLE IF EXISTS {self.index};")
        self._create_table()

        return "Obliviated"

    def get_relevant(self, data: str, num_relevant: int = 5) -> Optional[List[Any]]:
        """
        Returns all the data in the memory that is relevant to the given data.
        Args:
            data: The data to compare to.
            num_relevant: The number of relevant data to return.

        Returns: A list of the most relevant data.
        """
        # If there are no data points in the memory, return empty list
        if self.embeddings.shape[0] == 0:
            return []

        # Create the data embedding
        embedding = get_embedding(data)
        embedding = np.expand_dims(embedding, axis=0)

        # Get the nearest neighbors
        nbrs = NearestNeighbors(
            n_neighbors=min(num_relevant, self.embeddings.shape[0]),
            algorithm="brute",
            metric="cosine",
            n_jobs=-1,
        )
        nbrs.fit(self.embeddings)

        # Get the indices of the nearest neighbors
        (scores, indices) = nbrs.kneighbors(embedding)

        # Get the data from the database
        results = []
        with self.sql_conn as conn:
            for index in indices[0]:
                res = conn.execute(
                    f"SELECT data FROM {self.index} WHERE id = ?;", (int(index),)
                )
                memory = res.fetchone()
                if memory:
                    results.append(memory)

        return results

    def get_stats(self):
        """
        Returns: The stats of the memory index.
        """
        return {
            "num_memories": self.embeddings.shape[0],
            "index": self.index,
            "embedder": self.embedder_name,
        }
