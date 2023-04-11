from pathlib import Path
import numpy as np

path1: Path
path2: str

d1 = np.DataSource(path1)
d2 = np.DataSource(path2)
d3 = np.DataSource(None)

reveal_type(d1.abspath("..."))  # E: str
reveal_type(d2.abspath("..."))  # E: str
reveal_type(d3.abspath("..."))  # E: str

reveal_type(d1.exists("..."))  # E: bool
reveal_type(d2.exists("..."))  # E: bool
reveal_type(d3.exists("..."))  # E: bool

reveal_type(d1.open("...", "r"))  # E: IO[Any]
reveal_type(d2.open("...", encoding="utf8"))  # E: IO[Any]
reveal_type(d3.open("...", newline="/n"))  # E: IO[Any]
