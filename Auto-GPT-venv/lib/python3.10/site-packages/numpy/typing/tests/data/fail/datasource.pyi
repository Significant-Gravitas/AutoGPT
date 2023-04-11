from pathlib import Path
import numpy as np

path: Path
d1: np.DataSource

d1.abspath(path)  # E: incompatible type
d1.abspath(b"...")  # E: incompatible type

d1.exists(path)  # E: incompatible type
d1.exists(b"...")  # E: incompatible type

d1.open(path, "r")  # E: incompatible type
d1.open(b"...", encoding="utf8")  # E: incompatible type
d1.open(None, newline="/n")  # E: incompatible type
