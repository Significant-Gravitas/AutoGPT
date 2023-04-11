import numpy as np
from typing import Any

SEED_FLOAT: float = 457.3
SEED_ARR_FLOAT: np.ndarray[Any, np.dtype[np.float64]] = np.array([1.0, 2, 3, 4])
SEED_ARRLIKE_FLOAT: list[float] = [1.0, 2.0, 3.0, 4.0]
SEED_SEED_SEQ: np.random.SeedSequence = np.random.SeedSequence(0)
SEED_STR: str = "String seeding not allowed"
# default rng
np.random.default_rng(SEED_FLOAT)  # E: incompatible type
np.random.default_rng(SEED_ARR_FLOAT)  # E: incompatible type
np.random.default_rng(SEED_ARRLIKE_FLOAT)  # E: incompatible type
np.random.default_rng(SEED_STR)  # E: incompatible type

# Seed Sequence
np.random.SeedSequence(SEED_FLOAT)  # E: incompatible type
np.random.SeedSequence(SEED_ARR_FLOAT)  # E: incompatible type
np.random.SeedSequence(SEED_ARRLIKE_FLOAT)  # E: incompatible type
np.random.SeedSequence(SEED_SEED_SEQ)  # E: incompatible type
np.random.SeedSequence(SEED_STR)  # E: incompatible type

seed_seq: np.random.bit_generator.SeedSequence = np.random.SeedSequence()
seed_seq.spawn(11.5)  # E: incompatible type
seed_seq.generate_state(3.14)  # E: incompatible type
seed_seq.generate_state(3, np.uint8)  # E: incompatible type
seed_seq.generate_state(3, "uint8")  # E: incompatible type
seed_seq.generate_state(3, "u1")  # E: incompatible type
seed_seq.generate_state(3, np.uint16)  # E: incompatible type
seed_seq.generate_state(3, "uint16")  # E: incompatible type
seed_seq.generate_state(3, "u2")  # E: incompatible type
seed_seq.generate_state(3, np.int32)  # E: incompatible type
seed_seq.generate_state(3, "int32")  # E: incompatible type
seed_seq.generate_state(3, "i4")  # E: incompatible type

# Bit Generators
np.random.MT19937(SEED_FLOAT)  # E: incompatible type
np.random.MT19937(SEED_ARR_FLOAT)  # E: incompatible type
np.random.MT19937(SEED_ARRLIKE_FLOAT)  # E: incompatible type
np.random.MT19937(SEED_STR)  # E: incompatible type

np.random.PCG64(SEED_FLOAT)  # E: incompatible type
np.random.PCG64(SEED_ARR_FLOAT)  # E: incompatible type
np.random.PCG64(SEED_ARRLIKE_FLOAT)  # E: incompatible type
np.random.PCG64(SEED_STR)  # E: incompatible type

np.random.Philox(SEED_FLOAT)  # E: incompatible type
np.random.Philox(SEED_ARR_FLOAT)  # E: incompatible type
np.random.Philox(SEED_ARRLIKE_FLOAT)  # E: incompatible type
np.random.Philox(SEED_STR)  # E: incompatible type

np.random.SFC64(SEED_FLOAT)  # E: incompatible type
np.random.SFC64(SEED_ARR_FLOAT)  # E: incompatible type
np.random.SFC64(SEED_ARRLIKE_FLOAT)  # E: incompatible type
np.random.SFC64(SEED_STR)  # E: incompatible type

# Generator
np.random.Generator(None)  # E: incompatible type
np.random.Generator(12333283902830213)  # E: incompatible type
np.random.Generator("OxFEEDF00D")  # E: incompatible type
np.random.Generator([123, 234])  # E: incompatible type
np.random.Generator(np.array([123, 234], dtype="u4"))  # E: incompatible type
