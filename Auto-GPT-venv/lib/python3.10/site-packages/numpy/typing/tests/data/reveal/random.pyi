from __future__ import annotations

from typing import Any

import numpy as np

def_rng = np.random.default_rng()
seed_seq = np.random.SeedSequence()
mt19937 = np.random.MT19937()
pcg64 = np.random.PCG64()
sfc64 = np.random.SFC64()
philox = np.random.Philox()
seedless_seq = np.random.bit_generator.SeedlessSeedSequence()

reveal_type(def_rng)  # E: random._generator.Generator
reveal_type(mt19937)  # E: random._mt19937.MT19937
reveal_type(pcg64)  # E: random._pcg64.PCG64
reveal_type(sfc64)  # E: random._sfc64.SFC64
reveal_type(philox)  # E: random._philox.Philox
reveal_type(seed_seq)  # E: random.bit_generator.SeedSequence
reveal_type(seedless_seq)  # E: random.bit_generator.SeedlessSeedSequence

mt19937_jumped = mt19937.jumped()
mt19937_jumped3 = mt19937.jumped(3)
mt19937_raw = mt19937.random_raw()
mt19937_raw_arr = mt19937.random_raw(5)

reveal_type(mt19937_jumped)  # E: random._mt19937.MT19937
reveal_type(mt19937_jumped3)  # E: random._mt19937.MT19937
reveal_type(mt19937_raw)  # E: int
reveal_type(mt19937_raw_arr)  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]
reveal_type(mt19937.lock)  # E: threading.Lock

pcg64_jumped = pcg64.jumped()
pcg64_jumped3 = pcg64.jumped(3)
pcg64_adv = pcg64.advance(3)
pcg64_raw = pcg64.random_raw()
pcg64_raw_arr = pcg64.random_raw(5)

reveal_type(pcg64_jumped)  # E: random._pcg64.PCG64
reveal_type(pcg64_jumped3)  # E: random._pcg64.PCG64
reveal_type(pcg64_adv)  # E: random._pcg64.PCG64
reveal_type(pcg64_raw)  # E: int
reveal_type(pcg64_raw_arr)  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]
reveal_type(pcg64.lock)  # E: threading.Lock

philox_jumped = philox.jumped()
philox_jumped3 = philox.jumped(3)
philox_adv = philox.advance(3)
philox_raw = philox.random_raw()
philox_raw_arr = philox.random_raw(5)

reveal_type(philox_jumped)  # E: random._philox.Philox
reveal_type(philox_jumped3)  # E: random._philox.Philox
reveal_type(philox_adv)  # E: random._philox.Philox
reveal_type(philox_raw)  # E: int
reveal_type(philox_raw_arr)  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]
reveal_type(philox.lock)  # E: threading.Lock

sfc64_raw = sfc64.random_raw()
sfc64_raw_arr = sfc64.random_raw(5)

reveal_type(sfc64_raw)  # E: int
reveal_type(sfc64_raw_arr)  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]
reveal_type(sfc64.lock)  # E: threading.Lock

reveal_type(seed_seq.pool)  # ndarray[Any, dtype[unsignedinteger[typing._32Bit]]]
reveal_type(seed_seq.entropy)  # E:Union[None, int, Sequence[int]]
reveal_type(seed_seq.spawn(1))  # E: list[random.bit_generator.SeedSequence]
reveal_type(seed_seq.generate_state(8, "uint32"))  # E: ndarray[Any, dtype[Union[unsignedinteger[typing._32Bit], unsignedinteger[typing._64Bit]]]]
reveal_type(seed_seq.generate_state(8, "uint64"))  # E: ndarray[Any, dtype[Union[unsignedinteger[typing._32Bit], unsignedinteger[typing._64Bit]]]]


def_gen: np.random.Generator = np.random.default_rng()

D_arr_0p1: np.ndarray[Any, np.dtype[np.float64]] = np.array([0.1])
D_arr_0p5: np.ndarray[Any, np.dtype[np.float64]] = np.array([0.5])
D_arr_0p9: np.ndarray[Any, np.dtype[np.float64]] = np.array([0.9])
D_arr_1p5: np.ndarray[Any, np.dtype[np.float64]] = np.array([1.5])
I_arr_10: np.ndarray[Any, np.dtype[np.int_]] = np.array([10], dtype=np.int_)
I_arr_20: np.ndarray[Any, np.dtype[np.int_]] = np.array([20], dtype=np.int_)
D_arr_like_0p1: list[float] = [0.1]
D_arr_like_0p5: list[float] = [0.5]
D_arr_like_0p9: list[float] = [0.9]
D_arr_like_1p5: list[float] = [1.5]
I_arr_like_10: list[int] = [10]
I_arr_like_20: list[int] = [20]
D_2D_like: list[list[float]] = [[1, 2], [2, 3], [3, 4], [4, 5.1]]
D_2D: np.ndarray[Any, np.dtype[np.float64]] = np.array(D_2D_like)
S_out: np.ndarray[Any, np.dtype[np.float32]] = np.empty(1, dtype=np.float32)
D_out: np.ndarray[Any, np.dtype[np.float64]] = np.empty(1)

reveal_type(def_gen.standard_normal())  # E: float
reveal_type(def_gen.standard_normal(dtype=np.float32))  # E: float
reveal_type(def_gen.standard_normal(dtype="float32"))  # E: float
reveal_type(def_gen.standard_normal(dtype="double"))  # E: float
reveal_type(def_gen.standard_normal(dtype=np.float64))  # E: float
reveal_type(def_gen.standard_normal(size=None))  # E: float
reveal_type(def_gen.standard_normal(size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]
reveal_type(def_gen.standard_normal(size=1, dtype=np.float32))  # E: ndarray[Any, dtype[floating[typing._32Bit]]]
reveal_type(def_gen.standard_normal(size=1, dtype="f4"))  # E: ndarray[Any, dtype[floating[typing._32Bit]]]
reveal_type(def_gen.standard_normal(size=1, dtype="float32", out=S_out))  # E: ndarray[Any, dtype[floating[typing._32Bit]]]
reveal_type(def_gen.standard_normal(dtype=np.float32, out=S_out))  # E: ndarray[Any, dtype[floating[typing._32Bit]]]
reveal_type(def_gen.standard_normal(size=1, dtype=np.float64))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]
reveal_type(def_gen.standard_normal(size=1, dtype="float64"))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]
reveal_type(def_gen.standard_normal(size=1, dtype="f8"))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]
reveal_type(def_gen.standard_normal(out=D_out))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]
reveal_type(def_gen.standard_normal(size=1, dtype="float64"))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]
reveal_type(def_gen.standard_normal(size=1, dtype="float64", out=D_out))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]

reveal_type(def_gen.random())  # E: float
reveal_type(def_gen.random(dtype=np.float32))  # E: float
reveal_type(def_gen.random(dtype="float32"))  # E: float
reveal_type(def_gen.random(dtype="double"))  # E: float
reveal_type(def_gen.random(dtype=np.float64))  # E: float
reveal_type(def_gen.random(size=None))  # E: float
reveal_type(def_gen.random(size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]
reveal_type(def_gen.random(size=1, dtype=np.float32))  # E: ndarray[Any, dtype[floating[typing._32Bit]]]
reveal_type(def_gen.random(size=1, dtype="f4"))  # E: ndarray[Any, dtype[floating[typing._32Bit]]]
reveal_type(def_gen.random(size=1, dtype="float32", out=S_out))  # E: ndarray[Any, dtype[floating[typing._32Bit]]]
reveal_type(def_gen.random(dtype=np.float32, out=S_out))  # E: ndarray[Any, dtype[floating[typing._32Bit]]]
reveal_type(def_gen.random(size=1, dtype=np.float64))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]
reveal_type(def_gen.random(size=1, dtype="float64"))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]
reveal_type(def_gen.random(size=1, dtype="f8"))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]
reveal_type(def_gen.random(out=D_out))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]
reveal_type(def_gen.random(size=1, dtype="float64"))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]
reveal_type(def_gen.random(size=1, dtype="float64", out=D_out))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]

reveal_type(def_gen.standard_cauchy())  # E: float
reveal_type(def_gen.standard_cauchy(size=None))  # E: float
reveal_type(def_gen.standard_cauchy(size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(def_gen.standard_exponential())  # E: float
reveal_type(def_gen.standard_exponential(method="inv"))  # E: float
reveal_type(def_gen.standard_exponential(dtype=np.float32))  # E: float
reveal_type(def_gen.standard_exponential(dtype="float32"))  # E: float
reveal_type(def_gen.standard_exponential(dtype="double"))  # E: float
reveal_type(def_gen.standard_exponential(dtype=np.float64))  # E: float
reveal_type(def_gen.standard_exponential(size=None))  # E: float
reveal_type(def_gen.standard_exponential(size=None, method="inv"))  # E: float
reveal_type(def_gen.standard_exponential(size=1, method="inv"))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]
reveal_type(def_gen.standard_exponential(size=1, dtype=np.float32))  # E: ndarray[Any, dtype[floating[typing._32Bit]]]
reveal_type(def_gen.standard_exponential(size=1, dtype="f4", method="inv"))  # E: ndarray[Any, dtype[floating[typing._32Bit]]]
reveal_type(def_gen.standard_exponential(size=1, dtype="float32", out=S_out))  # E: ndarray[Any, dtype[floating[typing._32Bit]]]
reveal_type(def_gen.standard_exponential(dtype=np.float32, out=S_out))  # E: ndarray[Any, dtype[floating[typing._32Bit]]]
reveal_type(def_gen.standard_exponential(size=1, dtype=np.float64, method="inv"))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]
reveal_type(def_gen.standard_exponential(size=1, dtype="float64"))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]
reveal_type(def_gen.standard_exponential(size=1, dtype="f8"))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]
reveal_type(def_gen.standard_exponential(out=D_out))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]
reveal_type(def_gen.standard_exponential(size=1, dtype="float64"))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]
reveal_type(def_gen.standard_exponential(size=1, dtype="float64", out=D_out))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]

reveal_type(def_gen.zipf(1.5))  # E: int
reveal_type(def_gen.zipf(1.5, size=None))  # E: int
reveal_type(def_gen.zipf(1.5, size=1))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.zipf(D_arr_1p5))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.zipf(D_arr_1p5, size=1))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.zipf(D_arr_like_1p5))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.zipf(D_arr_like_1p5, size=1))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]

reveal_type(def_gen.weibull(0.5))  # E: float
reveal_type(def_gen.weibull(0.5, size=None))  # E: float
reveal_type(def_gen.weibull(0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.weibull(D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.weibull(D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.weibull(D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.weibull(D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(def_gen.standard_t(0.5))  # E: float
reveal_type(def_gen.standard_t(0.5, size=None))  # E: float
reveal_type(def_gen.standard_t(0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.standard_t(D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.standard_t(D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.standard_t(D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.standard_t(D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(def_gen.poisson(0.5))  # E: int
reveal_type(def_gen.poisson(0.5, size=None))  # E: int
reveal_type(def_gen.poisson(0.5, size=1))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.poisson(D_arr_0p5))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.poisson(D_arr_0p5, size=1))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.poisson(D_arr_like_0p5))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.poisson(D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]

reveal_type(def_gen.power(0.5))  # E: float
reveal_type(def_gen.power(0.5, size=None))  # E: float
reveal_type(def_gen.power(0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.power(D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.power(D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.power(D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.power(D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(def_gen.pareto(0.5))  # E: float
reveal_type(def_gen.pareto(0.5, size=None))  # E: float
reveal_type(def_gen.pareto(0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.pareto(D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.pareto(D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.pareto(D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.pareto(D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(def_gen.chisquare(0.5))  # E: float
reveal_type(def_gen.chisquare(0.5, size=None))  # E: float
reveal_type(def_gen.chisquare(0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.chisquare(D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.chisquare(D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.chisquare(D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.chisquare(D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(def_gen.exponential(0.5))  # E: float
reveal_type(def_gen.exponential(0.5, size=None))  # E: float
reveal_type(def_gen.exponential(0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.exponential(D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.exponential(D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.exponential(D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.exponential(D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(def_gen.geometric(0.5))  # E: int
reveal_type(def_gen.geometric(0.5, size=None))  # E: int
reveal_type(def_gen.geometric(0.5, size=1))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.geometric(D_arr_0p5))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.geometric(D_arr_0p5, size=1))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.geometric(D_arr_like_0p5))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.geometric(D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]

reveal_type(def_gen.logseries(0.5))  # E: int
reveal_type(def_gen.logseries(0.5, size=None))  # E: int
reveal_type(def_gen.logseries(0.5, size=1))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.logseries(D_arr_0p5))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.logseries(D_arr_0p5, size=1))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.logseries(D_arr_like_0p5))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.logseries(D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]

reveal_type(def_gen.rayleigh(0.5))  # E: float
reveal_type(def_gen.rayleigh(0.5, size=None))  # E: float
reveal_type(def_gen.rayleigh(0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.rayleigh(D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.rayleigh(D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.rayleigh(D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.rayleigh(D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(def_gen.standard_gamma(0.5))  # E: float
reveal_type(def_gen.standard_gamma(0.5, size=None))  # E: float
reveal_type(def_gen.standard_gamma(0.5, dtype="float32"))  # E: float
reveal_type(def_gen.standard_gamma(0.5, size=None, dtype="float32"))  # E: float
reveal_type(def_gen.standard_gamma(0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]
reveal_type(def_gen.standard_gamma(D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]
reveal_type(def_gen.standard_gamma(D_arr_0p5, dtype="f4"))  # E: ndarray[Any, dtype[floating[typing._32Bit]]]
reveal_type(def_gen.standard_gamma(0.5, size=1, dtype="float32", out=S_out))  # E: ndarray[Any, dtype[floating[typing._32Bit]]]
reveal_type(def_gen.standard_gamma(D_arr_0p5, dtype=np.float32, out=S_out))  # E: ndarray[Any, dtype[floating[typing._32Bit]]]
reveal_type(def_gen.standard_gamma(D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]
reveal_type(def_gen.standard_gamma(D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]
reveal_type(def_gen.standard_gamma(D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]
reveal_type(def_gen.standard_gamma(0.5, out=D_out))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]
reveal_type(def_gen.standard_gamma(D_arr_like_0p5, out=D_out))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]
reveal_type(def_gen.standard_gamma(D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]
reveal_type(def_gen.standard_gamma(D_arr_like_0p5, size=1, out=D_out, dtype=np.float64))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]

reveal_type(def_gen.vonmises(0.5, 0.5))  # E: float
reveal_type(def_gen.vonmises(0.5, 0.5, size=None))  # E: float
reveal_type(def_gen.vonmises(0.5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.vonmises(D_arr_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.vonmises(0.5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.vonmises(D_arr_0p5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.vonmises(0.5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.vonmises(D_arr_like_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.vonmises(0.5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.vonmises(D_arr_0p5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.vonmises(D_arr_like_0p5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.vonmises(D_arr_0p5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.vonmises(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(def_gen.wald(0.5, 0.5))  # E: float
reveal_type(def_gen.wald(0.5, 0.5, size=None))  # E: float
reveal_type(def_gen.wald(0.5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.wald(D_arr_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.wald(0.5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.wald(D_arr_0p5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.wald(0.5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.wald(D_arr_like_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.wald(0.5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.wald(D_arr_0p5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.wald(D_arr_like_0p5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.wald(D_arr_0p5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.wald(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(def_gen.uniform(0.5, 0.5))  # E: float
reveal_type(def_gen.uniform(0.5, 0.5, size=None))  # E: float
reveal_type(def_gen.uniform(0.5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.uniform(D_arr_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.uniform(0.5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.uniform(D_arr_0p5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.uniform(0.5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.uniform(D_arr_like_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.uniform(0.5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.uniform(D_arr_0p5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.uniform(D_arr_like_0p5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.uniform(D_arr_0p5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.uniform(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(def_gen.beta(0.5, 0.5))  # E: float
reveal_type(def_gen.beta(0.5, 0.5, size=None))  # E: float
reveal_type(def_gen.beta(0.5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.beta(D_arr_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.beta(0.5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.beta(D_arr_0p5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.beta(0.5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.beta(D_arr_like_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.beta(0.5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.beta(D_arr_0p5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.beta(D_arr_like_0p5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.beta(D_arr_0p5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.beta(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(def_gen.f(0.5, 0.5))  # E: float
reveal_type(def_gen.f(0.5, 0.5, size=None))  # E: float
reveal_type(def_gen.f(0.5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.f(D_arr_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.f(0.5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.f(D_arr_0p5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.f(0.5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.f(D_arr_like_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.f(0.5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.f(D_arr_0p5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.f(D_arr_like_0p5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.f(D_arr_0p5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.f(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(def_gen.gamma(0.5, 0.5))  # E: float
reveal_type(def_gen.gamma(0.5, 0.5, size=None))  # E: float
reveal_type(def_gen.gamma(0.5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.gamma(D_arr_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.gamma(0.5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.gamma(D_arr_0p5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.gamma(0.5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.gamma(D_arr_like_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.gamma(0.5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.gamma(D_arr_0p5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.gamma(D_arr_like_0p5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.gamma(D_arr_0p5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.gamma(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(def_gen.gumbel(0.5, 0.5))  # E: float
reveal_type(def_gen.gumbel(0.5, 0.5, size=None))  # E: float
reveal_type(def_gen.gumbel(0.5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.gumbel(D_arr_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.gumbel(0.5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.gumbel(D_arr_0p5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.gumbel(0.5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.gumbel(D_arr_like_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.gumbel(0.5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.gumbel(D_arr_0p5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.gumbel(D_arr_like_0p5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.gumbel(D_arr_0p5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.gumbel(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(def_gen.laplace(0.5, 0.5))  # E: float
reveal_type(def_gen.laplace(0.5, 0.5, size=None))  # E: float
reveal_type(def_gen.laplace(0.5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.laplace(D_arr_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.laplace(0.5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.laplace(D_arr_0p5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.laplace(0.5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.laplace(D_arr_like_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.laplace(0.5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.laplace(D_arr_0p5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.laplace(D_arr_like_0p5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.laplace(D_arr_0p5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.laplace(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(def_gen.logistic(0.5, 0.5))  # E: float
reveal_type(def_gen.logistic(0.5, 0.5, size=None))  # E: float
reveal_type(def_gen.logistic(0.5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.logistic(D_arr_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.logistic(0.5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.logistic(D_arr_0p5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.logistic(0.5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.logistic(D_arr_like_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.logistic(0.5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.logistic(D_arr_0p5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.logistic(D_arr_like_0p5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.logistic(D_arr_0p5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.logistic(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(def_gen.lognormal(0.5, 0.5))  # E: float
reveal_type(def_gen.lognormal(0.5, 0.5, size=None))  # E: float
reveal_type(def_gen.lognormal(0.5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.lognormal(D_arr_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.lognormal(0.5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.lognormal(D_arr_0p5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.lognormal(0.5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.lognormal(D_arr_like_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.lognormal(0.5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.lognormal(D_arr_0p5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.lognormal(D_arr_like_0p5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.lognormal(D_arr_0p5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.lognormal(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(def_gen.noncentral_chisquare(0.5, 0.5))  # E: float
reveal_type(def_gen.noncentral_chisquare(0.5, 0.5, size=None))  # E: float
reveal_type(def_gen.noncentral_chisquare(0.5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.noncentral_chisquare(D_arr_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.noncentral_chisquare(0.5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.noncentral_chisquare(D_arr_0p5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.noncentral_chisquare(0.5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.noncentral_chisquare(D_arr_like_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.noncentral_chisquare(0.5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.noncentral_chisquare(D_arr_0p5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.noncentral_chisquare(D_arr_like_0p5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.noncentral_chisquare(D_arr_0p5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.noncentral_chisquare(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(def_gen.normal(0.5, 0.5))  # E: float
reveal_type(def_gen.normal(0.5, 0.5, size=None))  # E: float
reveal_type(def_gen.normal(0.5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.normal(D_arr_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.normal(0.5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.normal(D_arr_0p5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.normal(0.5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.normal(D_arr_like_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.normal(0.5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.normal(D_arr_0p5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.normal(D_arr_like_0p5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.normal(D_arr_0p5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.normal(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(def_gen.triangular(0.1, 0.5, 0.9))  # E: float
reveal_type(def_gen.triangular(0.1, 0.5, 0.9, size=None))  # E: float
reveal_type(def_gen.triangular(0.1, 0.5, 0.9, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.triangular(D_arr_0p1, 0.5, 0.9))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.triangular(0.1, D_arr_0p5, 0.9))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.triangular(D_arr_0p1, 0.5, D_arr_like_0p9, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.triangular(0.1, D_arr_0p5, 0.9, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.triangular(D_arr_like_0p1, 0.5, D_arr_0p9))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.triangular(0.5, D_arr_like_0p5, 0.9))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.triangular(D_arr_0p1, D_arr_0p5, 0.9))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.triangular(D_arr_like_0p1, D_arr_like_0p5, 0.9))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.triangular(D_arr_0p1, D_arr_0p5, D_arr_0p9, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.triangular(D_arr_like_0p1, D_arr_like_0p5, D_arr_like_0p9, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(def_gen.noncentral_f(0.1, 0.5, 0.9))  # E: float
reveal_type(def_gen.noncentral_f(0.1, 0.5, 0.9, size=None))  # E: float
reveal_type(def_gen.noncentral_f(0.1, 0.5, 0.9, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.noncentral_f(D_arr_0p1, 0.5, 0.9))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.noncentral_f(0.1, D_arr_0p5, 0.9))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.noncentral_f(D_arr_0p1, 0.5, D_arr_like_0p9, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.noncentral_f(0.1, D_arr_0p5, 0.9, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.noncentral_f(D_arr_like_0p1, 0.5, D_arr_0p9))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.noncentral_f(0.5, D_arr_like_0p5, 0.9))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.noncentral_f(D_arr_0p1, D_arr_0p5, 0.9))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.noncentral_f(D_arr_like_0p1, D_arr_like_0p5, 0.9))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.noncentral_f(D_arr_0p1, D_arr_0p5, D_arr_0p9, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.noncentral_f(D_arr_like_0p1, D_arr_like_0p5, D_arr_like_0p9, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(def_gen.binomial(10, 0.5))  # E: int
reveal_type(def_gen.binomial(10, 0.5, size=None))  # E: int
reveal_type(def_gen.binomial(10, 0.5, size=1))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.binomial(I_arr_10, 0.5))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.binomial(10, D_arr_0p5))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.binomial(I_arr_10, 0.5, size=1))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.binomial(10, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.binomial(I_arr_like_10, 0.5))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.binomial(10, D_arr_like_0p5))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.binomial(I_arr_10, D_arr_0p5))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.binomial(I_arr_like_10, D_arr_like_0p5))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.binomial(I_arr_10, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.binomial(I_arr_like_10, D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]

reveal_type(def_gen.negative_binomial(10, 0.5))  # E: int
reveal_type(def_gen.negative_binomial(10, 0.5, size=None))  # E: int
reveal_type(def_gen.negative_binomial(10, 0.5, size=1))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.negative_binomial(I_arr_10, 0.5))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.negative_binomial(10, D_arr_0p5))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.negative_binomial(I_arr_10, 0.5, size=1))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.negative_binomial(10, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.negative_binomial(I_arr_like_10, 0.5))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.negative_binomial(10, D_arr_like_0p5))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.negative_binomial(I_arr_10, D_arr_0p5))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.negative_binomial(I_arr_like_10, D_arr_like_0p5))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.negative_binomial(I_arr_10, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.negative_binomial(I_arr_like_10, D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]

reveal_type(def_gen.hypergeometric(20, 20, 10))  # E: int
reveal_type(def_gen.hypergeometric(20, 20, 10, size=None))  # E: int
reveal_type(def_gen.hypergeometric(20, 20, 10, size=1))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.hypergeometric(I_arr_20, 20, 10))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.hypergeometric(20, I_arr_20, 10))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.hypergeometric(I_arr_20, 20, I_arr_like_10, size=1))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.hypergeometric(20, I_arr_20, 10, size=1))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.hypergeometric(I_arr_like_20, 20, I_arr_10))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.hypergeometric(20, I_arr_like_20, 10))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.hypergeometric(I_arr_20, I_arr_20, 10))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.hypergeometric(I_arr_like_20, I_arr_like_20, 10))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.hypergeometric(I_arr_20, I_arr_20, I_arr_10, size=1))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.hypergeometric(I_arr_like_20, I_arr_like_20, I_arr_like_10, size=1))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]

I_int64_100: np.ndarray[Any, np.dtype[np.int64]] = np.array([100], dtype=np.int64)

reveal_type(def_gen.integers(0, 100))  # E: int
reveal_type(def_gen.integers(100))  # E: int
reveal_type(def_gen.integers([100]))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(0, [100]))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]

I_bool_low: np.ndarray[Any, np.dtype[np.bool_]] = np.array([0], dtype=np.bool_)
I_bool_low_like: list[int] = [0]
I_bool_high_open: np.ndarray[Any, np.dtype[np.bool_]] = np.array([1], dtype=np.bool_)
I_bool_high_closed: np.ndarray[Any, np.dtype[np.bool_]] = np.array([1], dtype=np.bool_)

reveal_type(def_gen.integers(2, dtype=bool))  # E: builtins.bool
reveal_type(def_gen.integers(0, 2, dtype=bool))  # E: builtins.bool
reveal_type(def_gen.integers(1, dtype=bool, endpoint=True))  # E: builtins.bool
reveal_type(def_gen.integers(0, 1, dtype=bool, endpoint=True))  # E: builtins.bool
reveal_type(def_gen.integers(I_bool_low_like, 1, dtype=bool, endpoint=True))  # E: ndarray[Any, dtype[bool_]
reveal_type(def_gen.integers(I_bool_high_open, dtype=bool))  # E: ndarray[Any, dtype[bool_]
reveal_type(def_gen.integers(I_bool_low, I_bool_high_open, dtype=bool))  # E: ndarray[Any, dtype[bool_]
reveal_type(def_gen.integers(0, I_bool_high_open, dtype=bool))  # E: ndarray[Any, dtype[bool_]
reveal_type(def_gen.integers(I_bool_high_closed, dtype=bool, endpoint=True))  # E: ndarray[Any, dtype[bool_]
reveal_type(def_gen.integers(I_bool_low, I_bool_high_closed, dtype=bool, endpoint=True))  # E: ndarray[Any, dtype[bool_]
reveal_type(def_gen.integers(0, I_bool_high_closed, dtype=bool, endpoint=True))  # E: ndarray[Any, dtype[bool_]

reveal_type(def_gen.integers(2, dtype=np.bool_))  # E: builtins.bool
reveal_type(def_gen.integers(0, 2, dtype=np.bool_))  # E: builtins.bool
reveal_type(def_gen.integers(1, dtype=np.bool_, endpoint=True))  # E: builtins.bool
reveal_type(def_gen.integers(0, 1, dtype=np.bool_, endpoint=True))  # E: builtins.bool
reveal_type(def_gen.integers(I_bool_low_like, 1, dtype=np.bool_, endpoint=True))  # E: ndarray[Any, dtype[bool_]
reveal_type(def_gen.integers(I_bool_high_open, dtype=np.bool_))  # E: ndarray[Any, dtype[bool_]
reveal_type(def_gen.integers(I_bool_low, I_bool_high_open, dtype=np.bool_))  # E: ndarray[Any, dtype[bool_]
reveal_type(def_gen.integers(0, I_bool_high_open, dtype=np.bool_))  # E: ndarray[Any, dtype[bool_]
reveal_type(def_gen.integers(I_bool_high_closed, dtype=np.bool_, endpoint=True))  # E: ndarray[Any, dtype[bool_]
reveal_type(def_gen.integers(I_bool_low, I_bool_high_closed, dtype=np.bool_, endpoint=True))  # E: ndarray[Any, dtype[bool_]
reveal_type(def_gen.integers(0, I_bool_high_closed, dtype=np.bool_, endpoint=True))  # E: ndarray[Any, dtype[bool_]

I_u1_low: np.ndarray[Any, np.dtype[np.uint8]] = np.array([0], dtype=np.uint8)
I_u1_low_like: list[int] = [0]
I_u1_high_open: np.ndarray[Any, np.dtype[np.uint8]] = np.array([255], dtype=np.uint8)
I_u1_high_closed: np.ndarray[Any, np.dtype[np.uint8]] = np.array([255], dtype=np.uint8)

reveal_type(def_gen.integers(256, dtype="u1"))  # E: int
reveal_type(def_gen.integers(0, 256, dtype="u1"))  # E: int
reveal_type(def_gen.integers(255, dtype="u1", endpoint=True))  # E: int
reveal_type(def_gen.integers(0, 255, dtype="u1", endpoint=True))  # E: int
reveal_type(def_gen.integers(I_u1_low_like, 255, dtype="u1", endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(I_u1_high_open, dtype="u1"))  # E: ndarray[Any, dtype[unsignedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(I_u1_low, I_u1_high_open, dtype="u1"))  # E: ndarray[Any, dtype[unsignedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(0, I_u1_high_open, dtype="u1"))  # E: ndarray[Any, dtype[unsignedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(I_u1_high_closed, dtype="u1", endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(I_u1_low, I_u1_high_closed, dtype="u1", endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(0, I_u1_high_closed, dtype="u1", endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._8Bit]]]

reveal_type(def_gen.integers(256, dtype="uint8"))  # E: int
reveal_type(def_gen.integers(0, 256, dtype="uint8"))  # E: int
reveal_type(def_gen.integers(255, dtype="uint8", endpoint=True))  # E: int
reveal_type(def_gen.integers(0, 255, dtype="uint8", endpoint=True))  # E: int
reveal_type(def_gen.integers(I_u1_low_like, 255, dtype="uint8", endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(I_u1_high_open, dtype="uint8"))  # E: ndarray[Any, dtype[unsignedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(I_u1_low, I_u1_high_open, dtype="uint8"))  # E: ndarray[Any, dtype[unsignedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(0, I_u1_high_open, dtype="uint8"))  # E: ndarray[Any, dtype[unsignedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(I_u1_high_closed, dtype="uint8", endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(I_u1_low, I_u1_high_closed, dtype="uint8", endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(0, I_u1_high_closed, dtype="uint8", endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._8Bit]]]

reveal_type(def_gen.integers(256, dtype=np.uint8))  # E: int
reveal_type(def_gen.integers(0, 256, dtype=np.uint8))  # E: int
reveal_type(def_gen.integers(255, dtype=np.uint8, endpoint=True))  # E: int
reveal_type(def_gen.integers(0, 255, dtype=np.uint8, endpoint=True))  # E: int
reveal_type(def_gen.integers(I_u1_low_like, 255, dtype=np.uint8, endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(I_u1_high_open, dtype=np.uint8))  # E: ndarray[Any, dtype[unsignedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(I_u1_low, I_u1_high_open, dtype=np.uint8))  # E: ndarray[Any, dtype[unsignedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(0, I_u1_high_open, dtype=np.uint8))  # E: ndarray[Any, dtype[unsignedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(I_u1_high_closed, dtype=np.uint8, endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(I_u1_low, I_u1_high_closed, dtype=np.uint8, endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(0, I_u1_high_closed, dtype=np.uint8, endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._8Bit]]]

I_u2_low: np.ndarray[Any, np.dtype[np.uint16]] = np.array([0], dtype=np.uint16)
I_u2_low_like: list[int] = [0]
I_u2_high_open: np.ndarray[Any, np.dtype[np.uint16]] = np.array([65535], dtype=np.uint16)
I_u2_high_closed: np.ndarray[Any, np.dtype[np.uint16]] = np.array([65535], dtype=np.uint16)

reveal_type(def_gen.integers(65536, dtype="u2"))  # E: int
reveal_type(def_gen.integers(0, 65536, dtype="u2"))  # E: int
reveal_type(def_gen.integers(65535, dtype="u2", endpoint=True))  # E: int
reveal_type(def_gen.integers(0, 65535, dtype="u2", endpoint=True))  # E: int
reveal_type(def_gen.integers(I_u2_low_like, 65535, dtype="u2", endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(I_u2_high_open, dtype="u2"))  # E: ndarray[Any, dtype[unsignedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(I_u2_low, I_u2_high_open, dtype="u2"))  # E: ndarray[Any, dtype[unsignedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(0, I_u2_high_open, dtype="u2"))  # E: ndarray[Any, dtype[unsignedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(I_u2_high_closed, dtype="u2", endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(I_u2_low, I_u2_high_closed, dtype="u2", endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(0, I_u2_high_closed, dtype="u2", endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._16Bit]]]

reveal_type(def_gen.integers(65536, dtype="uint16"))  # E: int
reveal_type(def_gen.integers(0, 65536, dtype="uint16"))  # E: int
reveal_type(def_gen.integers(65535, dtype="uint16", endpoint=True))  # E: int
reveal_type(def_gen.integers(0, 65535, dtype="uint16", endpoint=True))  # E: int
reveal_type(def_gen.integers(I_u2_low_like, 65535, dtype="uint16", endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(I_u2_high_open, dtype="uint16"))  # E: ndarray[Any, dtype[unsignedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(I_u2_low, I_u2_high_open, dtype="uint16"))  # E: ndarray[Any, dtype[unsignedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(0, I_u2_high_open, dtype="uint16"))  # E: ndarray[Any, dtype[unsignedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(I_u2_high_closed, dtype="uint16", endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(I_u2_low, I_u2_high_closed, dtype="uint16", endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(0, I_u2_high_closed, dtype="uint16", endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._16Bit]]]

reveal_type(def_gen.integers(65536, dtype=np.uint16))  # E: int
reveal_type(def_gen.integers(0, 65536, dtype=np.uint16))  # E: int
reveal_type(def_gen.integers(65535, dtype=np.uint16, endpoint=True))  # E: int
reveal_type(def_gen.integers(0, 65535, dtype=np.uint16, endpoint=True))  # E: int
reveal_type(def_gen.integers(I_u2_low_like, 65535, dtype=np.uint16, endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(I_u2_high_open, dtype=np.uint16))  # E: ndarray[Any, dtype[unsignedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(I_u2_low, I_u2_high_open, dtype=np.uint16))  # E: ndarray[Any, dtype[unsignedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(0, I_u2_high_open, dtype=np.uint16))  # E: ndarray[Any, dtype[unsignedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(I_u2_high_closed, dtype=np.uint16, endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(I_u2_low, I_u2_high_closed, dtype=np.uint16, endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(0, I_u2_high_closed, dtype=np.uint16, endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._16Bit]]]

I_u4_low: np.ndarray[Any, np.dtype[np.uint32]] = np.array([0], dtype=np.uint32)
I_u4_low_like: list[int] = [0]
I_u4_high_open: np.ndarray[Any, np.dtype[np.uint32]] = np.array([4294967295], dtype=np.uint32)
I_u4_high_closed: np.ndarray[Any, np.dtype[np.uint32]] = np.array([4294967295], dtype=np.uint32)

reveal_type(def_gen.integers(4294967296, dtype=np.int_))  # E: int
reveal_type(def_gen.integers(0, 4294967296, dtype=np.int_))  # E: int
reveal_type(def_gen.integers(4294967295, dtype=np.int_, endpoint=True))  # E: int
reveal_type(def_gen.integers(0, 4294967295, dtype=np.int_, endpoint=True))  # E: int
reveal_type(def_gen.integers(I_u4_low_like, 4294967295, dtype=np.int_, endpoint=True))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(def_gen.integers(I_u4_high_open, dtype=np.int_))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(def_gen.integers(I_u4_low, I_u4_high_open, dtype=np.int_))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(def_gen.integers(0, I_u4_high_open, dtype=np.int_))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(def_gen.integers(I_u4_high_closed, dtype=np.int_, endpoint=True))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(def_gen.integers(I_u4_low, I_u4_high_closed, dtype=np.int_, endpoint=True))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(def_gen.integers(0, I_u4_high_closed, dtype=np.int_, endpoint=True))  # E: ndarray[Any, dtype[{int_}]]


reveal_type(def_gen.integers(4294967296, dtype="u4"))  # E: int
reveal_type(def_gen.integers(0, 4294967296, dtype="u4"))  # E: int
reveal_type(def_gen.integers(4294967295, dtype="u4", endpoint=True))  # E: int
reveal_type(def_gen.integers(0, 4294967295, dtype="u4", endpoint=True))  # E: int
reveal_type(def_gen.integers(I_u4_low_like, 4294967295, dtype="u4", endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(I_u4_high_open, dtype="u4"))  # E: ndarray[Any, dtype[unsignedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(I_u4_low, I_u4_high_open, dtype="u4"))  # E: ndarray[Any, dtype[unsignedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(0, I_u4_high_open, dtype="u4"))  # E: ndarray[Any, dtype[unsignedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(I_u4_high_closed, dtype="u4", endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(I_u4_low, I_u4_high_closed, dtype="u4", endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(0, I_u4_high_closed, dtype="u4", endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._32Bit]]]

reveal_type(def_gen.integers(4294967296, dtype="uint32"))  # E: int
reveal_type(def_gen.integers(0, 4294967296, dtype="uint32"))  # E: int
reveal_type(def_gen.integers(4294967295, dtype="uint32", endpoint=True))  # E: int
reveal_type(def_gen.integers(0, 4294967295, dtype="uint32", endpoint=True))  # E: int
reveal_type(def_gen.integers(I_u4_low_like, 4294967295, dtype="uint32", endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(I_u4_high_open, dtype="uint32"))  # E: ndarray[Any, dtype[unsignedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(I_u4_low, I_u4_high_open, dtype="uint32"))  # E: ndarray[Any, dtype[unsignedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(0, I_u4_high_open, dtype="uint32"))  # E: ndarray[Any, dtype[unsignedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(I_u4_high_closed, dtype="uint32", endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(I_u4_low, I_u4_high_closed, dtype="uint32", endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(0, I_u4_high_closed, dtype="uint32", endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._32Bit]]]

reveal_type(def_gen.integers(4294967296, dtype=np.uint32))  # E: int
reveal_type(def_gen.integers(0, 4294967296, dtype=np.uint32))  # E: int
reveal_type(def_gen.integers(4294967295, dtype=np.uint32, endpoint=True))  # E: int
reveal_type(def_gen.integers(0, 4294967295, dtype=np.uint32, endpoint=True))  # E: int
reveal_type(def_gen.integers(I_u4_low_like, 4294967295, dtype=np.uint32, endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(I_u4_high_open, dtype=np.uint32))  # E: ndarray[Any, dtype[unsignedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(I_u4_low, I_u4_high_open, dtype=np.uint32))  # E: ndarray[Any, dtype[unsignedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(0, I_u4_high_open, dtype=np.uint32))  # E: ndarray[Any, dtype[unsignedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(I_u4_high_closed, dtype=np.uint32, endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(I_u4_low, I_u4_high_closed, dtype=np.uint32, endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(0, I_u4_high_closed, dtype=np.uint32, endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._32Bit]]]

reveal_type(def_gen.integers(4294967296, dtype=np.uint))  # E: int
reveal_type(def_gen.integers(0, 4294967296, dtype=np.uint))  # E: int
reveal_type(def_gen.integers(4294967295, dtype=np.uint, endpoint=True))  # E: int
reveal_type(def_gen.integers(0, 4294967295, dtype=np.uint, endpoint=True))  # E: int
reveal_type(def_gen.integers(I_u4_low_like, 4294967295, dtype=np.uint, endpoint=True))  # E: ndarray[Any, dtype[{uint}]]
reveal_type(def_gen.integers(I_u4_high_open, dtype=np.uint))  # E: ndarray[Any, dtype[{uint}]]
reveal_type(def_gen.integers(I_u4_low, I_u4_high_open, dtype=np.uint))  # E: ndarray[Any, dtype[{uint}]]
reveal_type(def_gen.integers(0, I_u4_high_open, dtype=np.uint))  # E: ndarray[Any, dtype[{uint}]]
reveal_type(def_gen.integers(I_u4_high_closed, dtype=np.uint, endpoint=True))  # E: ndarray[Any, dtype[{uint}]]
reveal_type(def_gen.integers(I_u4_low, I_u4_high_closed, dtype=np.uint, endpoint=True))  # E: ndarray[Any, dtype[{uint}]]
reveal_type(def_gen.integers(0, I_u4_high_closed, dtype=np.uint, endpoint=True))  # E: ndarray[Any, dtype[{uint}]]

I_u8_low: np.ndarray[Any, np.dtype[np.uint64]] = np.array([0], dtype=np.uint64)
I_u8_low_like: list[int] = [0]
I_u8_high_open: np.ndarray[Any, np.dtype[np.uint64]] = np.array([18446744073709551615], dtype=np.uint64)
I_u8_high_closed: np.ndarray[Any, np.dtype[np.uint64]] = np.array([18446744073709551615], dtype=np.uint64)

reveal_type(def_gen.integers(18446744073709551616, dtype="u8"))  # E: int
reveal_type(def_gen.integers(0, 18446744073709551616, dtype="u8"))  # E: int
reveal_type(def_gen.integers(18446744073709551615, dtype="u8", endpoint=True))  # E: int
reveal_type(def_gen.integers(0, 18446744073709551615, dtype="u8", endpoint=True))  # E: int
reveal_type(def_gen.integers(I_u8_low_like, 18446744073709551615, dtype="u8", endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(I_u8_high_open, dtype="u8"))  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(I_u8_low, I_u8_high_open, dtype="u8"))  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(0, I_u8_high_open, dtype="u8"))  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(I_u8_high_closed, dtype="u8", endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(I_u8_low, I_u8_high_closed, dtype="u8", endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(0, I_u8_high_closed, dtype="u8", endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]

reveal_type(def_gen.integers(18446744073709551616, dtype="uint64"))  # E: int
reveal_type(def_gen.integers(0, 18446744073709551616, dtype="uint64"))  # E: int
reveal_type(def_gen.integers(18446744073709551615, dtype="uint64", endpoint=True))  # E: int
reveal_type(def_gen.integers(0, 18446744073709551615, dtype="uint64", endpoint=True))  # E: int
reveal_type(def_gen.integers(I_u8_low_like, 18446744073709551615, dtype="uint64", endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(I_u8_high_open, dtype="uint64"))  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(I_u8_low, I_u8_high_open, dtype="uint64"))  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(0, I_u8_high_open, dtype="uint64"))  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(I_u8_high_closed, dtype="uint64", endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(I_u8_low, I_u8_high_closed, dtype="uint64", endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(0, I_u8_high_closed, dtype="uint64", endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]

reveal_type(def_gen.integers(18446744073709551616, dtype=np.uint64))  # E: int
reveal_type(def_gen.integers(0, 18446744073709551616, dtype=np.uint64))  # E: int
reveal_type(def_gen.integers(18446744073709551615, dtype=np.uint64, endpoint=True))  # E: int
reveal_type(def_gen.integers(0, 18446744073709551615, dtype=np.uint64, endpoint=True))  # E: int
reveal_type(def_gen.integers(I_u8_low_like, 18446744073709551615, dtype=np.uint64, endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(I_u8_high_open, dtype=np.uint64))  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(I_u8_low, I_u8_high_open, dtype=np.uint64))  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(0, I_u8_high_open, dtype=np.uint64))  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(I_u8_high_closed, dtype=np.uint64, endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(I_u8_low, I_u8_high_closed, dtype=np.uint64, endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(0, I_u8_high_closed, dtype=np.uint64, endpoint=True))  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]

I_i1_low: np.ndarray[Any, np.dtype[np.int8]] = np.array([-128], dtype=np.int8)
I_i1_low_like: list[int] = [-128]
I_i1_high_open: np.ndarray[Any, np.dtype[np.int8]] = np.array([127], dtype=np.int8)
I_i1_high_closed: np.ndarray[Any, np.dtype[np.int8]] = np.array([127], dtype=np.int8)

reveal_type(def_gen.integers(128, dtype="i1"))  # E: int
reveal_type(def_gen.integers(-128, 128, dtype="i1"))  # E: int
reveal_type(def_gen.integers(127, dtype="i1", endpoint=True))  # E: int
reveal_type(def_gen.integers(-128, 127, dtype="i1", endpoint=True))  # E: int
reveal_type(def_gen.integers(I_i1_low_like, 127, dtype="i1", endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(I_i1_high_open, dtype="i1"))  # E: ndarray[Any, dtype[signedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(I_i1_low, I_i1_high_open, dtype="i1"))  # E: ndarray[Any, dtype[signedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(-128, I_i1_high_open, dtype="i1"))  # E: ndarray[Any, dtype[signedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(I_i1_high_closed, dtype="i1", endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(I_i1_low, I_i1_high_closed, dtype="i1", endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(-128, I_i1_high_closed, dtype="i1", endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._8Bit]]]

reveal_type(def_gen.integers(128, dtype="int8"))  # E: int
reveal_type(def_gen.integers(-128, 128, dtype="int8"))  # E: int
reveal_type(def_gen.integers(127, dtype="int8", endpoint=True))  # E: int
reveal_type(def_gen.integers(-128, 127, dtype="int8", endpoint=True))  # E: int
reveal_type(def_gen.integers(I_i1_low_like, 127, dtype="int8", endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(I_i1_high_open, dtype="int8"))  # E: ndarray[Any, dtype[signedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(I_i1_low, I_i1_high_open, dtype="int8"))  # E: ndarray[Any, dtype[signedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(-128, I_i1_high_open, dtype="int8"))  # E: ndarray[Any, dtype[signedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(I_i1_high_closed, dtype="int8", endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(I_i1_low, I_i1_high_closed, dtype="int8", endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(-128, I_i1_high_closed, dtype="int8", endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._8Bit]]]

reveal_type(def_gen.integers(128, dtype=np.int8))  # E: int
reveal_type(def_gen.integers(-128, 128, dtype=np.int8))  # E: int
reveal_type(def_gen.integers(127, dtype=np.int8, endpoint=True))  # E: int
reveal_type(def_gen.integers(-128, 127, dtype=np.int8, endpoint=True))  # E: int
reveal_type(def_gen.integers(I_i1_low_like, 127, dtype=np.int8, endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(I_i1_high_open, dtype=np.int8))  # E: ndarray[Any, dtype[signedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(I_i1_low, I_i1_high_open, dtype=np.int8))  # E: ndarray[Any, dtype[signedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(-128, I_i1_high_open, dtype=np.int8))  # E: ndarray[Any, dtype[signedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(I_i1_high_closed, dtype=np.int8, endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(I_i1_low, I_i1_high_closed, dtype=np.int8, endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._8Bit]]]
reveal_type(def_gen.integers(-128, I_i1_high_closed, dtype=np.int8, endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._8Bit]]]

I_i2_low: np.ndarray[Any, np.dtype[np.int16]] = np.array([-32768], dtype=np.int16)
I_i2_low_like: list[int] = [-32768]
I_i2_high_open: np.ndarray[Any, np.dtype[np.int16]] = np.array([32767], dtype=np.int16)
I_i2_high_closed: np.ndarray[Any, np.dtype[np.int16]] = np.array([32767], dtype=np.int16)

reveal_type(def_gen.integers(32768, dtype="i2"))  # E: int
reveal_type(def_gen.integers(-32768, 32768, dtype="i2"))  # E: int
reveal_type(def_gen.integers(32767, dtype="i2", endpoint=True))  # E: int
reveal_type(def_gen.integers(-32768, 32767, dtype="i2", endpoint=True))  # E: int
reveal_type(def_gen.integers(I_i2_low_like, 32767, dtype="i2", endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(I_i2_high_open, dtype="i2"))  # E: ndarray[Any, dtype[signedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(I_i2_low, I_i2_high_open, dtype="i2"))  # E: ndarray[Any, dtype[signedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(-32768, I_i2_high_open, dtype="i2"))  # E: ndarray[Any, dtype[signedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(I_i2_high_closed, dtype="i2", endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(I_i2_low, I_i2_high_closed, dtype="i2", endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(-32768, I_i2_high_closed, dtype="i2", endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._16Bit]]]

reveal_type(def_gen.integers(32768, dtype="int16"))  # E: int
reveal_type(def_gen.integers(-32768, 32768, dtype="int16"))  # E: int
reveal_type(def_gen.integers(32767, dtype="int16", endpoint=True))  # E: int
reveal_type(def_gen.integers(-32768, 32767, dtype="int16", endpoint=True))  # E: int
reveal_type(def_gen.integers(I_i2_low_like, 32767, dtype="int16", endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(I_i2_high_open, dtype="int16"))  # E: ndarray[Any, dtype[signedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(I_i2_low, I_i2_high_open, dtype="int16"))  # E: ndarray[Any, dtype[signedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(-32768, I_i2_high_open, dtype="int16"))  # E: ndarray[Any, dtype[signedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(I_i2_high_closed, dtype="int16", endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(I_i2_low, I_i2_high_closed, dtype="int16", endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(-32768, I_i2_high_closed, dtype="int16", endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._16Bit]]]

reveal_type(def_gen.integers(32768, dtype=np.int16))  # E: int
reveal_type(def_gen.integers(-32768, 32768, dtype=np.int16))  # E: int
reveal_type(def_gen.integers(32767, dtype=np.int16, endpoint=True))  # E: int
reveal_type(def_gen.integers(-32768, 32767, dtype=np.int16, endpoint=True))  # E: int
reveal_type(def_gen.integers(I_i2_low_like, 32767, dtype=np.int16, endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(I_i2_high_open, dtype=np.int16))  # E: ndarray[Any, dtype[signedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(I_i2_low, I_i2_high_open, dtype=np.int16))  # E: ndarray[Any, dtype[signedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(-32768, I_i2_high_open, dtype=np.int16))  # E: ndarray[Any, dtype[signedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(I_i2_high_closed, dtype=np.int16, endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(I_i2_low, I_i2_high_closed, dtype=np.int16, endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._16Bit]]]
reveal_type(def_gen.integers(-32768, I_i2_high_closed, dtype=np.int16, endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._16Bit]]]

I_i4_low: np.ndarray[Any, np.dtype[np.int32]] = np.array([-2147483648], dtype=np.int32)
I_i4_low_like: list[int] = [-2147483648]
I_i4_high_open: np.ndarray[Any, np.dtype[np.int32]] = np.array([2147483647], dtype=np.int32)
I_i4_high_closed: np.ndarray[Any, np.dtype[np.int32]] = np.array([2147483647], dtype=np.int32)

reveal_type(def_gen.integers(2147483648, dtype="i4"))  # E: int
reveal_type(def_gen.integers(-2147483648, 2147483648, dtype="i4"))  # E: int
reveal_type(def_gen.integers(2147483647, dtype="i4", endpoint=True))  # E: int
reveal_type(def_gen.integers(-2147483648, 2147483647, dtype="i4", endpoint=True))  # E: int
reveal_type(def_gen.integers(I_i4_low_like, 2147483647, dtype="i4", endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(I_i4_high_open, dtype="i4"))  # E: ndarray[Any, dtype[signedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(I_i4_low, I_i4_high_open, dtype="i4"))  # E: ndarray[Any, dtype[signedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(-2147483648, I_i4_high_open, dtype="i4"))  # E: ndarray[Any, dtype[signedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(I_i4_high_closed, dtype="i4", endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(I_i4_low, I_i4_high_closed, dtype="i4", endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(-2147483648, I_i4_high_closed, dtype="i4", endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._32Bit]]]

reveal_type(def_gen.integers(2147483648, dtype="int32"))  # E: int
reveal_type(def_gen.integers(-2147483648, 2147483648, dtype="int32"))  # E: int
reveal_type(def_gen.integers(2147483647, dtype="int32", endpoint=True))  # E: int
reveal_type(def_gen.integers(-2147483648, 2147483647, dtype="int32", endpoint=True))  # E: int
reveal_type(def_gen.integers(I_i4_low_like, 2147483647, dtype="int32", endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(I_i4_high_open, dtype="int32"))  # E: ndarray[Any, dtype[signedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(I_i4_low, I_i4_high_open, dtype="int32"))  # E: ndarray[Any, dtype[signedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(-2147483648, I_i4_high_open, dtype="int32"))  # E: ndarray[Any, dtype[signedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(I_i4_high_closed, dtype="int32", endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(I_i4_low, I_i4_high_closed, dtype="int32", endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(-2147483648, I_i4_high_closed, dtype="int32", endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._32Bit]]]

reveal_type(def_gen.integers(2147483648, dtype=np.int32))  # E: int
reveal_type(def_gen.integers(-2147483648, 2147483648, dtype=np.int32))  # E: int
reveal_type(def_gen.integers(2147483647, dtype=np.int32, endpoint=True))  # E: int
reveal_type(def_gen.integers(-2147483648, 2147483647, dtype=np.int32, endpoint=True))  # E: int
reveal_type(def_gen.integers(I_i4_low_like, 2147483647, dtype=np.int32, endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(I_i4_high_open, dtype=np.int32))  # E: ndarray[Any, dtype[signedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(I_i4_low, I_i4_high_open, dtype=np.int32))  # E: ndarray[Any, dtype[signedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(-2147483648, I_i4_high_open, dtype=np.int32))  # E: ndarray[Any, dtype[signedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(I_i4_high_closed, dtype=np.int32, endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(I_i4_low, I_i4_high_closed, dtype=np.int32, endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._32Bit]]]
reveal_type(def_gen.integers(-2147483648, I_i4_high_closed, dtype=np.int32, endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._32Bit]]]

I_i8_low: np.ndarray[Any, np.dtype[np.int64]] = np.array([-9223372036854775808], dtype=np.int64)
I_i8_low_like: list[int] = [-9223372036854775808]
I_i8_high_open: np.ndarray[Any, np.dtype[np.int64]] = np.array([9223372036854775807], dtype=np.int64)
I_i8_high_closed: np.ndarray[Any, np.dtype[np.int64]] = np.array([9223372036854775807], dtype=np.int64)

reveal_type(def_gen.integers(9223372036854775808, dtype="i8"))  # E: int
reveal_type(def_gen.integers(-9223372036854775808, 9223372036854775808, dtype="i8"))  # E: int
reveal_type(def_gen.integers(9223372036854775807, dtype="i8", endpoint=True))  # E: int
reveal_type(def_gen.integers(-9223372036854775808, 9223372036854775807, dtype="i8", endpoint=True))  # E: int
reveal_type(def_gen.integers(I_i8_low_like, 9223372036854775807, dtype="i8", endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(I_i8_high_open, dtype="i8"))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(I_i8_low, I_i8_high_open, dtype="i8"))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(-9223372036854775808, I_i8_high_open, dtype="i8"))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(I_i8_high_closed, dtype="i8", endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(I_i8_low, I_i8_high_closed, dtype="i8", endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(-9223372036854775808, I_i8_high_closed, dtype="i8", endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]

reveal_type(def_gen.integers(9223372036854775808, dtype="int64"))  # E: int
reveal_type(def_gen.integers(-9223372036854775808, 9223372036854775808, dtype="int64"))  # E: int
reveal_type(def_gen.integers(9223372036854775807, dtype="int64", endpoint=True))  # E: int
reveal_type(def_gen.integers(-9223372036854775808, 9223372036854775807, dtype="int64", endpoint=True))  # E: int
reveal_type(def_gen.integers(I_i8_low_like, 9223372036854775807, dtype="int64", endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(I_i8_high_open, dtype="int64"))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(I_i8_low, I_i8_high_open, dtype="int64"))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(-9223372036854775808, I_i8_high_open, dtype="int64"))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(I_i8_high_closed, dtype="int64", endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(I_i8_low, I_i8_high_closed, dtype="int64", endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(-9223372036854775808, I_i8_high_closed, dtype="int64", endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]

reveal_type(def_gen.integers(9223372036854775808, dtype=np.int64))  # E: int
reveal_type(def_gen.integers(-9223372036854775808, 9223372036854775808, dtype=np.int64))  # E: int
reveal_type(def_gen.integers(9223372036854775807, dtype=np.int64, endpoint=True))  # E: int
reveal_type(def_gen.integers(-9223372036854775808, 9223372036854775807, dtype=np.int64, endpoint=True))  # E: int
reveal_type(def_gen.integers(I_i8_low_like, 9223372036854775807, dtype=np.int64, endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(I_i8_high_open, dtype=np.int64))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(I_i8_low, I_i8_high_open, dtype=np.int64))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(-9223372036854775808, I_i8_high_open, dtype=np.int64))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(I_i8_high_closed, dtype=np.int64, endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(I_i8_low, I_i8_high_closed, dtype=np.int64, endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.integers(-9223372036854775808, I_i8_high_closed, dtype=np.int64, endpoint=True))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]


reveal_type(def_gen.bit_generator)  # E: BitGenerator

reveal_type(def_gen.bytes(2))  # E: bytes

reveal_type(def_gen.choice(5))  # E: int
reveal_type(def_gen.choice(5, 3))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.choice(5, 3, replace=True))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.choice(5, 3, p=[1 / 5] * 5))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.choice(5, 3, p=[1 / 5] * 5, replace=False))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]

reveal_type(def_gen.choice(["pooh", "rabbit", "piglet", "Christopher"]))  # E: Any
reveal_type(def_gen.choice(["pooh", "rabbit", "piglet", "Christopher"], 3))  # E: ndarray[Any, Any]
reveal_type(def_gen.choice(["pooh", "rabbit", "piglet", "Christopher"], 3, p=[1 / 4] * 4))  # E: ndarray[Any, Any]
reveal_type(def_gen.choice(["pooh", "rabbit", "piglet", "Christopher"], 3, replace=True))  # E: ndarray[Any, Any]
reveal_type(def_gen.choice(["pooh", "rabbit", "piglet", "Christopher"], 3, replace=False, p=np.array([1 / 8, 1 / 8, 1 / 2, 1 / 4])))  # E: ndarray[Any, Any]

reveal_type(def_gen.dirichlet([0.5, 0.5]))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.dirichlet(np.array([0.5, 0.5])))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.dirichlet(np.array([0.5, 0.5]), size=3))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(def_gen.multinomial(20, [1 / 6.0] * 6))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.multinomial(20, np.array([0.5, 0.5])))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.multinomial(20, [1 / 6.0] * 6, size=2))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.multinomial([[10], [20]], [1 / 6.0] * 6, size=(2, 2)))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.multinomial(np.array([[10], [20]]), np.array([0.5, 0.5]), size=(2, 2)))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]

reveal_type(def_gen.multivariate_hypergeometric([3, 5, 7], 2))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.multivariate_hypergeometric(np.array([3, 5, 7]), 2))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.multivariate_hypergeometric(np.array([3, 5, 7]), 2, size=4))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.multivariate_hypergeometric(np.array([3, 5, 7]), 2, size=(4, 7)))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.multivariate_hypergeometric([3, 5, 7], 2, method="count"))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.multivariate_hypergeometric(np.array([3, 5, 7]), 2, method="marginals"))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]

reveal_type(def_gen.multivariate_normal([0.0], [[1.0]]))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.multivariate_normal([0.0], np.array([[1.0]])))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.multivariate_normal(np.array([0.0]), [[1.0]]))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(def_gen.multivariate_normal([0.0], np.array([[1.0]])))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(def_gen.permutation(10))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(def_gen.permutation([1, 2, 3, 4]))  # E: ndarray[Any, Any]
reveal_type(def_gen.permutation(np.array([1, 2, 3, 4])))  # E: ndarray[Any, Any]
reveal_type(def_gen.permutation(D_2D, axis=1))  # E: ndarray[Any, Any]
reveal_type(def_gen.permuted(D_2D))  # E: ndarray[Any, Any]
reveal_type(def_gen.permuted(D_2D_like))  # E: ndarray[Any, Any]
reveal_type(def_gen.permuted(D_2D, axis=1))  # E: ndarray[Any, Any]
reveal_type(def_gen.permuted(D_2D, out=D_2D))  # E: ndarray[Any, Any]
reveal_type(def_gen.permuted(D_2D_like, out=D_2D))  # E: ndarray[Any, Any]
reveal_type(def_gen.permuted(D_2D_like, out=D_2D))  # E: ndarray[Any, Any]
reveal_type(def_gen.permuted(D_2D, axis=1, out=D_2D))  # E: ndarray[Any, Any]

reveal_type(def_gen.shuffle(np.arange(10)))  # E: None
reveal_type(def_gen.shuffle([1, 2, 3, 4, 5]))  # E: None
reveal_type(def_gen.shuffle(D_2D, axis=1))  # E: None

reveal_type(np.random.Generator(pcg64))  # E: Generator
reveal_type(def_gen.__str__())  # E: str
reveal_type(def_gen.__repr__())  # E: str
def_gen_state = def_gen.__getstate__()
reveal_type(def_gen_state)  # E: builtins.dict[builtins.str, Any]
reveal_type(def_gen.__setstate__(def_gen_state))  # E: None

# RandomState
random_st: np.random.RandomState = np.random.RandomState()

reveal_type(random_st.standard_normal())  # E: float
reveal_type(random_st.standard_normal(size=None))  # E: float
reveal_type(random_st.standard_normal(size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]

reveal_type(random_st.random())  # E: float
reveal_type(random_st.random(size=None))  # E: float
reveal_type(random_st.random(size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]

reveal_type(random_st.standard_cauchy())  # E: float
reveal_type(random_st.standard_cauchy(size=None))  # E: float
reveal_type(random_st.standard_cauchy(size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(random_st.standard_exponential())  # E: float
reveal_type(random_st.standard_exponential(size=None))  # E: float
reveal_type(random_st.standard_exponential(size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]

reveal_type(random_st.zipf(1.5))  # E: int
reveal_type(random_st.zipf(1.5, size=None))  # E: int
reveal_type(random_st.zipf(1.5, size=1))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.zipf(D_arr_1p5))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.zipf(D_arr_1p5, size=1))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.zipf(D_arr_like_1p5))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.zipf(D_arr_like_1p5, size=1))  # E: ndarray[Any, dtype[{int_}]]

reveal_type(random_st.weibull(0.5))  # E: float
reveal_type(random_st.weibull(0.5, size=None))  # E: float
reveal_type(random_st.weibull(0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.weibull(D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.weibull(D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.weibull(D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.weibull(D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(random_st.standard_t(0.5))  # E: float
reveal_type(random_st.standard_t(0.5, size=None))  # E: float
reveal_type(random_st.standard_t(0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.standard_t(D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.standard_t(D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.standard_t(D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.standard_t(D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(random_st.poisson(0.5))  # E: int
reveal_type(random_st.poisson(0.5, size=None))  # E: int
reveal_type(random_st.poisson(0.5, size=1))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.poisson(D_arr_0p5))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.poisson(D_arr_0p5, size=1))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.poisson(D_arr_like_0p5))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.poisson(D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[{int_}]]

reveal_type(random_st.power(0.5))  # E: float
reveal_type(random_st.power(0.5, size=None))  # E: float
reveal_type(random_st.power(0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.power(D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.power(D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.power(D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.power(D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(random_st.pareto(0.5))  # E: float
reveal_type(random_st.pareto(0.5, size=None))  # E: float
reveal_type(random_st.pareto(0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.pareto(D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.pareto(D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.pareto(D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.pareto(D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(random_st.chisquare(0.5))  # E: float
reveal_type(random_st.chisquare(0.5, size=None))  # E: float
reveal_type(random_st.chisquare(0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.chisquare(D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.chisquare(D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.chisquare(D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.chisquare(D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(random_st.exponential(0.5))  # E: float
reveal_type(random_st.exponential(0.5, size=None))  # E: float
reveal_type(random_st.exponential(0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.exponential(D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.exponential(D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.exponential(D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.exponential(D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(random_st.geometric(0.5))  # E: int
reveal_type(random_st.geometric(0.5, size=None))  # E: int
reveal_type(random_st.geometric(0.5, size=1))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.geometric(D_arr_0p5))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.geometric(D_arr_0p5, size=1))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.geometric(D_arr_like_0p5))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.geometric(D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[{int_}]]

reveal_type(random_st.logseries(0.5))  # E: int
reveal_type(random_st.logseries(0.5, size=None))  # E: int
reveal_type(random_st.logseries(0.5, size=1))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.logseries(D_arr_0p5))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.logseries(D_arr_0p5, size=1))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.logseries(D_arr_like_0p5))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.logseries(D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[{int_}]]

reveal_type(random_st.rayleigh(0.5))  # E: float
reveal_type(random_st.rayleigh(0.5, size=None))  # E: float
reveal_type(random_st.rayleigh(0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.rayleigh(D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.rayleigh(D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.rayleigh(D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.rayleigh(D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(random_st.standard_gamma(0.5))  # E: float
reveal_type(random_st.standard_gamma(0.5, size=None))  # E: float
reveal_type(random_st.standard_gamma(0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]
reveal_type(random_st.standard_gamma(D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]
reveal_type(random_st.standard_gamma(D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]
reveal_type(random_st.standard_gamma(D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]
reveal_type(random_st.standard_gamma(D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]
reveal_type(random_st.standard_gamma(D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]]

reveal_type(random_st.vonmises(0.5, 0.5))  # E: float
reveal_type(random_st.vonmises(0.5, 0.5, size=None))  # E: float
reveal_type(random_st.vonmises(0.5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.vonmises(D_arr_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.vonmises(0.5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.vonmises(D_arr_0p5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.vonmises(0.5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.vonmises(D_arr_like_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.vonmises(0.5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.vonmises(D_arr_0p5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.vonmises(D_arr_like_0p5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.vonmises(D_arr_0p5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.vonmises(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(random_st.wald(0.5, 0.5))  # E: float
reveal_type(random_st.wald(0.5, 0.5, size=None))  # E: float
reveal_type(random_st.wald(0.5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.wald(D_arr_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.wald(0.5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.wald(D_arr_0p5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.wald(0.5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.wald(D_arr_like_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.wald(0.5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.wald(D_arr_0p5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.wald(D_arr_like_0p5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.wald(D_arr_0p5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.wald(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(random_st.uniform(0.5, 0.5))  # E: float
reveal_type(random_st.uniform(0.5, 0.5, size=None))  # E: float
reveal_type(random_st.uniform(0.5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.uniform(D_arr_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.uniform(0.5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.uniform(D_arr_0p5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.uniform(0.5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.uniform(D_arr_like_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.uniform(0.5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.uniform(D_arr_0p5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.uniform(D_arr_like_0p5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.uniform(D_arr_0p5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.uniform(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(random_st.beta(0.5, 0.5))  # E: float
reveal_type(random_st.beta(0.5, 0.5, size=None))  # E: float
reveal_type(random_st.beta(0.5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.beta(D_arr_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.beta(0.5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.beta(D_arr_0p5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.beta(0.5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.beta(D_arr_like_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.beta(0.5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.beta(D_arr_0p5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.beta(D_arr_like_0p5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.beta(D_arr_0p5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.beta(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(random_st.f(0.5, 0.5))  # E: float
reveal_type(random_st.f(0.5, 0.5, size=None))  # E: float
reveal_type(random_st.f(0.5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.f(D_arr_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.f(0.5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.f(D_arr_0p5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.f(0.5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.f(D_arr_like_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.f(0.5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.f(D_arr_0p5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.f(D_arr_like_0p5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.f(D_arr_0p5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.f(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(random_st.gamma(0.5, 0.5))  # E: float
reveal_type(random_st.gamma(0.5, 0.5, size=None))  # E: float
reveal_type(random_st.gamma(0.5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.gamma(D_arr_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.gamma(0.5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.gamma(D_arr_0p5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.gamma(0.5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.gamma(D_arr_like_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.gamma(0.5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.gamma(D_arr_0p5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.gamma(D_arr_like_0p5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.gamma(D_arr_0p5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.gamma(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(random_st.gumbel(0.5, 0.5))  # E: float
reveal_type(random_st.gumbel(0.5, 0.5, size=None))  # E: float
reveal_type(random_st.gumbel(0.5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.gumbel(D_arr_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.gumbel(0.5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.gumbel(D_arr_0p5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.gumbel(0.5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.gumbel(D_arr_like_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.gumbel(0.5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.gumbel(D_arr_0p5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.gumbel(D_arr_like_0p5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.gumbel(D_arr_0p5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.gumbel(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(random_st.laplace(0.5, 0.5))  # E: float
reveal_type(random_st.laplace(0.5, 0.5, size=None))  # E: float
reveal_type(random_st.laplace(0.5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.laplace(D_arr_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.laplace(0.5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.laplace(D_arr_0p5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.laplace(0.5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.laplace(D_arr_like_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.laplace(0.5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.laplace(D_arr_0p5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.laplace(D_arr_like_0p5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.laplace(D_arr_0p5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.laplace(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(random_st.logistic(0.5, 0.5))  # E: float
reveal_type(random_st.logistic(0.5, 0.5, size=None))  # E: float
reveal_type(random_st.logistic(0.5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.logistic(D_arr_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.logistic(0.5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.logistic(D_arr_0p5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.logistic(0.5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.logistic(D_arr_like_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.logistic(0.5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.logistic(D_arr_0p5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.logistic(D_arr_like_0p5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.logistic(D_arr_0p5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.logistic(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(random_st.lognormal(0.5, 0.5))  # E: float
reveal_type(random_st.lognormal(0.5, 0.5, size=None))  # E: float
reveal_type(random_st.lognormal(0.5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.lognormal(D_arr_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.lognormal(0.5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.lognormal(D_arr_0p5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.lognormal(0.5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.lognormal(D_arr_like_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.lognormal(0.5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.lognormal(D_arr_0p5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.lognormal(D_arr_like_0p5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.lognormal(D_arr_0p5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.lognormal(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(random_st.noncentral_chisquare(0.5, 0.5))  # E: float
reveal_type(random_st.noncentral_chisquare(0.5, 0.5, size=None))  # E: float
reveal_type(random_st.noncentral_chisquare(0.5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.noncentral_chisquare(D_arr_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.noncentral_chisquare(0.5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.noncentral_chisquare(D_arr_0p5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.noncentral_chisquare(0.5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.noncentral_chisquare(D_arr_like_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.noncentral_chisquare(0.5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.noncentral_chisquare(D_arr_0p5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.noncentral_chisquare(D_arr_like_0p5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.noncentral_chisquare(D_arr_0p5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.noncentral_chisquare(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(random_st.normal(0.5, 0.5))  # E: float
reveal_type(random_st.normal(0.5, 0.5, size=None))  # E: float
reveal_type(random_st.normal(0.5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.normal(D_arr_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.normal(0.5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.normal(D_arr_0p5, 0.5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.normal(0.5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.normal(D_arr_like_0p5, 0.5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.normal(0.5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.normal(D_arr_0p5, D_arr_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.normal(D_arr_like_0p5, D_arr_like_0p5))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.normal(D_arr_0p5, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.normal(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(random_st.triangular(0.1, 0.5, 0.9))  # E: float
reveal_type(random_st.triangular(0.1, 0.5, 0.9, size=None))  # E: float
reveal_type(random_st.triangular(0.1, 0.5, 0.9, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.triangular(D_arr_0p1, 0.5, 0.9))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.triangular(0.1, D_arr_0p5, 0.9))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.triangular(D_arr_0p1, 0.5, D_arr_like_0p9, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.triangular(0.1, D_arr_0p5, 0.9, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.triangular(D_arr_like_0p1, 0.5, D_arr_0p9))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.triangular(0.5, D_arr_like_0p5, 0.9))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.triangular(D_arr_0p1, D_arr_0p5, 0.9))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.triangular(D_arr_like_0p1, D_arr_like_0p5, 0.9))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.triangular(D_arr_0p1, D_arr_0p5, D_arr_0p9, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.triangular(D_arr_like_0p1, D_arr_like_0p5, D_arr_like_0p9, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(random_st.noncentral_f(0.1, 0.5, 0.9))  # E: float
reveal_type(random_st.noncentral_f(0.1, 0.5, 0.9, size=None))  # E: float
reveal_type(random_st.noncentral_f(0.1, 0.5, 0.9, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.noncentral_f(D_arr_0p1, 0.5, 0.9))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.noncentral_f(0.1, D_arr_0p5, 0.9))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.noncentral_f(D_arr_0p1, 0.5, D_arr_like_0p9, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.noncentral_f(0.1, D_arr_0p5, 0.9, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.noncentral_f(D_arr_like_0p1, 0.5, D_arr_0p9))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.noncentral_f(0.5, D_arr_like_0p5, 0.9))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.noncentral_f(D_arr_0p1, D_arr_0p5, 0.9))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.noncentral_f(D_arr_like_0p1, D_arr_like_0p5, 0.9))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.noncentral_f(D_arr_0p1, D_arr_0p5, D_arr_0p9, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.noncentral_f(D_arr_like_0p1, D_arr_like_0p5, D_arr_like_0p9, size=1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(random_st.binomial(10, 0.5))  # E: int
reveal_type(random_st.binomial(10, 0.5, size=None))  # E: int
reveal_type(random_st.binomial(10, 0.5, size=1))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.binomial(I_arr_10, 0.5))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.binomial(10, D_arr_0p5))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.binomial(I_arr_10, 0.5, size=1))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.binomial(10, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.binomial(I_arr_like_10, 0.5))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.binomial(10, D_arr_like_0p5))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.binomial(I_arr_10, D_arr_0p5))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.binomial(I_arr_like_10, D_arr_like_0p5))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.binomial(I_arr_10, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.binomial(I_arr_like_10, D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[{int_}]]

reveal_type(random_st.negative_binomial(10, 0.5))  # E: int
reveal_type(random_st.negative_binomial(10, 0.5, size=None))  # E: int
reveal_type(random_st.negative_binomial(10, 0.5, size=1))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.negative_binomial(I_arr_10, 0.5))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.negative_binomial(10, D_arr_0p5))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.negative_binomial(I_arr_10, 0.5, size=1))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.negative_binomial(10, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.negative_binomial(I_arr_like_10, 0.5))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.negative_binomial(10, D_arr_like_0p5))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.negative_binomial(I_arr_10, D_arr_0p5))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.negative_binomial(I_arr_like_10, D_arr_like_0p5))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.negative_binomial(I_arr_10, D_arr_0p5, size=1))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.negative_binomial(I_arr_like_10, D_arr_like_0p5, size=1))  # E: ndarray[Any, dtype[{int_}]]

reveal_type(random_st.hypergeometric(20, 20, 10))  # E: int
reveal_type(random_st.hypergeometric(20, 20, 10, size=None))  # E: int
reveal_type(random_st.hypergeometric(20, 20, 10, size=1))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.hypergeometric(I_arr_20, 20, 10))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.hypergeometric(20, I_arr_20, 10))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.hypergeometric(I_arr_20, 20, I_arr_like_10, size=1))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.hypergeometric(20, I_arr_20, 10, size=1))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.hypergeometric(I_arr_like_20, 20, I_arr_10))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.hypergeometric(20, I_arr_like_20, 10))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.hypergeometric(I_arr_20, I_arr_20, 10))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.hypergeometric(I_arr_like_20, I_arr_like_20, 10))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.hypergeometric(I_arr_20, I_arr_20, I_arr_10, size=1))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.hypergeometric(I_arr_like_20, I_arr_like_20, I_arr_like_10, size=1))  # E: ndarray[Any, dtype[{int_}]]

reveal_type(random_st.randint(0, 100))  # E: int
reveal_type(random_st.randint(100))  # E: int
reveal_type(random_st.randint([100]))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.randint(0, [100]))  # E: ndarray[Any, dtype[{int_}]]

reveal_type(random_st.randint(2, dtype=bool))  # E: builtins.bool
reveal_type(random_st.randint(0, 2, dtype=bool))  # E: builtins.bool
reveal_type(random_st.randint(I_bool_high_open, dtype=bool))  # E: ndarray[Any, dtype[bool_]
reveal_type(random_st.randint(I_bool_low, I_bool_high_open, dtype=bool))  # E: ndarray[Any, dtype[bool_]
reveal_type(random_st.randint(0, I_bool_high_open, dtype=bool))  # E: ndarray[Any, dtype[bool_]

reveal_type(random_st.randint(2, dtype=np.bool_))  # E: builtins.bool
reveal_type(random_st.randint(0, 2, dtype=np.bool_))  # E: builtins.bool
reveal_type(random_st.randint(I_bool_high_open, dtype=np.bool_))  # E: ndarray[Any, dtype[bool_]
reveal_type(random_st.randint(I_bool_low, I_bool_high_open, dtype=np.bool_))  # E: ndarray[Any, dtype[bool_]
reveal_type(random_st.randint(0, I_bool_high_open, dtype=np.bool_))  # E: ndarray[Any, dtype[bool_]

reveal_type(random_st.randint(256, dtype="u1"))  # E: int
reveal_type(random_st.randint(0, 256, dtype="u1"))  # E: int
reveal_type(random_st.randint(I_u1_high_open, dtype="u1"))  # E: ndarray[Any, dtype[unsignedinteger[typing._8Bit]]]
reveal_type(random_st.randint(I_u1_low, I_u1_high_open, dtype="u1"))  # E: ndarray[Any, dtype[unsignedinteger[typing._8Bit]]]
reveal_type(random_st.randint(0, I_u1_high_open, dtype="u1"))  # E: ndarray[Any, dtype[unsignedinteger[typing._8Bit]]]

reveal_type(random_st.randint(256, dtype="uint8"))  # E: int
reveal_type(random_st.randint(0, 256, dtype="uint8"))  # E: int
reveal_type(random_st.randint(I_u1_high_open, dtype="uint8"))  # E: ndarray[Any, dtype[unsignedinteger[typing._8Bit]]]
reveal_type(random_st.randint(I_u1_low, I_u1_high_open, dtype="uint8"))  # E: ndarray[Any, dtype[unsignedinteger[typing._8Bit]]]
reveal_type(random_st.randint(0, I_u1_high_open, dtype="uint8"))  # E: ndarray[Any, dtype[unsignedinteger[typing._8Bit]]]

reveal_type(random_st.randint(256, dtype=np.uint8))  # E: int
reveal_type(random_st.randint(0, 256, dtype=np.uint8))  # E: int
reveal_type(random_st.randint(I_u1_high_open, dtype=np.uint8))  # E: ndarray[Any, dtype[unsignedinteger[typing._8Bit]]]
reveal_type(random_st.randint(I_u1_low, I_u1_high_open, dtype=np.uint8))  # E: ndarray[Any, dtype[unsignedinteger[typing._8Bit]]]
reveal_type(random_st.randint(0, I_u1_high_open, dtype=np.uint8))  # E: ndarray[Any, dtype[unsignedinteger[typing._8Bit]]]

reveal_type(random_st.randint(65536, dtype="u2"))  # E: int
reveal_type(random_st.randint(0, 65536, dtype="u2"))  # E: int
reveal_type(random_st.randint(I_u2_high_open, dtype="u2"))  # E: ndarray[Any, dtype[unsignedinteger[typing._16Bit]]]
reveal_type(random_st.randint(I_u2_low, I_u2_high_open, dtype="u2"))  # E: ndarray[Any, dtype[unsignedinteger[typing._16Bit]]]
reveal_type(random_st.randint(0, I_u2_high_open, dtype="u2"))  # E: ndarray[Any, dtype[unsignedinteger[typing._16Bit]]]

reveal_type(random_st.randint(65536, dtype="uint16"))  # E: int
reveal_type(random_st.randint(0, 65536, dtype="uint16"))  # E: int
reveal_type(random_st.randint(I_u2_high_open, dtype="uint16"))  # E: ndarray[Any, dtype[unsignedinteger[typing._16Bit]]]
reveal_type(random_st.randint(I_u2_low, I_u2_high_open, dtype="uint16"))  # E: ndarray[Any, dtype[unsignedinteger[typing._16Bit]]]
reveal_type(random_st.randint(0, I_u2_high_open, dtype="uint16"))  # E: ndarray[Any, dtype[unsignedinteger[typing._16Bit]]]

reveal_type(random_st.randint(65536, dtype=np.uint16))  # E: int
reveal_type(random_st.randint(0, 65536, dtype=np.uint16))  # E: int
reveal_type(random_st.randint(I_u2_high_open, dtype=np.uint16))  # E: ndarray[Any, dtype[unsignedinteger[typing._16Bit]]]
reveal_type(random_st.randint(I_u2_low, I_u2_high_open, dtype=np.uint16))  # E: ndarray[Any, dtype[unsignedinteger[typing._16Bit]]]
reveal_type(random_st.randint(0, I_u2_high_open, dtype=np.uint16))  # E: ndarray[Any, dtype[unsignedinteger[typing._16Bit]]]

reveal_type(random_st.randint(4294967296, dtype="u4"))  # E: int
reveal_type(random_st.randint(0, 4294967296, dtype="u4"))  # E: int
reveal_type(random_st.randint(I_u4_high_open, dtype="u4"))  # E: ndarray[Any, dtype[unsignedinteger[typing._32Bit]]]
reveal_type(random_st.randint(I_u4_low, I_u4_high_open, dtype="u4"))  # E: ndarray[Any, dtype[unsignedinteger[typing._32Bit]]]
reveal_type(random_st.randint(0, I_u4_high_open, dtype="u4"))  # E: ndarray[Any, dtype[unsignedinteger[typing._32Bit]]]

reveal_type(random_st.randint(4294967296, dtype="uint32"))  # E: int
reveal_type(random_st.randint(0, 4294967296, dtype="uint32"))  # E: int
reveal_type(random_st.randint(I_u4_high_open, dtype="uint32"))  # E: ndarray[Any, dtype[unsignedinteger[typing._32Bit]]]
reveal_type(random_st.randint(I_u4_low, I_u4_high_open, dtype="uint32"))  # E: ndarray[Any, dtype[unsignedinteger[typing._32Bit]]]
reveal_type(random_st.randint(0, I_u4_high_open, dtype="uint32"))  # E: ndarray[Any, dtype[unsignedinteger[typing._32Bit]]]

reveal_type(random_st.randint(4294967296, dtype=np.uint32))  # E: int
reveal_type(random_st.randint(0, 4294967296, dtype=np.uint32))  # E: int
reveal_type(random_st.randint(I_u4_high_open, dtype=np.uint32))  # E: ndarray[Any, dtype[unsignedinteger[typing._32Bit]]]
reveal_type(random_st.randint(I_u4_low, I_u4_high_open, dtype=np.uint32))  # E: ndarray[Any, dtype[unsignedinteger[typing._32Bit]]]
reveal_type(random_st.randint(0, I_u4_high_open, dtype=np.uint32))  # E: ndarray[Any, dtype[unsignedinteger[typing._32Bit]]]

reveal_type(random_st.randint(4294967296, dtype=np.uint))  # E: int
reveal_type(random_st.randint(0, 4294967296, dtype=np.uint))  # E: int
reveal_type(random_st.randint(I_u4_high_open, dtype=np.uint))  # E: ndarray[Any, dtype[{uint}]]
reveal_type(random_st.randint(I_u4_low, I_u4_high_open, dtype=np.uint))  # E: ndarray[Any, dtype[{uint}]]
reveal_type(random_st.randint(0, I_u4_high_open, dtype=np.uint))  # E: ndarray[Any, dtype[{uint}]]

reveal_type(random_st.randint(18446744073709551616, dtype="u8"))  # E: int
reveal_type(random_st.randint(0, 18446744073709551616, dtype="u8"))  # E: int
reveal_type(random_st.randint(I_u8_high_open, dtype="u8"))  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]
reveal_type(random_st.randint(I_u8_low, I_u8_high_open, dtype="u8"))  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]
reveal_type(random_st.randint(0, I_u8_high_open, dtype="u8"))  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]

reveal_type(random_st.randint(18446744073709551616, dtype="uint64"))  # E: int
reveal_type(random_st.randint(0, 18446744073709551616, dtype="uint64"))  # E: int
reveal_type(random_st.randint(I_u8_high_open, dtype="uint64"))  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]
reveal_type(random_st.randint(I_u8_low, I_u8_high_open, dtype="uint64"))  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]
reveal_type(random_st.randint(0, I_u8_high_open, dtype="uint64"))  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]

reveal_type(random_st.randint(18446744073709551616, dtype=np.uint64))  # E: int
reveal_type(random_st.randint(0, 18446744073709551616, dtype=np.uint64))  # E: int
reveal_type(random_st.randint(I_u8_high_open, dtype=np.uint64))  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]
reveal_type(random_st.randint(I_u8_low, I_u8_high_open, dtype=np.uint64))  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]
reveal_type(random_st.randint(0, I_u8_high_open, dtype=np.uint64))  # E: ndarray[Any, dtype[unsignedinteger[typing._64Bit]]]

reveal_type(random_st.randint(128, dtype="i1"))  # E: int
reveal_type(random_st.randint(-128, 128, dtype="i1"))  # E: int
reveal_type(random_st.randint(I_i1_high_open, dtype="i1"))  # E: ndarray[Any, dtype[signedinteger[typing._8Bit]]]
reveal_type(random_st.randint(I_i1_low, I_i1_high_open, dtype="i1"))  # E: ndarray[Any, dtype[signedinteger[typing._8Bit]]]
reveal_type(random_st.randint(-128, I_i1_high_open, dtype="i1"))  # E: ndarray[Any, dtype[signedinteger[typing._8Bit]]]

reveal_type(random_st.randint(128, dtype="int8"))  # E: int
reveal_type(random_st.randint(-128, 128, dtype="int8"))  # E: int
reveal_type(random_st.randint(I_i1_high_open, dtype="int8"))  # E: ndarray[Any, dtype[signedinteger[typing._8Bit]]]
reveal_type(random_st.randint(I_i1_low, I_i1_high_open, dtype="int8"))  # E: ndarray[Any, dtype[signedinteger[typing._8Bit]]]
reveal_type(random_st.randint(-128, I_i1_high_open, dtype="int8"))  # E: ndarray[Any, dtype[signedinteger[typing._8Bit]]]

reveal_type(random_st.randint(128, dtype=np.int8))  # E: int
reveal_type(random_st.randint(-128, 128, dtype=np.int8))  # E: int
reveal_type(random_st.randint(I_i1_high_open, dtype=np.int8))  # E: ndarray[Any, dtype[signedinteger[typing._8Bit]]]
reveal_type(random_st.randint(I_i1_low, I_i1_high_open, dtype=np.int8))  # E: ndarray[Any, dtype[signedinteger[typing._8Bit]]]
reveal_type(random_st.randint(-128, I_i1_high_open, dtype=np.int8))  # E: ndarray[Any, dtype[signedinteger[typing._8Bit]]]

reveal_type(random_st.randint(32768, dtype="i2"))  # E: int
reveal_type(random_st.randint(-32768, 32768, dtype="i2"))  # E: int
reveal_type(random_st.randint(I_i2_high_open, dtype="i2"))  # E: ndarray[Any, dtype[signedinteger[typing._16Bit]]]
reveal_type(random_st.randint(I_i2_low, I_i2_high_open, dtype="i2"))  # E: ndarray[Any, dtype[signedinteger[typing._16Bit]]]
reveal_type(random_st.randint(-32768, I_i2_high_open, dtype="i2"))  # E: ndarray[Any, dtype[signedinteger[typing._16Bit]]]
reveal_type(random_st.randint(32768, dtype="int16"))  # E: int
reveal_type(random_st.randint(-32768, 32768, dtype="int16"))  # E: int
reveal_type(random_st.randint(I_i2_high_open, dtype="int16"))  # E: ndarray[Any, dtype[signedinteger[typing._16Bit]]]
reveal_type(random_st.randint(I_i2_low, I_i2_high_open, dtype="int16"))  # E: ndarray[Any, dtype[signedinteger[typing._16Bit]]]
reveal_type(random_st.randint(-32768, I_i2_high_open, dtype="int16"))  # E: ndarray[Any, dtype[signedinteger[typing._16Bit]]]
reveal_type(random_st.randint(32768, dtype=np.int16))  # E: int
reveal_type(random_st.randint(-32768, 32768, dtype=np.int16))  # E: int
reveal_type(random_st.randint(I_i2_high_open, dtype=np.int16))  # E: ndarray[Any, dtype[signedinteger[typing._16Bit]]]
reveal_type(random_st.randint(I_i2_low, I_i2_high_open, dtype=np.int16))  # E: ndarray[Any, dtype[signedinteger[typing._16Bit]]]
reveal_type(random_st.randint(-32768, I_i2_high_open, dtype=np.int16))  # E: ndarray[Any, dtype[signedinteger[typing._16Bit]]]

reveal_type(random_st.randint(2147483648, dtype="i4"))  # E: int
reveal_type(random_st.randint(-2147483648, 2147483648, dtype="i4"))  # E: int
reveal_type(random_st.randint(I_i4_high_open, dtype="i4"))  # E: ndarray[Any, dtype[signedinteger[typing._32Bit]]]
reveal_type(random_st.randint(I_i4_low, I_i4_high_open, dtype="i4"))  # E: ndarray[Any, dtype[signedinteger[typing._32Bit]]]
reveal_type(random_st.randint(-2147483648, I_i4_high_open, dtype="i4"))  # E: ndarray[Any, dtype[signedinteger[typing._32Bit]]]

reveal_type(random_st.randint(2147483648, dtype="int32"))  # E: int
reveal_type(random_st.randint(-2147483648, 2147483648, dtype="int32"))  # E: int
reveal_type(random_st.randint(I_i4_high_open, dtype="int32"))  # E: ndarray[Any, dtype[signedinteger[typing._32Bit]]]
reveal_type(random_st.randint(I_i4_low, I_i4_high_open, dtype="int32"))  # E: ndarray[Any, dtype[signedinteger[typing._32Bit]]]
reveal_type(random_st.randint(-2147483648, I_i4_high_open, dtype="int32"))  # E: ndarray[Any, dtype[signedinteger[typing._32Bit]]]

reveal_type(random_st.randint(2147483648, dtype=np.int32))  # E: int
reveal_type(random_st.randint(-2147483648, 2147483648, dtype=np.int32))  # E: int
reveal_type(random_st.randint(I_i4_high_open, dtype=np.int32))  # E: ndarray[Any, dtype[signedinteger[typing._32Bit]]]
reveal_type(random_st.randint(I_i4_low, I_i4_high_open, dtype=np.int32))  # E: ndarray[Any, dtype[signedinteger[typing._32Bit]]]
reveal_type(random_st.randint(-2147483648, I_i4_high_open, dtype=np.int32))  # E: ndarray[Any, dtype[signedinteger[typing._32Bit]]]

reveal_type(random_st.randint(2147483648, dtype=np.int_))  # E: int
reveal_type(random_st.randint(-2147483648, 2147483648, dtype=np.int_))  # E: int
reveal_type(random_st.randint(I_i4_high_open, dtype=np.int_))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.randint(I_i4_low, I_i4_high_open, dtype=np.int_))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.randint(-2147483648, I_i4_high_open, dtype=np.int_))  # E: ndarray[Any, dtype[{int_}]]

reveal_type(random_st.randint(9223372036854775808, dtype="i8"))  # E: int
reveal_type(random_st.randint(-9223372036854775808, 9223372036854775808, dtype="i8"))  # E: int
reveal_type(random_st.randint(I_i8_high_open, dtype="i8"))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(random_st.randint(I_i8_low, I_i8_high_open, dtype="i8"))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(random_st.randint(-9223372036854775808, I_i8_high_open, dtype="i8"))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]

reveal_type(random_st.randint(9223372036854775808, dtype="int64"))  # E: int
reveal_type(random_st.randint(-9223372036854775808, 9223372036854775808, dtype="int64"))  # E: int
reveal_type(random_st.randint(I_i8_high_open, dtype="int64"))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(random_st.randint(I_i8_low, I_i8_high_open, dtype="int64"))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(random_st.randint(-9223372036854775808, I_i8_high_open, dtype="int64"))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]

reveal_type(random_st.randint(9223372036854775808, dtype=np.int64))  # E: int
reveal_type(random_st.randint(-9223372036854775808, 9223372036854775808, dtype=np.int64))  # E: int
reveal_type(random_st.randint(I_i8_high_open, dtype=np.int64))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(random_st.randint(I_i8_low, I_i8_high_open, dtype=np.int64))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]
reveal_type(random_st.randint(-9223372036854775808, I_i8_high_open, dtype=np.int64))  # E: ndarray[Any, dtype[signedinteger[typing._64Bit]]]

reveal_type(random_st._bit_generator)  # E: BitGenerator

reveal_type(random_st.bytes(2))  # E: bytes

reveal_type(random_st.choice(5))  # E: int
reveal_type(random_st.choice(5, 3))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.choice(5, 3, replace=True))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.choice(5, 3, p=[1 / 5] * 5))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.choice(5, 3, p=[1 / 5] * 5, replace=False))  # E: ndarray[Any, dtype[{int_}]]

reveal_type(random_st.choice(["pooh", "rabbit", "piglet", "Christopher"]))  # E: Any
reveal_type(random_st.choice(["pooh", "rabbit", "piglet", "Christopher"], 3))  # E: ndarray[Any, Any]
reveal_type(random_st.choice(["pooh", "rabbit", "piglet", "Christopher"], 3, p=[1 / 4] * 4))  # E: ndarray[Any, Any]
reveal_type(random_st.choice(["pooh", "rabbit", "piglet", "Christopher"], 3, replace=True))  # E: ndarray[Any, Any]
reveal_type(random_st.choice(["pooh", "rabbit", "piglet", "Christopher"], 3, replace=False, p=np.array([1 / 8, 1 / 8, 1 / 2, 1 / 4])))  # E: ndarray[Any, Any]

reveal_type(random_st.dirichlet([0.5, 0.5]))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.dirichlet(np.array([0.5, 0.5])))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.dirichlet(np.array([0.5, 0.5]), size=3))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(random_st.multinomial(20, [1 / 6.0] * 6))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.multinomial(20, np.array([0.5, 0.5])))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.multinomial(20, [1 / 6.0] * 6, size=2))  # E: ndarray[Any, dtype[{int_}]]

reveal_type(random_st.multivariate_normal([0.0], [[1.0]]))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.multivariate_normal([0.0], np.array([[1.0]])))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.multivariate_normal(np.array([0.0]), [[1.0]]))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.multivariate_normal([0.0], np.array([[1.0]])))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(random_st.permutation(10))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.permutation([1, 2, 3, 4]))  # E: ndarray[Any, Any]
reveal_type(random_st.permutation(np.array([1, 2, 3, 4])))  # E: ndarray[Any, Any]
reveal_type(random_st.permutation(D_2D))  # E: ndarray[Any, Any]

reveal_type(random_st.shuffle(np.arange(10)))  # E: None
reveal_type(random_st.shuffle([1, 2, 3, 4, 5]))  # E: None
reveal_type(random_st.shuffle(D_2D))  # E: None

reveal_type(np.random.RandomState(pcg64))  # E: RandomState
reveal_type(np.random.RandomState(0))  # E: RandomState
reveal_type(np.random.RandomState([0, 1, 2]))  # E: RandomState
reveal_type(random_st.__str__())  # E: str
reveal_type(random_st.__repr__())  # E: str
random_st_state = random_st.__getstate__()
reveal_type(random_st_state)  # E: builtins.dict[builtins.str, Any]
reveal_type(random_st.__setstate__(random_st_state))  # E: None
reveal_type(random_st.seed())  # E: None
reveal_type(random_st.seed(1))  # E: None
reveal_type(random_st.seed([0, 1]))  # E: None
random_st_get_state = random_st.get_state()
reveal_type(random_st_state)  # E: builtins.dict[builtins.str, Any]
random_st_get_state_legacy = random_st.get_state(legacy=True)
reveal_type(random_st_get_state_legacy)  # E: Union[builtins.dict[builtins.str, Any], Tuple[builtins.str, ndarray[Any, dtype[unsignedinteger[typing._32Bit]]], builtins.int, builtins.int, builtins.float]]
reveal_type(random_st.set_state(random_st_get_state))  # E: None

reveal_type(random_st.rand())  # E: float
reveal_type(random_st.rand(1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.rand(1, 2))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.randn())  # E: float
reveal_type(random_st.randn(1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.randn(1, 2))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.random_sample())  # E: float
reveal_type(random_st.random_sample(1))  # E: ndarray[Any, dtype[floating[typing._64Bit]]
reveal_type(random_st.random_sample(size=(1, 2)))  # E: ndarray[Any, dtype[floating[typing._64Bit]]

reveal_type(random_st.tomaxint())  # E: int
reveal_type(random_st.tomaxint(1))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(random_st.tomaxint((1,)))  # E: ndarray[Any, dtype[{int_}]]

reveal_type(np.random.set_bit_generator(pcg64))  # E: None
reveal_type(np.random.get_bit_generator())  # E: BitGenerator
