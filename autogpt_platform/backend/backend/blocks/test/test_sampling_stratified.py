"""Regression tests for #15279: stratified top-up picked a stratum with no
spare items and random.sample() raised 'Sample larger than population'."""

import random
from collections import Counter

import pytest

from backend.blocks.sampling import DataSamplingBlock, SamplingMethod


async def _sample(data, k, seed=0):
    block = DataSamplingBlock()
    out = {}
    async for name, value in block.run(
        block.Input(
            data=data,
            sample_size=k,
            sampling_method=SamplingMethod.STRATIFIED,
            stratify_key="g",
            random_seed=seed,
        )
    ):
        out[name] = value
    return out


@pytest.mark.asyncio
async def test_reported_case_does_not_oversample_a_stratum():
    data = [{"g": 0}, {"g": 1}, {"g": 1}, {"g": 2}, {"g": 2}]
    out = await _sample(data, 4)

    indices = out["sample_indices"]
    assert len(indices) == 4
    assert len(set(indices)) == 4
    population = Counter(item["g"] for item in data)
    sampled = Counter(data[i]["g"] for i in indices)
    assert all(sampled[g] <= population[g] for g in sampled)


@pytest.mark.asyncio
async def test_random_strata_never_raise():
    rng = random.Random(1234)
    for _ in range(300):
        n = rng.randint(1, 30)
        data = [{"g": rng.randint(0, 6)} for _ in range(n)]
        k = rng.randint(1, n)
        out = await _sample(data, k, seed=rng.randint(0, 10_000))
        indices = out["sample_indices"]
        assert len(indices) == k
        assert len(set(indices)) == k
