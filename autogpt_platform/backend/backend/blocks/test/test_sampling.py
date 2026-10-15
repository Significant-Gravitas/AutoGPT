import pytest

from backend.blocks.sampling import DataSamplingBlock, SamplingMethod


async def _run_weighted(data, sample_size, random_seed=42):
    block = DataSamplingBlock()
    input_data = block.Input(
        data=data,
        sample_size=sample_size,
        sampling_method=SamplingMethod.WEIGHTED,
        weight_key="w",
        random_seed=random_seed,
    )
    outputs = {}
    async for name, value in block.run(input_data):
        outputs[name] = value
    return outputs


async def test_weighted_sampling_returns_distinct_items():
    # Issue #15280: random.choices samples with replacement, so weighted
    # output could contain the same item several times while every other
    # method returns distinct items.
    data = [{"w": float(n + 1), "i": n} for n in range(12)]
    outputs = await _run_weighted(data, sample_size=7)
    indices = outputs["sample_indices"]

    assert len(indices) == 7
    assert (
        len(set(indices)) == 7
    ), f"weighted sampling returned duplicate indices: {indices}"
    sampled = outputs["sampled_data"]
    assert sampled == [data[i] for i in indices]


async def test_weighted_sampling_skips_zero_weight_items():
    data = [
        {"w": 0.0, "i": 0},
        {"w": 0.0, "i": 1},
        {"w": 1.0, "i": 2},
        {"w": 1.0, "i": 3},
    ]
    outputs = await _run_weighted(data, sample_size=2)
    assert sorted(outputs["sample_indices"]) == [2, 3]


async def test_weighted_sampling_size_exceeding_eligible_count_raises():
    data = [{"w": 0.0, "i": 0}, {"w": 1.0, "i": 1}, {"w": 1.0, "i": 2}]
    block = DataSamplingBlock()
    input_data = block.Input(
        data=data,
        sample_size=3,
        sampling_method=SamplingMethod.WEIGHTED,
        weight_key="w",
        random_seed=42,
    )
    with pytest.raises(ValueError):
        async for _ in block.run(input_data):
            pass


async def test_weighted_sampling_all_zero_weights_raises():
    data = [{"w": 0.0, "i": 0}, {"w": 0.0, "i": 1}]
    block = DataSamplingBlock()
    input_data = block.Input(
        data=data,
        sample_size=1,
        sampling_method=SamplingMethod.WEIGHTED,
        weight_key="w",
        random_seed=42,
    )
    with pytest.raises(ValueError):
        async for _ in block.run(input_data):
            pass


async def test_random_sampling_still_returns_distinct_items():
    # Guard against collateral damage: the plain random method already
    # samples without replacement and must keep doing so.
    data = [{"w": 1.0, "i": n} for n in range(12)]
    block = DataSamplingBlock()
    input_data = block.Input(
        data=data,
        sample_size=7,
        sampling_method=SamplingMethod.RANDOM,
        random_seed=42,
    )
    outputs = {}
    async for name, value in block.run(input_data):
        outputs[name] = value
    indices = outputs["sample_indices"]
    assert len(indices) == 7
    assert len(set(indices)) == 7
