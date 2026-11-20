import builtins
import random
from typing import Any

import pytest
import pytest_mock

from backend.blocks import sampling
from backend.blocks.sampling import DataSamplingBlock, SamplingMethod


async def _sample(**fields: Any) -> dict[str, Any]:
    block = DataSamplingBlock()
    return {
        name: value
        async for name, value in block.run(DataSamplingBlock.Input(**fields))
    }


def _count_max_work(mocker: pytest_mock.MockFixture, limit: int) -> list[int]:
    """Spy on the module's ``max``: counts the values it compares and fails the
    test once they pass *limit*."""
    scanned = [0]

    def counting_max(*args, **kwargs):
        values = list(args[0]) if len(args) == 1 else list(args)
        scanned[0] += len(values)
        if scanned[0] > limit:
            raise AssertionError(f"max() compared more than {limit} values")
        return builtins.max(values, **kwargs)

    mocker.patch("backend.blocks.sampling.max", counting_max, create=True)
    return scanned


@pytest.mark.parametrize("method", list(SamplingMethod))
async def test_negative_sample_size_is_refused(
    mocker: pytest_mock.MockFixture, method: SamplingMethod
):
    _count_max_work(mocker, limit=10_000)
    data = [{"g": i % 3, "w": 1} for i in range(10)]

    with pytest.raises(ValueError, match="cannot be negative"):
        await _sample(
            data=data,
            sample_size=-5,
            sampling_method=method,
            stratify_key="g",
            weight_key="w",
            cluster_key="g",
        )


async def test_sample_larger_than_the_data_is_refused():
    with pytest.raises(ValueError, match="cannot be larger"):
        await _sample(data=[{"g": 1}], sample_size=2)


async def test_stratified_trim_reaches_the_sample_size_over_many_strata(
    mocker: pytest_mock.MockFixture,
):
    size = 3_000
    scanned = _count_max_work(mocker, limit=10 * size)
    # One item per stratum: every stratum starts at 1, so all but one are
    # trimmed back to reach a sample of 1.
    data = [{"g": i} for i in range(size)]

    out = await _sample(
        data=data,
        sample_size=1,
        sampling_method=SamplingMethod.STRATIFIED,
        stratify_key="g",
        random_seed=0,
    )

    assert len(out["sample_indices"]) == 1
    assert scanned[0] <= 10 * size


async def test_cluster_sampling_draws_each_cluster_once(
    mocker: pytest_mock.MockFixture,
):
    size = 2_000
    limit = 10 * size
    picked_from = [0]

    def counting(method):
        def wrapper(rng, population, *args, **kwargs):
            picked_from[0] += len(population)
            if picked_from[0] > limit:
                raise AssertionError(f"picked from more than {limit} values")
            return method(rng, population, *args, **kwargs)

        return wrapper

    for name in ("choice", "shuffle", "sample"):
        mocker.patch.object(random.Random, name, counting(getattr(random.Random, name)))
    data = [{"c": i} for i in range(size)]

    out = await _sample(
        data=data,
        sample_size=size,
        sampling_method=SamplingMethod.CLUSTER,
        cluster_key="c",
        random_seed=0,
    )

    assert sorted(out["sample_indices"]) == list(range(size))
    assert picked_from[0] <= limit


async def test_cluster_sampling_takes_whole_clusters_until_full():
    data = [{"c": i // 3} for i in range(30)]

    out = await _sample(
        data=data,
        sample_size=6,
        sampling_method=SamplingMethod.CLUSTER,
        cluster_key="c",
        random_seed=7,
    )

    indices = out["sample_indices"]
    assert len(indices) == 6
    assert len(set(indices)) == 6
    clusters = {data[i]["c"] for i in indices}
    assert all(sum(data[i]["c"] == c for i in indices) == 3 for c in clusters)


async def test_cluster_sampling_takes_clusters_in_seeded_shuffle_order():
    clusters = 10
    data = [{"c": i // 2} for i in range(2 * clusters)]

    async def first_cluster(seed: int) -> int:
        out = await _sample(
            data=data,
            sample_size=2,
            sampling_method=SamplingMethod.CLUSTER,
            cluster_key="c",
            random_seed=seed,
        )
        picked = {data[i]["c"] for i in out["sample_indices"]}
        assert len(picked) == 1
        return picked.pop()

    for seed in (0, 7, 42):
        order = [str(c) for c in range(clusters)]
        random.Random(seed).shuffle(order)
        assert await first_cluster(seed) == int(order[0])
        assert await first_cluster(seed) == await first_cluster(seed)

    firsts = {await first_cluster(seed) for seed in range(300)}
    assert firsts == set(range(clusters))


async def test_seeded_sample_is_kept_when_another_run_reseeds_mid_sample(
    mocker: pytest_mock.MockFixture,
):
    data = [{"g": i % 4} for i in range(200)]
    fields: dict[str, Any] = {
        "data": data,
        "sample_size": 40,
        "sampling_method": SamplingMethod.STRATIFIED,
        "stratify_key": "g",
        "random_seed": 1,
    }
    balance = sampling._balance_stratum_sizes

    def another_run_reseeds(*args: Any) -> None:
        # What a seeded run on another worker thread does between this run's
        # seeding and its draws.
        random.seed(12345)
        balance(*args)

    state = random.getstate()
    try:
        expected = await _sample(**fields)
        mocker.patch.object(sampling, "_balance_stratum_sizes", another_run_reseeds)
        assert await _sample(**fields) == expected
    finally:
        random.setstate(state)


async def test_seeded_sample_leaves_the_module_random_state_alone():
    state = random.getstate()

    await _sample(data=[{"v": i} for i in range(100)], sample_size=10, random_seed=3)

    assert random.getstate() == state


def _reference_stratum_sizes(
    strata: dict[str, list[int]], data_size: int, sample_size: int
) -> dict[str, int]:
    """The original step-by-step adjustment the block used to run."""
    sizes = {
        k: builtins.max(1, int(len(v) / data_size * sample_size))
        for k, v in strata.items()
    }
    while sum(sizes.values()) != sample_size:
        if sum(sizes.values()) < sample_size:
            candidates = [k for k in sizes if sizes[k] < len(strata[k])]
            sizes[
                builtins.max(
                    candidates,
                    key=lambda k: len(strata[k]) / data_size * sample_size - sizes[k],
                )
            ] += 1
        else:
            sizes[builtins.max(sizes, key=lambda k: sizes[k])] -= 1
    return sizes


async def test_stratified_sizes_match_the_original_adjustment():
    rng = random.Random(4321)
    for _ in range(300):
        n = rng.randint(1, 60)
        data = [{"g": rng.randint(0, 12)} for _ in range(n)]
        k = rng.randint(0, n)
        out = await _sample(
            data=data,
            sample_size=k,
            sampling_method=SamplingMethod.STRATIFIED,
            stratify_key="g",
            random_seed=rng.randint(0, 10_000),
        )

        strata: dict[str, list[int]] = {}
        for i, item in enumerate(data):
            strata.setdefault(str(item["g"]), []).append(i)
        expected = _reference_stratum_sizes(strata, n, k)
        got = {key: 0 for key in strata}
        for i in out["sample_indices"]:
            got[str(data[i]["g"])] += 1
        assert got == expected
