import heapq
import random
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField


class SamplingMethod(str, Enum):
    RANDOM = "random"
    SYSTEMATIC = "systematic"
    TOP = "top"
    BOTTOM = "bottom"
    STRATIFIED = "stratified"
    WEIGHTED = "weighted"
    RESERVOIR = "reservoir"
    CLUSTER = "cluster"


class DataSamplingBlock(Block):
    class Input(BlockSchemaInput):
        data: Union[Dict[str, Any], List[Union[dict, List[Any]]]] = SchemaField(
            description="The dataset to sample from. Can be a single dictionary, a list of dictionaries, or a list of lists.",
            placeholder="{'id': 1, 'value': 'a'} or [{'id': 1, 'value': 'a'}, {'id': 2, 'value': 'b'}, ...]",
        )
        sample_size: int = SchemaField(
            description="The number of samples to take from the dataset.",
            placeholder="10",
            default=10,
        )
        sampling_method: SamplingMethod = SchemaField(
            description="The method to use for sampling.",
            default=SamplingMethod.RANDOM,
        )
        accumulate: bool = SchemaField(
            description="Whether to accumulate data before sampling.",
            default=False,
        )
        random_seed: Optional[int] = SchemaField(
            description="Seed for random number generator (optional).",
            default=None,
        )
        stratify_key: Optional[str] = SchemaField(
            description="Key to use for stratified sampling (required for stratified sampling).",
            default=None,
        )
        weight_key: Optional[str] = SchemaField(
            description="Key to use for weighted sampling (required for weighted sampling).",
            default=None,
        )
        cluster_key: Optional[str] = SchemaField(
            description="Key to use for cluster sampling (required for cluster sampling).",
            default=None,
        )

    class Output(BlockSchemaOutput):
        sampled_data: List[Union[dict, List[Any]]] = SchemaField(
            description="The sampled subset of the input data."
        )
        sample_indices: List[int] = SchemaField(
            description="The indices of the sampled data in the original dataset."
        )

    def __init__(self):
        super().__init__(
            id="4a448883-71fa-49cf-91cf-70d793bd7d87",
            description="This block samples data from a given dataset using various sampling methods.",
            categories={BlockCategory.LOGIC},
            input_schema=DataSamplingBlock.Input,
            output_schema=DataSamplingBlock.Output,
            test_input={
                "data": [
                    {"id": i, "value": chr(97 + i), "group": i % 3} for i in range(10)
                ],
                "sample_size": 3,
                "sampling_method": SamplingMethod.STRATIFIED,
                "accumulate": False,
                "random_seed": 42,
                "stratify_key": "group",
            },
            test_output=[
                (
                    "sampled_data",
                    [
                        {"id": 0, "value": "a", "group": 0},
                        {"id": 1, "value": "b", "group": 1},
                        {"id": 8, "value": "i", "group": 2},
                    ],
                ),
                ("sample_indices", [0, 1, 8]),
            ],
            effect=BlockEffect.NONE,
        )
        self.accumulated_data = []

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        if input_data.sample_size < 0:
            raise ValueError(
                f"Sample size ({input_data.sample_size}) cannot be negative."
            )

        if input_data.accumulate:
            if isinstance(input_data.data, dict):
                self.accumulated_data.append(input_data.data)
            elif isinstance(input_data.data, list):
                self.accumulated_data.extend(input_data.data)
            else:
                raise ValueError(f"Unsupported data type: {type(input_data.data)}")

            # If we don't have enough data yet, return without sampling
            if len(self.accumulated_data) < input_data.sample_size:
                return

            data_to_sample = self.accumulated_data
        else:
            # If not accumulating, use the input data directly
            data_to_sample = (
                input_data.data
                if isinstance(input_data.data, list)
                else [input_data.data]
            )

        # Its own generator: runs on other threads cannot reseed it mid-sample.
        rng = random.Random(input_data.random_seed)

        data_size = len(data_to_sample)

        if input_data.sample_size > data_size:
            raise ValueError(
                f"Sample size ({input_data.sample_size}) cannot be larger than the dataset size ({data_size})."
            )

        indices = []

        if input_data.sampling_method == SamplingMethod.RANDOM:
            indices = rng.sample(range(data_size), input_data.sample_size)
        elif input_data.sampling_method == SamplingMethod.SYSTEMATIC:
            step = data_size // input_data.sample_size
            start = rng.randint(0, step - 1)
            indices = list(range(start, data_size, step))[: input_data.sample_size]
        elif input_data.sampling_method == SamplingMethod.TOP:
            indices = list(range(input_data.sample_size))
        elif input_data.sampling_method == SamplingMethod.BOTTOM:
            indices = list(range(data_size - input_data.sample_size, data_size))
        elif input_data.sampling_method == SamplingMethod.STRATIFIED:
            if not input_data.stratify_key:
                raise ValueError(
                    "Stratify key must be provided for stratified sampling."
                )
            strata = defaultdict(list)
            for i, item in enumerate(data_to_sample):
                if isinstance(item, dict):
                    strata_value = item.get(input_data.stratify_key)
                elif hasattr(item, input_data.stratify_key):
                    strata_value = getattr(item, input_data.stratify_key)
                else:
                    raise ValueError(
                        f"Stratify key '{input_data.stratify_key}' not found in item {item}"
                    )

                if strata_value is None:
                    raise ValueError(
                        f"Stratify value for key '{input_data.stratify_key}' is None"
                    )

                strata[str(strata_value)].append(i)

            # Calculate the number of samples to take from each stratum
            stratum_sizes = {
                k: max(1, int(len(v) / data_size * input_data.sample_size))
                for k, v in strata.items()
            }

            # Adjust sizes to ensure we get exactly sample_size samples
            _balance_stratum_sizes(
                stratum_sizes, strata, data_size, input_data.sample_size
            )

            for stratum, size in stratum_sizes.items():
                indices.extend(rng.sample(strata[stratum], size))
        elif input_data.sampling_method == SamplingMethod.WEIGHTED:
            if not input_data.weight_key:
                raise ValueError("Weight key must be provided for weighted sampling.")
            weights = []
            for item in data_to_sample:
                if isinstance(item, dict):
                    weight = item.get(input_data.weight_key)
                elif hasattr(item, input_data.weight_key):
                    weight = getattr(item, input_data.weight_key)
                else:
                    raise ValueError(
                        f"Weight key '{input_data.weight_key}' not found in item {item}"
                    )

                if weight is None:
                    raise ValueError(
                        f"Weight value for key '{input_data.weight_key}' is None"
                    )
                try:
                    weights.append(float(weight))
                except ValueError:
                    raise ValueError(
                        f"Weight value '{weight}' cannot be converted to a number"
                    )

            if not weights:
                raise ValueError(
                    f"No valid weights found using key '{input_data.weight_key}'"
                )

            indices = rng.choices(
                range(data_size), weights=weights, k=input_data.sample_size
            )
        elif input_data.sampling_method == SamplingMethod.RESERVOIR:
            indices = list(range(input_data.sample_size))
            for i in range(input_data.sample_size, data_size):
                j = rng.randint(0, i)
                if j < input_data.sample_size:
                    indices[j] = i
        elif input_data.sampling_method == SamplingMethod.CLUSTER:
            if not input_data.cluster_key:
                raise ValueError("Cluster key must be provided for cluster sampling.")
            clusters = defaultdict(list)
            for i, item in enumerate(data_to_sample):
                if isinstance(item, dict):
                    cluster_value = item.get(input_data.cluster_key)
                elif hasattr(item, input_data.cluster_key):
                    cluster_value = getattr(item, input_data.cluster_key)
                else:
                    raise TypeError(
                        f"Item {item} does not have the cluster key '{input_data.cluster_key}'"
                    )

                clusters[str(cluster_value)].append(i)

            # Take clusters in random order until we have enough samples
            cluster_order = list(clusters)
            rng.shuffle(cluster_order)
            for cluster in cluster_order:
                if len(indices) >= input_data.sample_size:
                    break
                indices.extend(clusters[cluster])

            # If we have more samples than needed, randomly remove some
            if len(indices) > input_data.sample_size:
                indices = rng.sample(indices, input_data.sample_size)
        else:
            raise ValueError(f"Unknown sampling method: {input_data.sampling_method}")

        sampled_data = [data_to_sample[i] for i in indices]

        # Clear accumulated data after sampling if accumulation is enabled
        if input_data.accumulate:
            self.accumulated_data = []

        yield "sampled_data", sampled_data
        yield "sample_indices", indices


def _balance_stratum_sizes(
    stratum_sizes: dict[str, int],
    strata: dict[str, list[int]],
    data_size: int,
    sample_size: int,
) -> None:
    """Top up or trim *stratum_sizes* in place until they sum to *sample_size*.

    Each step adjusts the stratum that is furthest below its proportional
    share (top-up) or the largest one (trim), the earliest stratum winning a
    tie. Heaps pick that stratum at each step.
    """
    order = {key: position for position, key in enumerate(stratum_sizes)}
    total = sum(stratum_sizes.values())

    if total > sample_size:
        largest = [(-size, order[key], key) for key, size in stratum_sizes.items()]
        heapq.heapify(largest)
        for _ in range(total - sample_size):
            negative_size, position, key = heapq.heappop(largest)
            stratum_sizes[key] -= 1
            heapq.heappush(largest, (negative_size + 1, position, key))
        return

    def shortfall(key: str) -> float:
        return len(strata[key]) / data_size * sample_size - stratum_sizes[key]

    # Only top up strata that still have unsampled items, or random.sample()
    # asks for more than the stratum has. Total capacity is
    # data_size >= sample_size, so a candidate always exists.
    furthest_below = [
        (-shortfall(key), order[key], key)
        for key in stratum_sizes
        if stratum_sizes[key] < len(strata[key])
    ]
    heapq.heapify(furthest_below)
    for _ in range(sample_size - total):
        if not furthest_below:
            return
        _, position, key = heapq.heappop(furthest_below)
        stratum_sizes[key] += 1
        if stratum_sizes[key] < len(strata[key]):
            heapq.heappush(furthest_below, (-shortfall(key), position, key))
