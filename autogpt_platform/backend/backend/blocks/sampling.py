import random
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
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
    class Input(BlockSchema):
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

    class Output(BlockSchema):
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
        )
        self.accumulated_data = []

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
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

        if input_data.random_seed is not None:
            random.seed(input_data.random_seed)

        data_size = len(data_to_sample)

        if input_data.sample_size > data_size:
            raise ValueError(
                f"Sample size ({input_data.sample_size}) cannot be larger than the dataset size ({data_size})."
            )

        indices = []

        if input_data.sampling_method == SamplingMethod.RANDOM:
            indices = random.sample(range(data_size), input_data.sample_size)
        elif input_data.sampling_method == SamplingMethod.SYSTEMATIC:
            step = data_size // input_data.sample_size
            start = random.randint(0, step - 1)
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
            while sum(stratum_sizes.values()) != input_data.sample_size:
                if sum(stratum_sizes.values()) < input_data.sample_size:
                    stratum_sizes[
                        max(stratum_sizes, key=lambda k: stratum_sizes[k])
                    ] += 1
                else:
                    stratum_sizes[
                        max(stratum_sizes, key=lambda k: stratum_sizes[k])
                    ] -= 1

            for stratum, size in stratum_sizes.items():
                indices.extend(random.sample(strata[stratum], size))
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

            indices = random.choices(
                range(data_size), weights=weights, k=input_data.sample_size
            )
        elif input_data.sampling_method == SamplingMethod.RESERVOIR:
            indices = list(range(input_data.sample_size))
            for i in range(input_data.sample_size, data_size):
                j = random.randint(0, i)
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

            # Randomly select clusters until we have enough samples
            selected_clusters = []
            while (
                sum(len(clusters[c]) for c in selected_clusters)
                < input_data.sample_size
            ):
                available_clusters = [c for c in clusters if c not in selected_clusters]
                if not available_clusters:
                    break
                selected_clusters.append(random.choice(available_clusters))

            for cluster in selected_clusters:
                indices.extend(clusters[cluster])

            # If we have more samples than needed, randomly remove some
            if len(indices) > input_data.sample_size:
                indices = random.sample(indices, input_data.sample_size)
        else:
            raise ValueError(f"Unknown sampling method: {input_data.sampling_method}")

        sampled_data = [data_to_sample[i] for i in indices]

        # Clear accumulated data after sampling if accumulation is enabled
        if input_data.accumulate:
            self.accumulated_data = []

        yield "sampled_data", sampled_data
        yield "sample_indices", indices
