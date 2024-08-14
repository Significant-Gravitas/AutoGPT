from typing import Any, List, Union, Dict, Optional
import random
from enum import Enum
from collections import defaultdict

from autogpt_server.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from autogpt_server.data.model import SchemaField


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
        data: List[Union[dict, List[Any]]] = SchemaField(
            description="The dataset to sample from. Can be a list of dictionaries or a list of lists.",
            placeholder="[{'id': 1, 'value': 'a'}, {'id': 2, 'value': 'b'}, ...]",
        )
        sample_size: int = SchemaField(
            description="The number of samples to take from the dataset.",
            placeholder="10",
        )
        sampling_method: SamplingMethod = SchemaField(
            description="The method to use for sampling.",
            default=SamplingMethod.RANDOM,
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
            categories={BlockCategory.BASIC},
            input_schema=DataSamplingBlock.Input,
            output_schema=DataSamplingBlock.Output,
            test_input={
                "data": [
                    {"id": i, "value": chr(97 + i), "group": i % 3} for i in range(10)
                ],
                "sample_size": 3,
                "sampling_method": SamplingMethod.STRATIFIED,
                "random_seed": 42,
                "stratify_key": "group",
            },
            test_output=[
                (
                    "sampled_data",
                    [
                        {"id": 2, "value": "c", "group": 2},
                        {"id": 1, "value": "b", "group": 1},
                        {"id": 3, "value": "d", "group": 0},
                    ],
                ),
                ("sample_indices", [2, 1, 3]),
            ],
        )

    def run(self, input_data: Input) -> BlockOutput:
        if input_data.random_seed is not None:
            random.seed(input_data.random_seed)

        data_size = len(input_data.data)

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
            for i, item in enumerate(input_data.data):
                strata[str(item[int(input_data.stratify_key)])].append(i)

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
            weights = [
                item[input_data.weight_key]
                for item in input_data.data
                if isinstance(item, dict) and input_data.weight_key in item
            ]
            if not weights:
                raise ValueError("Weight key not found in data items.")
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
            for i, item in enumerate(input_data.data):
                if isinstance(item, dict):
                    cluster_value = item.get(input_data.cluster_key)
                elif isinstance(item, list):
                    try:
                        cluster_value = item[int(input_data.cluster_key)]
                    except (IndexError, ValueError):
                        raise ValueError(
                            f"Invalid cluster_key '{input_data.cluster_key}' for list data."
                        )
                else:
                    raise TypeError("Data items must be either dictionaries or lists.")
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

        sampled_data = [input_data.data[i] for i in indices]

        yield "sampled_data", sampled_data
        yield "sample_indices", indices
