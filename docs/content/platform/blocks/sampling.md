# Data Sampling

## What it is
The Data Sampling block is a tool for selecting a subset of data from a larger dataset using various sampling methods.

## What it does
This block takes a dataset as input and returns a smaller sample of that data based on specified criteria. It supports multiple sampling methods, allowing users to choose the most appropriate technique for their needs.

## How it works
The block processes the input data and applies the chosen sampling method to select a subset of items. It can work with different data structures and supports data accumulation for scenarios where data is received in batches.

## Inputs
| Input | Description |
|-------|-------------|
| Data | The dataset to sample from. This can be a single dictionary, a list of dictionaries, or a list of lists. |
| Sample Size | The number of items to select from the dataset. |
| Sampling Method | The technique used to select the sample. Options include random, systematic, top, bottom, stratified, weighted, reservoir, and cluster sampling. |
| Accumulate | A flag indicating whether to accumulate data before sampling. This is useful for scenarios where data is received in batches. |
| Random Seed | An optional value to ensure reproducible random sampling. |
| Stratify Key | The key to use for stratified sampling (required when using the stratified sampling method). |
| Weight Key | The key to use for weighted sampling (required when using the weighted sampling method). |
| Cluster Key | The key to use for cluster sampling (required when using the cluster sampling method). |

## Outputs
| Output | Description |
|--------|-------------|
| Sampled Data | The selected subset of the input data. |
| Sample Indices | The indices of the sampled items in the original dataset. |

## Possible use case
A data scientist working with a large customer dataset wants to create a representative sample for analysis. They could use this Data Sampling block to select a smaller subset of customers using stratified sampling, ensuring that the sample maintains the same proportions of different customer segments as the full dataset.