
## Data Sampling Block

### What it is
A specialized tool that helps select representative samples from larger datasets using various sampling methods.

### What it does
This block takes a dataset and extracts a smaller subset of data points using different sampling techniques. It can work with different data formats and supports multiple sampling methods to ensure the selected samples meet specific requirements.

### How it works
The block receives input data and a specified sampling method, then applies the chosen method to select a subset of the data. It can either process data immediately or accumulate data over time before sampling. The block supports eight different sampling methods:
- Random: Selects samples completely at random
- Systematic: Picks samples at regular intervals
- Top: Takes samples from the beginning
- Bottom: Takes samples from the end
- Stratified: Ensures samples are taken from different groups proportionally
- Weighted: Selects samples based on specified importance values
- Reservoir: Maintains a representative sample as data streams in
- Cluster: Groups similar items and samples from these groups

### Inputs
- Data: The dataset to sample from, which can be a single item or a list of items
- Sample Size: The number of samples you want to select from the dataset
- Sampling Method: The technique to use for selecting samples (defaults to random)
- Accumulate: Whether to collect data over time before sampling (defaults to false)
- Random Seed: An optional number to ensure consistent sampling results
- Stratify Key: The category field to use for stratified sampling
- Weight Key: The importance field to use for weighted sampling
- Cluster Key: The grouping field to use for cluster sampling

### Outputs
- Sampled Data: The selected subset of data points
- Sample Indices: The positions of the selected items in the original dataset

### Possible use cases
1. Quality Control: Randomly selecting products from a production line for inspection
2. Market Research: Ensuring survey responses come from different demographic groups using stratified sampling
3. Data Analysis: Creating a smaller, manageable dataset for testing analytics processes
4. Stream Processing: Maintaining a representative sample of continuous data using reservoir sampling
5. Customer Feedback: Selecting customer reviews based on purchase amounts using weighted sampling
6. Geographic Analysis: Sampling data from different regions using cluster sampling

