
## Data Sampling Block

### What it is
A specialized tool designed to extract representative samples from datasets using various sampling methods.

### What it does
Takes a dataset and creates a smaller subset using one of eight different sampling techniques, ensuring the selected data meets specific criteria or patterns based on the chosen method.

### How it works
The block receives input data and a set of parameters, then applies the specified sampling method to select a subset of the data. It can either process data immediately or accumulate it over time before sampling. The block supports various sampling strategies, from simple random selection to more complex methods like stratified or weighted sampling.

### Inputs
- Data: The source dataset that you want to sample from. Can be structured as a single record or a list of records
- Sample Size: The number of items you want in your final sample
- Sampling Method: Choose from eight different approaches:
  - Random: Selects items completely at random
  - Systematic: Picks items at regular intervals
  - Top: Takes items from the beginning
  - Bottom: Takes items from the end
  - Stratified: Samples from different groups proportionally
  - Weighted: Selects items based on their importance (weight)
  - Reservoir: Maintains a running sample as data streams in
  - Cluster: Samples entire groups of related items
- Accumulate: Option to collect data over multiple runs before sampling
- Random Seed: Optional number to ensure consistent sampling results
- Stratify Key: Field name used to group data for stratified sampling
- Weight Key: Field name containing importance values for weighted sampling
- Cluster Key: Field name identifying groups for cluster sampling

### Outputs
- Sampled Data: The selected subset of records from your input data
- Sample Indices: The positions of the selected items in the original dataset

### Possible use cases
1. Quality Control: Randomly selecting products from a production line for inspection
2. Market Research: Ensuring survey responses come from different demographic groups using stratified sampling
3. Data Analysis: Creating smaller, representative datasets for testing or analysis
4. Customer Feedback: Selecting customer reviews based on rating weights
5. Geographic Studies: Sampling data by location clusters
6. Stream Processing: Maintaining a representative sample of continuous data feeds

### Tips for Usage
- Choose Random sampling for simple, unbiased selection
- Use Stratified sampling when you need representation from different groups
- Apply Weighted sampling when some items are more important than others
- Select Cluster sampling when items naturally group together
- Enable Accumulate when working with streaming data
- Set a Random Seed when you need reproducible results

### Common Applications
- Social Science Research
- Manufacturing Quality Control
- Customer Survey Analysis
- Scientific Studies
- Big Data Processing
- Machine Learning Dataset Preparation
- Population Studies
- Performance Testing

