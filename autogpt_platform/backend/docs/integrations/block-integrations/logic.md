# Logic
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Calculator

### What it is
Performs a mathematical operation on two numbers.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| operation | Choose the math operation you want to perform | "Add" \| "Subtract" \| "Multiply" \| "Divide" \| "Power" | Yes |
| a | Enter the first number (A) | float | Yes |
| b | Enter the second number (B) | float | Yes |
| round_result | Do you want to round the result to a whole number? | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| result | The result of your calculation | float |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Condition

### What it is
Handles conditional logic based on comparison operators

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| value1 | Enter the first value for comparison | Value1 | Yes |
| operator | Choose the comparison operator | "==" \| "!=" \| ">" \| "<" \| ">=" \| "<=" | Yes |
| value2 | Enter the second value for comparison | Value2 | Yes |
| yes_value | (Optional) Value to output if the condition is true. If not provided, value1 will be used. | Yes Value | No |
| no_value | (Optional) Value to output if the condition is false. If not provided, value1 will be used. | No Value | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| result | The result of the condition evaluation (True or False) | bool |
| yes_output | The output value if the condition is true | Yes Output |
| no_output | The output value if the condition is false | No Output |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Count Items

### What it is
Counts the number of items in a collection.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| collection | Enter the collection you want to count. This can be a list, dictionary, string, or any other iterable. | Collection | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| count | The number of items in the collection | int |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Data Sampling

### What it is
This block samples data from a given dataset using various sampling methods.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| data | The dataset to sample from. Can be a single dictionary, a list of dictionaries, or a list of lists. | Dict[str, Any] \| List[Dict[str, Any] \| List[Any]] | Yes |
| sample_size | The number of samples to take from the dataset. | int | No |
| sampling_method | The method to use for sampling. | "random" \| "systematic" \| "top" \| "bottom" \| "stratified" \| "weighted" \| "reservoir" \| "cluster" | No |
| accumulate | Whether to accumulate data before sampling. | bool | No |
| random_seed | Seed for random number generator (optional). | int | No |
| stratify_key | Key to use for stratified sampling (required for stratified sampling). | str | No |
| weight_key | Key to use for weighted sampling (required for weighted sampling). | str | No |
| cluster_key | Key to use for cluster sampling (required for cluster sampling). | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| sampled_data | The sampled subset of the input data. | List[Dict[str, Any] \| List[Any]] |
| sample_indices | The indices of the sampled data in the original dataset. | List[int] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## If Input Matches

### What it is
Handles conditional logic based on comparison operators

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| input | The input to match against | Input | Yes |
| value | The value to output if the input matches | Value | Yes |
| yes_value | The value to output if the input matches | Yes Value | No |
| no_value | The value to output if the input does not match | No Value | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| result | The result of the condition evaluation (True or False) | bool |
| yes_output | The output value if the condition is true | Yes Output |
| no_output | The output value if the condition is false | No Output |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Pinecone Init

### What it is
Initializes a Pinecone index

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| index_name | Name of the Pinecone index | str | Yes |
| dimension | Dimension of the vectors | int | No |
| metric | Distance metric for the index | str | No |
| cloud | Cloud provider for serverless | str | No |
| region | Region for serverless | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| index | Name of the initialized Pinecone index | str |
| message | Status message | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Pinecone Insert

### What it is
Upload data to a Pinecone index

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| index | Initialized Pinecone index | str | Yes |
| chunks | List of text chunks to ingest | List[Any] | Yes |
| embeddings | List of embeddings corresponding to the chunks | List[Any] | Yes |
| namespace | Namespace to use in Pinecone | str | No |
| metadata | Additional metadata to store with each vector | Dict[str, Any] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| upsert_response | Response from Pinecone upsert operation | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Pinecone Query

### What it is
Queries a Pinecone index

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query_vector | Query vector | List[Any] | Yes |
| namespace | Namespace to query in Pinecone | str | No |
| top_k | Number of top results to return | int | No |
| include_values | Whether to include vector values in the response | bool | No |
| include_metadata | Whether to include metadata in the response | bool | No |
| host | Host for pinecone | str | No |
| idx_name | Index name for pinecone | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| results | Query results from Pinecone | Results |
| combined_results | Combined results from Pinecone | Combined Results |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Step Through Items

### What it is
Iterates over a list or dictionary and outputs each item.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| items | The list or dictionary of items to iterate over | List[Any] | No |
| items_object | The list or dictionary of items to iterate over | Dict[str, Any] | No |
| items_str | The list or dictionary of items to iterate over | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| item | The current item in the iteration | Item |
| key | The key or index of the current item in the iteration | Key |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
