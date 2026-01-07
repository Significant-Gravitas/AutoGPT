# Jina Chunking

### What it is
Chunks texts using Jina AI's segmentation service.

### What it does
Chunks texts using Jina AI's segmentation service

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| texts | List of texts to chunk | List[Any] | Yes |
| max_chunk_length | Maximum length of each chunk | int | No |
| return_tokens | Whether to return token information | bool | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| chunks | List of chunked texts | List[Any] |
| tokens | List of token information for each chunk | List[Any] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
