# Jina Chunking
<!-- MANUAL: file_description -->
Blocks for splitting text into semantic chunks using Jina AI.
<!-- END MANUAL -->

## Jina Chunking

### What it is
Chunks texts using Jina AI's segmentation service

### How it works
<!-- MANUAL: how_it_works -->
This block uses Jina AI's segmentation service to split texts into semantically meaningful chunks. Unlike simple splitting by character count, Jina's chunking preserves semantic coherence, making it ideal for RAG applications.

Configure maximum chunk length and optionally return token information for each chunk.
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
**RAG Preprocessing**: Chunk documents for retrieval-augmented generation systems.

**Embedding Preparation**: Split long texts into optimal chunks for embedding generation.

**Document Processing**: Break down large documents for analysis or storage in vector databases.
<!-- END MANUAL -->

---
