# Jina Embeddings
<!-- MANUAL: file_description -->
Blocks for generating text embeddings using Jina AI.
<!-- END MANUAL -->

## Jina Embedding

### What it is
Generates embeddings using Jina AI

### How it works
<!-- MANUAL: how_it_works -->
This block generates vector embeddings for text using Jina AI's embedding models. Embeddings are numerical representations that capture semantic meaning, enabling similarity search and clustering.

Optionally specify which Jina model to use for embedding generation.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| texts | List of texts to embed | List[Any] | Yes |
| model | Jina embedding model to use | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| embeddings | List of embeddings | List[Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Semantic Search**: Generate embeddings to enable semantic similarity search over documents.

**Vector Database**: Create embeddings for storage in vector databases like Pinecone or Weaviate.

**Document Clustering**: Embed documents to cluster similar content or find related items.
<!-- END MANUAL -->

---
