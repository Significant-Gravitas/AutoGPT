
# Pinecone Blocks Documentation

## Pinecone Initialize

### What it is
A block that sets up and initializes a Pinecone vector database index.

### What it does
Creates a new Pinecone index or connects to an existing one with specified parameters for vector storage and retrieval.

### How it works
The block connects to Pinecone using provided credentials, checks if the requested index exists, and either creates a new one or connects to an existing index with the specified configuration.

### Inputs
- Credentials: Pinecone API key for authentication
- Index Name: Name for the new or existing Pinecone index
- Dimension: Size of the vectors to be stored (default: 768)
- Metric: Distance measurement method for comparing vectors (default: cosine)
- Cloud: Cloud provider for serverless deployment (default: aws)
- Region: Geographic region for the index (default: us-east-1)

### Outputs
- Index: Name of the initialized Pinecone index
- Message: Status message indicating success or failure of the operation

### Possible use case
Setting up a new vector database to store and search through embedded documents in a document management system.

## Pinecone Query

### What it is
A block that searches through vectors stored in a Pinecone index.

### What it does
Performs similarity searches in the Pinecone index using provided query vectors to find the most relevant matches.

### How it works
Takes a query vector and searches the specified Pinecone index for similar vectors, returning the top matches based on similarity scores.

### Inputs
- Credentials: Pinecone API key for authentication
- Query Vector: Vector to search for in the index
- Namespace: Specific section of the index to search (optional)
- Top K: Number of results to return (default: 3)
- Include Values: Whether to include vector values in results (default: false)
- Include Metadata: Whether to include metadata in results (default: true)
- Host: Pinecone host address
- Index Name: Name of the index to query

### Outputs
- Results: Raw query results including matches and metadata
- Combined Results: Concatenated text from all matching results

### Possible use case
Implementing semantic search in a document repository to find similar documents based on their content.

## Pinecone Insert

### What it is
A block that uploads new data to a Pinecone index.

### What it does
Adds new vectors and their associated metadata to an existing Pinecone index.

### How it works
Takes text chunks and their corresponding embeddings, combines them with metadata, and uploads them to the specified Pinecone index.

### Inputs
- Credentials: Pinecone API key for authentication
- Index: Name of the target Pinecone index
- Chunks: List of text pieces to store
- Embeddings: Vector representations of the text chunks
- Namespace: Specific section of the index to store data (optional)
- Metadata: Additional information to store with the vectors

### Outputs
- Upsert Response: Confirmation message of successful data upload

### Possible use case
Building a knowledge base by uploading and indexing document chunks for later retrieval and searching.
