
# Pinecone Blocks Documentation

## Pinecone Initialize

### What it is
A block that sets up and initializes a Pinecone vector database index.

### What it does
Creates a new Pinecone index if it doesn't exist, or connects to an existing one if it does. This index will be used to store and search through vector data.

### How it works
The block connects to Pinecone using your API credentials, checks if the requested index exists, and either creates a new one with your specified settings or connects to the existing one.

### Inputs
- Credentials: Your Pinecone API key for authentication
- Index Name: The name you want to give to your Pinecone index
- Dimension: The size of the vectors to be stored (default: 768)
- Metric: The method used to measure similarity between vectors (default: cosine)
- Cloud: The cloud provider for serverless deployment (default: aws)
- Region: The geographical region for your index (default: us-east-1)

### Outputs
- Index: The name of the initialized Pinecone index
- Message: A status message indicating whether the index was created or already existed

### Possible use case
Setting up a new vector database to store document embeddings for a semantic search application.

## Pinecone Query

### What it is
A block that searches through vectors stored in a Pinecone index to find similar items.

### What it does
Takes a query vector and searches the Pinecone index to find the most similar vectors, returning the top matches along with their associated metadata.

### How it works
The block connects to your Pinecone index, performs a similarity search using your query vector, and returns the most relevant results based on your specified parameters.

### Inputs
- Credentials: Your Pinecone API key for authentication
- Query Vector: The vector to search for in the index
- Namespace: A specific section of the index to search in (optional)
- Top K: Number of results to return (default: 3)
- Include Values: Whether to include vector values in results (default: false)
- Include Metadata: Whether to include metadata in results (default: true)
- Host: The Pinecone host address
- Index Name: The name of the Pinecone index to query

### Outputs
- Results: The raw query results from Pinecone
- Combined Results: A consolidated version of the results

### Possible use case
Implementing a semantic search feature where users can find similar documents based on meaning rather than exact keyword matches.

## Pinecone Insert

### What it is
A block that uploads new vector data to a Pinecone index.

### What it does
Takes text chunks and their corresponding vector embeddings and stores them in the Pinecone index along with any additional metadata.

### How it works
The block connects to your Pinecone index, processes the provided chunks and embeddings, assigns unique IDs to each vector, and uploads them to the specified namespace in the index.

### Inputs
- Credentials: Your Pinecone API key for authentication
- Index: The name of the initialized Pinecone index
- Chunks: List of text pieces to be stored
- Embeddings: Vector representations of the text chunks
- Namespace: Specific section of the index to store data in (optional)
- Metadata: Additional information to store with the vectors

### Outputs
- Upsert Response: Confirmation message about the upload status

### Possible use case
Uploading a collection of documents to a vector database after converting them into embeddings, allowing them to be searched semantically later.
