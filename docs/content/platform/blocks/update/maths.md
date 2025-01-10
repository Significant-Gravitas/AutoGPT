

## Pinecone Initialization Block

### What it is
A block that sets up and initializes a new Pinecone vector database index, or connects to an existing one.

### What it does
Creates a new searchable database index in Pinecone if it doesn't exist, or connects to an existing index with the specified name.

### How it works
The block connects to Pinecone using your API credentials, checks if the requested index exists, and either creates a new one with your specified settings or connects to the existing one.

### Inputs
- Credentials: Your Pinecone API key for authentication
- Index Name: The name you want to give to your database index
- Dimension: The size of the vectors to be stored (default: 768)
- Metric: The method used to measure similarity between vectors (default: cosine)
- Cloud: The cloud provider for the serverless deployment (default: aws)
- Region: The geographical region for data storage (default: us-east-1)

### Outputs
- Index: The name of the initialized Pinecone index
- Message: A status message indicating whether a new index was created or an existing one was connected

### Possible use case
Setting up a new semantic search system where you need to initialize a database to store and search through document embeddings.

## Pinecone Query Block

### What it is
A block that searches through a Pinecone vector database to find similar vectors.

### What it does
Searches the specified Pinecone index using a query vector to find the most similar items in the database.

### How it works
Takes a query vector, searches the Pinecone database for the most similar vectors, and returns both the raw results and a combined text of the matches.

### Inputs
- Credentials: Your Pinecone API key for authentication
- Query Vector: The vector to search for in the database
- Namespace: Optional subdivision of your index (default: empty)
- Top K: Number of results to return (default: 3)
- Include Values: Whether to include vector values in results (default: false)
- Include Metadata: Whether to include metadata in results (default: true)
- Host: Pinecone host address (default: empty)
- Index Name: Name of the Pinecone index to query

### Outputs
- Results: Raw query results including matches and combined text
- Combined Results: A concatenated string of all matched text

### Possible use case
Implementing a semantic search feature where users can find similar documents or text passages based on meaning rather than exact keyword matches.

## Pinecone Insert Block

### What it is
A block that adds new data to a Pinecone vector database.

### What it does
Uploads text chunks and their corresponding vector embeddings to a specified Pinecone index.

### How it works
Takes text chunks and their vector representations, adds unique identifiers and metadata, then uploads them to the specified Pinecone index.

### Inputs
- Credentials: Your Pinecone API key for authentication
- Index: Name of the Pinecone index to upload to
- Chunks: List of text pieces to store
- Embeddings: Vector representations of the text chunks
- Namespace: Optional subdivision of your index (default: empty)
- Metadata: Additional information to store with each vector (default: empty dictionary)

### Outputs
- Upsert Response: Confirmation message of successful upload

### Possible use case
Building a document storage system where you need to continuously add new documents and their vector representations for later similarity search.

