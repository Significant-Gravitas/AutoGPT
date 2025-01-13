
<file_name>autogpt_platform/backend/backend/blocks/jina/chunking.md</file_name>

## Jina Chunking

### What it is
A specialized block that divides text into smaller, manageable segments using Jina AI's segmentation service.

### What it does
Takes large pieces of text and breaks them down into smaller chunks while maintaining context and meaning. It can also provide information about the tokens (individual words and characters) in each chunk if requested.

### How it works
1. Receives one or more texts as input
2. Connects to Jina AI's segmentation service
3. Processes each text by breaking it into smaller chunks based on the specified maximum length
4. Optionally analyzes and returns token information for each chunk
5. Returns the chunked texts and optional token information

### Inputs
- texts: A list of text documents that need to be broken down into smaller chunks
- credentials: Jina AI authentication credentials required to access the segmentation service
- max_chunk_length: The maximum number of characters allowed in each chunk (default is 1000)
- return_tokens: A toggle to indicate whether token information should be included in the output (default is False)

### Outputs
- chunks: A list of smaller text segments created from the input texts
- tokens: (Optional) A list containing token information for each chunk, only provided if return_tokens is set to True

### Possible use case
A content manager needs to process a large document (like a book or long article) for analysis or summarization. The Jina Chunking block can break down this document into smaller, more manageable pieces while maintaining context. These chunks can then be processed individually by other AI tools for tasks like summarization, translation, or sentiment analysis.

### Notes
- This block falls under both AI and TEXT categories
- Authentication with Jina AI's service is required through API credentials
- The block processes texts sequentially and maintains the order of chunks
- Token information can be useful for more detailed text analysis or processing tasks
