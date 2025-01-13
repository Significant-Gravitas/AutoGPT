
<file_name>autogpt_platform/backend/backend/blocks/exa/contents.md</file_name>

## Exa Contents Block

### What it is
A specialized search block that retrieves detailed document contents using the Exa AI platform's content retrieval system.

### What it does
This block fetches comprehensive content from documents that have been previously identified through searches. It can retrieve full text, relevant highlights, and summaries from specified documents using their unique identifiers.

### How it works
The block connects to Exa's content API using provided credentials, sends a request with specified document IDs and content retrieval settings, and returns the requested document contents. It processes multiple documents simultaneously and can customize the content retrieval based on specific requirements for text length, highlighting, and summarization.

### Inputs
- Credentials: Authentication details required to access the Exa API
- Document IDs: A list of unique identifiers for the documents whose contents you want to retrieve
- Content Retrieval Settings:
  - Text Settings: Controls the maximum number of characters and HTML tag inclusion
  - Highlight Settings: Determines the number of relevant sentences and highlights per URL
  - Summary Settings: Configures how document summaries are generated

### Outputs
- Results: A list containing the retrieved document contents, including requested text, highlights, and summaries for each document ID provided

### Possible use case
A research team needs to analyze multiple academic papers. They first search for relevant papers using Exa's search capability, then use this block to retrieve specific sections, key highlights, and summaries from each paper. The retrieved content can then be used for further analysis or comparison, making it easier to identify relevant information across multiple documents without having to manually read through each one.

### Additional Notes
- The block includes safety features that handle potential errors during the content retrieval process
- Content retrieval settings can be customized for different use cases, from brief overviews to detailed content analysis
- The block is particularly useful when working with large sets of documents where manual content extraction would be time-consuming

