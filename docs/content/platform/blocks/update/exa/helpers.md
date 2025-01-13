
<file_name>autogpt_platform/backend/backend/blocks/exa/similar.md</file_name>

## Exa Find Similar

### What it is
A specialized search block that finds web content similar to a given URL using the Exa AI platform.

### What it does
This block analyzes a provided webpage URL and discovers other web content that is thematically or contextually similar, allowing users to find related articles, blog posts, or web pages.

### How it works
The block connects to Exa's findSimilar API service, sends the provided URL along with any specified search parameters, and returns a list of similar content matches. It filters and ranks the results based on various criteria such as dates, domains, and text patterns.

### Inputs
- Credentials: Authentication details required to access the Exa API
- URL: The web address for which you want to find similar content
- Number of Results: How many similar items you want to receive (default: 10)
- Include Domains: List of specific website domains to include in the search
- Exclude Domains: List of website domains to exclude from the search
- Start Crawl Date: Beginning date for when content was indexed
- End Crawl Date: Final date for when content was indexed
- Start Published Date: Beginning date for when content was published
- End Published Date: Final date for when content was published
- Include Text: Specific text patterns to look for (limited to 1 string, maximum 5 words)
- Exclude Text: Text patterns to filter out (limited to 1 string, maximum 5 words)
- Content Settings: Configuration for how content should be retrieved and processed

### Outputs
- Results: A list of similar documents, each containing:
  - Title of the content
  - URL where it can be found
  - Publication date
  - Author information
  - Similarity score indicating how closely it matches the input URL

### Possible use case
A content researcher wants to find articles similar to a compelling piece they've just read about artificial intelligence. They input the article's URL, set the published date range to the last six months, and exclude certain competitor domains. The block returns a curated list of related articles that match their criteria, helping them discover new perspectives on the same topic and build a comprehensive research collection.

