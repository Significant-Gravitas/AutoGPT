
<file_name>autogpt_platform/backend/backend/blocks/exa/search.md</file_name>

## Exa Search

### What it is
A powerful web search tool that leverages Exa's advanced search API to find relevant content across the internet.

### What it does
This block performs comprehensive web searches with highly customizable parameters, allowing users to filter results by dates, domains, text patterns, and content types. It can automatically enhance search queries and retrieve specific types of content based on user preferences.

### How it works
The block takes a search query and various filtering parameters, sends them to Exa's search API, and returns a list of relevant search results. It can automatically improve search queries and allows fine-grained control over what content is included or excluded from the results.

### Inputs
- Credentials: Authentication details required to access the Exa API
- Query: The search term or phrase you want to look up
- Use Auto Prompt: Option to enable automatic query enhancement for better results (default: enabled)
- Type: Specific type of search to perform
- Category: Specific category to limit the search within
- Number of Results: How many search results to return (default: 10)
- Include Domains: List of websites to specifically include in the search
- Exclude Domains: List of websites to exclude from the search
- Start Crawl Date: Beginning date for when content was indexed
- End Crawl Date: End date for when content was indexed
- Start Published Date: Beginning date for when content was published
- End Published Date: End date for when content was published
- Include Text: Specific text patterns that must appear in results
- Exclude Text: Text patterns that should not appear in results
- Content Settings: Specific settings for content retrieval

### Outputs
- Results: A list of search results matching the specified criteria, including relevant metadata and content details

### Possible use cases
- Market research: Track mentions of a company or product across the web within specific date ranges
- Content monitoring: Follow specific topics across selected websites while excluding unwanted sources
- Academic research: Find scholarly content within particular date ranges and domains
- Competitive analysis: Monitor competitor activities by searching for their mentions across specific time periods
- News aggregation: Collect news articles about specific topics from preferred sources while excluding others

### Advanced Features
- Date filtering for both crawled and published content
- Domain inclusion and exclusion lists
- Text pattern matching and filtering
- Customizable content retrieval settings
- Automatic query enhancement

