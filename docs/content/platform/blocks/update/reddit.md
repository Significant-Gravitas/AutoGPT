
## RSS Feed Reader

### What it is
A specialized tool that monitors and retrieves content from RSS feeds, which are web feeds that allow users to access updates from websites in a standardized format.

### What it does
This block continuously checks a specified RSS feed URL for new entries within a defined time period. When it finds new content, it extracts and formats relevant information such as the title, link, description, publication date, author, and categories.

### How it works
The block operates by:
1. Connecting to a specified RSS feed URL
2. Checking for entries published within the user-defined time period
3. Processing each new entry to extract relevant information
4. Delivering the formatted entries one at a time
5. If set to run continuously, it repeats this process after waiting for the specified polling interval

### Inputs
- RSS URL: The web address of the RSS feed you want to monitor
- Time Period: The number of minutes to look back for new entries (default is 1440 minutes, or 24 hours)
- Polling Rate: How often the block should check for new entries (in seconds)
- Run Continuously: Whether the block should keep checking for new entries (true) or run only once (false)

### Outputs
- RSS Entry: A structured package of information containing:
  - Title: The headline or name of the entry
  - Link: The web address where the full content can be found
  - Description: A summary or preview of the content
  - Publication Date: When the entry was published
  - Author: Who created the content
  - Categories: Topics or tags associated with the entry

### Possible use case
A news monitoring system that needs to track updates from multiple news sources. For example, a PR firm could use this block to monitor industry news websites for mentions of their clients, automatically collecting new articles as they're published. The block could run continuously, checking every few minutes for updates, and feed the information to other blocks for analysis or notification purposes.

