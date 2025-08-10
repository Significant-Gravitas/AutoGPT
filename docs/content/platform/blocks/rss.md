# Read RSS Feed

## What it is
A block that retrieves and processes entries from an RSS feed.

## What it does
This block reads entries from a specified RSS feed URL, filters them based on a given time period, and outputs the entries one by one.

## How it works
The block connects to the provided RSS feed URL, fetches the feed content, and processes each entry. It checks if the entry's publication date falls within the specified time period and, if so, formats and outputs the entry information.

## Inputs
| Input | Description |
|-------|-------------|
| RSS URL | The web address of the RSS feed you want to read from |
| Time Period | The number of minutes to look back for new entries, relative to when the block starts running |
| Polling Rate | How often (in seconds) the block should check for new entries |
| Run Continuously | Whether the block should keep checking for new entries indefinitely or just run once |

## Outputs
| Output | Description |
|--------|-------------|
| Entry | An RSS feed item containing the following information: |
| | - Title: The headline or name of the item |
| | - Link: The web address where the full item can be found |
| | - Description: A brief summary or excerpt of the item |
| | - Publication Date: When the item was published |
| | - Author: Who wrote or created the item |
| | - Categories: Topics or tags associated with the item |

## Possible use case
A news aggregator application could use this block to continuously monitor multiple RSS feeds from different news sources. The application could then display the latest news items to users, categorized by topic and sorted by publication date.