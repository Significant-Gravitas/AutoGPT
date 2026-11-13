# Hacker News Items
<!-- MANUAL: file_description -->
Read one Hacker News story, comment, poll or job with its comment thread. The block gets the thread from HN Search (hn.algolia.com), and the live points, comment count and ranking of the replies from Hacker News's official API (hacker-news.firebaseio.com). Neither needs an account or API key.
<!-- END MANUAL -->

## Hacker News Get Item

### What it is
Get a Hacker News story, comment, poll or job by its id or link, with its comments as a list in reading order. Each comment is followed by its replies, and direct replies come in Hacker News's ranking.

### How it works
<!-- MANUAL: how_it_works -->
Takes an id such as `8863` or a link such as `https://news.ycombinator.com/item?id=8863`, and makes two requests at once: HN Search's `items` endpoint, which returns the item and its whole comment tree in one call, and the official API's `item` endpoint. HN Search keeps replies oldest first, not in the order Hacker News shows them, so the block puts the item's direct replies in the order the official API ranks them. Deeper replies stay oldest first. Each comment is followed by its replies, and `depth` is 1 for a direct reply. Deleted comments are left out but their replies are kept, with `parent_id` still pointing at the deleted comment. `max_comments` (up to 1,000) cuts the list, while `comment_count` counts every comment in the thread. With `include_comments` off you get only the item and its count.

HN Search adds new items and comments a minute or two after they are posted, and doesn't keep killed (dead) ones. So `points`, `num_comments` and `comment_count` come from the official API, which is live: the count is Hacker News's own, or the number of comments found if that is higher. The newest comments can be missing from the list for a minute or two while the count already includes them. When HN Search doesn't have the item at all, the block reads it from the official API and fetches up to `max_comments` comments one request each, 10 at a time, in Hacker News's order at every level. A comment has no `title` or `url`; `story_id` names its story. A poll comes back without its options. Text comes back as plain text, and times are in UTC.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| item | The item's id, such as 8863, or its link, such as https://news.ycombinator.com/item?id=8863 | str | Yes |
| include_comments | Also return the item's comments | bool | No |
| max_comments | Most comments to return, in reading order. comment_count still counts them all. | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| item | The story, comment, poll or job | HackerNewsItem |
| comments | Its comments in reading order: each comment followed by its replies, with direct replies in Hacker News's ranking | List[HackerNewsComment] |
| comment | Each comment | HackerNewsComment |
| comment_count | How many comments the item has, including any that max_comments left out. Hacker News's own count when it is higher. | int |

### Possible use case
<!-- MANUAL: use_case -->
**Discussion Summary**: Read the comments on a launch post and have an AI block sum up what people liked and what they asked for.

**Reply Tracking**: Check a comment you posted for new replies.

**Story Details**: Feed ids from Hacker News Search or Hacker News Get Stories into this block to read each story's text and top comments.
<!-- END MANUAL -->

---
