# Twitter List Follows
<!-- MANUAL: file_description -->
Blocks for following and unfollowing Twitter/X lists.
<!-- END MANUAL -->

## Twitter Follow List

### What it is
This block follows a specified Twitter list for the authenticated user.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Twitter API v2 via Tweepy to follow a Twitter List. Following a list adds it to your Lists tab and shows tweets from list members in your timeline when viewing that list.

The block authenticates using OAuth 2.0 with list write permissions and sends a POST request to add the follow relationship. Returns a success indicator confirming the list was followed.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| list_id | The ID of the List to follow | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the follow was successful | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Curated Content Discovery**: Follow lists curated by industry experts to access filtered content streams.

**Topic Monitoring**: Follow lists focused on specific topics to stay informed without following individual accounts.

**Research Organization**: Follow competitor or industry lists to monitor activity without cluttering your main timeline.
<!-- END MANUAL -->

---

## Twitter Unfollow List

### What it is
This block unfollows a specified Twitter list for the authenticated user.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Twitter API v2 via Tweepy to unfollow a Twitter List. The list is removed from your Lists tab but remains accessible if it's public.

The block authenticates using OAuth 2.0 with list write permissions and sends a DELETE request to remove the follow relationship. Returns a success indicator confirming the list was unfollowed.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| list_id | The ID of the List to unfollow | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the unfollow was successful | bool |

### Possible use case
<!-- MANUAL: use_case -->
**List Cleanup**: Unfollow lists that are no longer relevant or useful to your interests.

**Focus Management**: Reduce information overload by unfollowing less important lists.

**Account Organization**: Clean up your followed lists as part of regular account maintenance.
<!-- END MANUAL -->

---
