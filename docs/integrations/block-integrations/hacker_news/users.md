# Hacker News Users
<!-- MANUAL: file_description -->
Read a Hacker News user's public profile from Hacker News's official API (hacker-news.firebaseio.com), which needs no account or API key.
<!-- END MANUAL -->

## Hacker News Get User

### What it is
Get a Hacker News user's profile by username: karma, when the account was created, the about text and how many items they have posted. Uses Hacker News's official API, which needs no account.

### How it works
<!-- MANUAL: how_it_works -->
Calls the official API's `user` endpoint. Usernames are case-sensitive (`pg` exists, `PG` doesn't), and a profile link such as `https://news.ycombinator.com/user?id=pg` works too. The API answers `null` for a user that doesn't exist, and the block turns that into an error.

`about` comes back as plain text, with paragraphs as blank lines and links as full URLs. `submission_count` counts the stories, comments and polls on the user's submission list. The profile holds only what Hacker News shows publicly.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| username | The username, such as pg (case-sensitive), or a profile link such as https://news.ycombinator.com/user?id=pg | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| username | The username, spelled the way Hacker News has it | str |
| karma | The user's karma | int |
| created_at | When the account was created, in ISO 8601 (UTC) | str |
| about | The user's about text as plain text; empty if they wrote none | str |
| submission_count | How many stories, comments and polls the user has posted | int |
| profile_url | Link to the profile on Hacker News | str |

### Possible use case
<!-- MANUAL: use_case -->
**Lead Context**: Before replying to someone who mentioned your product, check their karma and what they say about themselves.

**Expert Finder**: Look up the authors of the best comments on a topic to see who they are.

**New Account Check**: Flag mentions that come from accounts created in the last few days.
<!-- END MANUAL -->

---
