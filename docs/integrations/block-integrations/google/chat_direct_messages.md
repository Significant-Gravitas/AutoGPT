# Google Chat Direct Messages
<!-- MANUAL: file_description -->
Blocks for Google Chat direct messages: find the user's existing direct message with someone, or open one. Find Direct Message only reads, with the `chat.spaces.readonly` scope. Start Direct Message can create the conversation, with the `chat.spaces.create` scope. Google classes both scopes as sensitive.

Setup: the Google Chat API must be enabled in the Google Cloud project behind AutoGPT's Google sign-in. Chat API calls that create something, like Start Direct Message, also need a Google Chat app (name, avatar and description) configured in that project ([Configure the Google Chat API](https://developers.google.com/workspace/chat/configure-chat-api)).
<!-- END MANUAL -->

## Google Chat Find Direct Message

### What it is
Find the user's existing Google Chat direct message with a person, by email address or user ID, and get its ID for reading or sending messages.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Chat API `spaces.findDirectMessage` endpoint with the person as `users/<email or ID>` and returns the direct message between them and the user.

If the user has never messaged that person in Google Chat, or Google can't find them, the block stops with an error that points to Google Chat Start Direct Message. An empty person input stops with an input error.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| person | The person's email address or Chat user ID (users/...) | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| space | The direct message conversation | ChatSpace |

### Possible use case
<!-- MANUAL: use_case -->
**Meeting Prep**: Read the recent history with a colleague before a one-to-one meeting.

**Customer Updates**: Find the direct message with a customer contact and send them a status update.

**Conversation Check**: Check whether the user has already talked with someone in Google Chat before reaching out another way.
<!-- END MANUAL -->

---

## Google Chat Start Direct Message

### What it is
Open a Google Chat direct message with a person, by email address or user ID: returns the existing conversation, or creates an empty one. Nothing is sent. Pass the result to Google Chat Send Message.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Chat API `spaces.setup` endpoint with the person as the only member. If a direct message with them already exists, Google returns it; otherwise Google creates an empty one. Nothing is sent, and Google Chat doesn't list a new direct message until a message is sent in it.

Creating a conversation needs a Google Chat app configured in the Google Cloud project behind AutoGPT's Google sign-in; without one, the block reports that this setup is missing. Google refuses to create the direct message if either person has blocked the other.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| person | The person's email address or Chat user ID (users/...) | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| space | The direct message conversation, existing or new | ChatSpace |

### Possible use case
<!-- MANUAL: use_case -->
**First Contact**: Send a reminder to someone the user hasn't messaged in Google Chat before, together with Google Chat Send Message.

**Contact List Outreach**: Get a direct message for each person in a spreadsheet of contacts, whether or not they have talked before.

**Personal Alerts**: Open a direct message with the on-call engineer and send them an alert from a monitoring workflow.
<!-- END MANUAL -->

---
