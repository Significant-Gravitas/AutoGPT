# Baas Bots
<!-- MANUAL: file_description -->
Blocks for deploying and managing meeting recording bots using the BaaS (Bot as a Service) API.
<!-- END MANUAL -->

## Baas Bot Delete Recording

### What it is
Permanently delete a meeting's recorded data

### How it works
<!-- MANUAL: how_it_works -->
This block permanently deletes the recorded data for a meeting bot using the BaaS (Bot as a Service) API. The deletion is irreversible and removes all associated recording files and transcripts.

Provide the bot_id from a previous recording session to delete that specific meeting's data.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| bot_id | UUID of the bot whose data to delete | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| deleted | Whether the data was successfully deleted | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Privacy Compliance**: Delete recordings to comply with data retention policies or user requests.

**Storage Management**: Clean up old recordings to manage storage costs.

**Post-Processing Cleanup**: Delete recordings after extracting needed information.
<!-- END MANUAL -->

---

## Baas Bot Fetch Meeting Data

### What it is
Retrieve recorded meeting data

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves recorded meeting data including video URL, transcript, and metadata from a completed bot session. The video URL is time-limited and should be downloaded promptly.

Enable include_transcripts to receive the full meeting transcript with speaker identification and timestamps.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| bot_id | UUID of the bot whose data to fetch | str | Yes |
| include_transcripts | Include transcript data in response | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| mp4_url | URL to download the meeting recording (time-limited) | str |
| transcript | Meeting transcript data | List[Any] |
| metadata | Meeting metadata and bot information | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Meeting Summarization**: Retrieve transcripts for AI summarization and action item extraction.

**Recording Archive**: Download and store meeting recordings for compliance or reference.

**Analytics**: Extract meeting metadata for participation and duration analytics.
<!-- END MANUAL -->

---

## Baas Bot Join Meeting

### What it is
Deploy a bot to join and record a meeting

### How it works
<!-- MANUAL: how_it_works -->
This block deploys a recording bot to join a video meeting (Zoom, Google Meet, Teams). Configure the bot's display name, avatar, and entry message. The bot joins, records, and transcribes the meeting.

Use webhooks to receive notifications when the meeting ends and recordings are ready.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| meeting_url | The URL of the meeting the bot should join | str | Yes |
| bot_name | Display name for the bot in the meeting | str | Yes |
| bot_image | URL to an image for the bot's avatar (16:9 ratio recommended) | str | No |
| entry_message | Chat message the bot will post upon entry | str | No |
| reserved | Use a reserved bot slot (joins 4 min before meeting) | bool | No |
| start_time | Unix timestamp (ms) when bot should join | int | No |
| webhook_url | URL to receive webhook events for this bot | str | No |
| timeouts | Automatic leave timeouts configuration | Dict[str, Any] | No |
| extra | Custom metadata to attach to the bot | Dict[str, Any] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| bot_id | UUID of the deployed bot | str |
| join_response | Full response from join operation | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Automated Recording**: Record meetings automatically without requiring host intervention.

**Meeting Assistant**: Deploy bots to take notes and transcribe customer or team meetings.

**Compliance Recording**: Ensure all meetings are recorded for compliance or quality assurance.
<!-- END MANUAL -->

---

## Baas Bot Leave Meeting

### What it is
Remove a bot from an ongoing meeting

### How it works
<!-- MANUAL: how_it_works -->
This block removes a recording bot from an ongoing meeting. Use this when you need to stop recording before the meeting naturally ends.

The bot leaves gracefully and recording data becomes available for retrieval.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| bot_id | UUID of the bot to remove from meeting | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| left | Whether the bot successfully left | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Early Termination**: Stop recording when a meeting transitions to an off-record discussion.

**Time-Based Recording**: Leave after capturing a specific portion of a meeting.

**Error Recovery**: Remove and redeploy bots when issues occur during recording.
<!-- END MANUAL -->

---
