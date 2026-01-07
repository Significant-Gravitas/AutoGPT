# Baas Bot Delete Recording

### What it is
Permanently delete a meeting's recorded data.

### What it does
Permanently delete a meeting's recorded data

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Baas Bot Fetch Meeting Data

### What it is
Retrieve recorded meeting data.

### What it does
Retrieve recorded meeting data

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
| metadata | Meeting metadata and bot information | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Baas Bot Join Meeting

### What it is
Deploy a bot to join and record a meeting.

### What it does
Deploy a bot to join and record a meeting

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
| timeouts | Automatic leave timeouts configuration | Dict[str, True] | No |
| extra | Custom metadata to attach to the bot | Dict[str, True] | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| bot_id | UUID of the deployed bot | str |
| join_response | Full response from join operation | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Baas Bot Leave Meeting

### What it is
Remove a bot from an ongoing meeting.

### What it does
Remove a bot from an ongoing meeting

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---
