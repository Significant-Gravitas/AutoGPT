# Google Calendar Create Event

### What it is
This block creates a new event in Google Calendar with customizable parameters.

### What it does
This block creates a new event in Google Calendar with customizable parameters.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| event_title | Title of the event | str | Yes |
| location | Location of the event | str | No |
| description | Description of the event | str | No |
| timing | Specify when the event starts and ends | Timing | No |
| calendar_id | Calendar ID (use 'primary' for your main calendar) | str | No |
| guest_emails | Email addresses of guests to invite | List[str] | No |
| send_notifications | Send email notifications to guests | bool | No |
| add_google_meet | Include a Google Meet video conference link | bool | No |
| recurrence | Whether the event repeats | Recurrence | No |
| reminder_minutes | When to send reminders before the event | List[int] | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| event_id | ID of the created event | str |
| event_link | Link to view the event in Google Calendar | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Calendar Read Events

### What it is
Retrieves upcoming events from a Google Calendar with filtering options.

### What it does
Retrieves upcoming events from a Google Calendar with filtering options

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| calendar_id | Calendar ID (use 'primary' for your main calendar) | str | No |
| max_events | Maximum number of events to retrieve | int | No |
| start_time | Retrieve events starting from this time | str (date-time) | No |
| time_range_days | Number of days to look ahead for events | int | No |
| search_term | Optional search term to filter events by | str | No |
| page_token | Page token from previous request to get the next batch of events. You can use this if you have lots of events you want to process in a loop | str | No |
| include_declined_events | Include events you've declined | bool | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| events | List of calendar events in the requested time range | List[CalendarEvent] |
| event | One of the calendar events in the requested time range | CalendarEvent |
| next_page_token | Token for retrieving the next page of events if more exist | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
