# Google Calendar
<!-- MANUAL: file_description -->
Blocks for creating and reading events from Google Calendar.
<!-- END MANUAL -->

## Google Calendar Create Event

### What it is
This block creates a new event in Google Calendar with customizable parameters.

### How it works
<!-- MANUAL: how_it_works -->
This block creates events in Google Calendar via the Google Calendar API. It handles various event parameters including timing, location, guest invitations, Google Meet links, and recurring schedules. The block authenticates using your connected Google account credentials.

When you specify guests, they receive email invitations (if notifications are enabled). The Google Meet option adds a video conference link to the event automatically.
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
**Automated Meeting Scheduling**: Create calendar events when appointments are booked through a form or scheduling system.

**Event Reminders**: Schedule events with custom reminder notifications for team deadlines or milestones.

**Team Coordination**: Create recurring meetings with Google Meet links when onboarding new team members.
<!-- END MANUAL -->

---

## Google Calendar Read Events

### What it is
Retrieves upcoming events from a Google Calendar with filtering options

### How it works
<!-- MANUAL: how_it_works -->
This block fetches upcoming events from Google Calendar using the Calendar API. It retrieves events within a specified time range, with options to filter by search term or exclude declined events. Pagination support allows handling large numbers of events.

Events are returned with details like title, time, location, and attendees. Use 'primary' as the calendar_id to access your main calendar.
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
**Daily Briefings**: Fetch today's events to generate a morning summary or prepare for upcoming meetings.

**Schedule Conflicts**: Check for overlapping events before scheduling new appointments.

**Meeting Preparation**: Retrieve upcoming meetings to pre-load relevant documents or send reminders.
<!-- END MANUAL -->

---
