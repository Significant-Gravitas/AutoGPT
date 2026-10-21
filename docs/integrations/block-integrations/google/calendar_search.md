# Google Calendar Search
<!-- MANUAL: file_description -->
Blocks for finding things in Google Calendar: the calendars on the user's list, events that match a keyword, and single events by ID. They only read, with the `calendar.readonly` scope the Read Events block already asks for. Events come back with exact times, guests and each guest's reply, and their IDs work in the Update, Delete and Respond To Event blocks.
<!-- END MANUAL -->

## Google Calendar Get Event

### What it is
Get one Google Calendar event by its ID: title, exact times, location, description, video link, organizer, guests and their replies.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Calendar API `events.get` endpoint for the event ID on the chosen calendar (`primary` by default). The event comes back with its title, exact start and end (with UTC offset, or dates for all-day events), location, description, video link, organizer and every guest with their reply. `my_response` shows how the connected account answered, when it is a guest.

An ID Google can't find, or an event on a calendar the account can't see, fails with a not-found message. A deleted event fails with a message saying so, instead of returning an empty cancelled event.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| event_id | ID of the event, e.g. from Search Events, Read Events or Create Event | str | Yes |
| calendar_id | Calendar the event is on: 'primary' for the main calendar, or an ID from List Calendars | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| event | The event with its time, guests and their replies | CalendarEventDetails |

### Possible use case
<!-- MANUAL: use_case -->
**Meeting Prep**: Pull an event's agenda, location and guest list to draft a briefing before the meeting.

**RSVP Check**: See who has accepted, declined or not answered before deciding whether to reschedule.

**Follow-Up Emails**: Read the guest list after a meeting ends and send each guest a follow-up email.
<!-- END MANUAL -->

---

## Google Calendar List Calendars

### What it is
List the user's Google calendars with each one's ID, name, time zone and the user's access level. Use it to turn a calendar name into the calendar ID the other Calendar blocks need.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Calendar API `calendarList.list` endpoint, reading every page, and returns each calendar's ID, name, description, time zone and the account's access role (owner, writer, reader or freeBusyReader), with the main calendar first. `name_contains` filters by name or ID, ignoring case; `writable_only` keeps only calendars the account can add events to; hidden calendars are left out unless `include_hidden` is on.

The name is the one the user sees in Google Calendar, which can differ from the owner's name for the calendar. Pass a calendar's `id` to the other Calendar blocks' `calendar_id` input.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name_contains | Only calendars whose name or ID contains this text (not case-sensitive) | str | No |
| writable_only | Only calendars the account can add or change events on | bool | No |
| include_hidden | Include calendars the user has hidden from their list | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| calendars | Matching calendars, the main calendar first | List[CalendarInfo] |
| calendar | Each matching calendar | CalendarInfo |

### Possible use case
<!-- MANUAL: use_case -->
**Calendar Picker**: Turn a name like "the team calendar" into the calendar ID the other Calendar blocks need.

**Shared Calendar Audit**: List every calendar the account can edit to review what it has access to.

**Time Zone Check**: Read a calendar's time zone before scheduling events on it.
<!-- END MANUAL -->

---

## Google Calendar Search Events

### What it is
Search a Google Calendar for events by keyword, across past and future events, optionally within a time range. Returns each event with its time, location, video link, guests and their replies.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Calendar API `events.list` endpoint with your words as its free-text search, which matches event titles, descriptions, locations, and guests' names and emails. Unlike Read Events it has no default time window, so it finds past events too. `after` and `before` narrow the search to events that overlap that window; times without a UTC offset are read in your profile time zone.

A repeating event appears once, as the series. Turn on `expand_recurring` to get each occurrence instead, sorted by start time. An empty search is rejected, because Read Events is the block for listing a time range. Use `next_page_token` to page through large result sets.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | Words to look for in event titles, descriptions, locations, and guests' names and emails | str | Yes |
| calendar_id | Calendar to search: 'primary' for the main calendar, or an ID from List Calendars | str | No |
| after | Only events that end after this time. Empty includes past events. Times without a UTC offset use your profile time zone. | str (date-time) | No |
| before | Only events that start before this time. Empty means no limit. | str (date-time) | No |
| expand_recurring | List every occurrence of repeating events, in start-time order. Off lists each repeating event once. | bool | No |
| max_results | Maximum number of events to return | int | No |
| page_token | Page token from a previous search, to get the next page | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| events | Matching events with their times, guests and replies | List[CalendarEventDetails] |
| event | Each matching event | CalendarEventDetails |
| next_page_token | Token for the next page, when there are more results | str |

### Possible use case
<!-- MANUAL: use_case -->
**Find an Event to Change**: Look up "Quarterly planning" by name to get the ID for Update Event or Delete Event.

**Meeting History**: Find every past meeting with a customer before a renewal call.

**Duplicate Check**: Search for an event before creating it, so a workflow doesn't book the same meeting twice.
<!-- END MANUAL -->

---
