# Google Calendar Events
<!-- MANUAL: file_description -->
Blocks that change or delete existing Google Calendar events. They use the `calendar.events` scope, which covers events on every calendar the user can edit. Both can email the guests, so both are marked as irreversible actions and wait for approval when sensitive-action review is on.
<!-- END MANUAL -->

## Google Calendar Delete Event

### What it is
Delete a Google Calendar event, or one occurrence of a repeating event, and optionally email the guests a cancellation.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Calendar API `events.delete` endpoint. Deleting a repeating event's series ID removes every occurrence, while deleting one occurrence's ID (as Search Events returns with `expand_recurring` on) removes only that one. `send_updates` chooses who Google emails a cancellation to: every guest, only guests who don't use Google Calendar, or no one.

Deleting an event that is already gone fails with a message saying so, and an unknown ID fails as not found. A blank event ID is rejected before anything is sent to Google.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| event_id | ID of the event, e.g. from Search Events or Get Event | str | Yes |
| calendar_id | Calendar the event is on: 'primary' for the main calendar, or an ID from List Calendars | str | No |
| send_updates | Who Google emails a cancellation to | "all" \| "external_only" \| "none" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| event_id | ID of the deleted event | str |

### Possible use case
<!-- MANUAL: use_case -->
**Cancel on Request**: Delete a booking when a customer cancels through a form, emailing the guests.

**Clean Up Test Events**: Remove placeholder events a workflow created, without emailing anyone.

**Skip One Occurrence**: Delete the single occurrence of a weekly meeting that falls on a public holiday.
<!-- END MANUAL -->

---

## Google Calendar Update Event

### What it is
Change a Google Calendar event's title, time, location, description, guests or Google Meet link. Only the fields you set change, and guests can be emailed about the update.

### How it works
<!-- MANUAL: how_it_works -->
Reads the event with `events.get`, works out what changes, and sends only those fields with `events.patch`. The patch carries the event's ETag in an `If-Match` header, so if someone edits the event in between, Google refuses the write and the block asks you to run it again instead of overwriting their change. Moving only the start keeps the event's length. Times without a UTC offset are read in `time_zone`, or your profile time zone, which then becomes the event's time zone. All-day events need both a new start and a new end.

Changing guests rewrites the whole guest list, keeping everyone else and their replies, so the block refuses when Google has hidden part of the list from the account. Inviting someone already invited, removing someone who isn't, or asking for a Meet link the event already has changes nothing and sends no email. Empty text fields keep their current values.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| event_id | ID of the event, e.g. from Search Events or Get Event | str | Yes |
| calendar_id | Calendar the event is on: 'primary' for the main calendar, or an ID from List Calendars | str | No |
| title | New title. Empty keeps the current title. | str | No |
| start_time | New start time. Changing only the start keeps the event's length. | str (date-time) | No |
| end_time | New end time | str (date-time) | No |
| time_zone | Time zone for new times given without a UTC offset, e.g. Europe/London. Empty uses your profile time zone. | str | No |
| location | New location. Empty keeps the current location. | str | No |
| description | New description. Empty keeps the current description. | str | No |
| add_guest_emails | Email addresses to invite | List[str] | No |
| remove_guest_emails | Email addresses to take off the guest list | List[str] | No |
| add_google_meet | Add a Google Meet link if the event doesn't have one | bool | No |
| send_updates | Who Google emails about the change | "all" \| "external_only" \| "none" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| event | The event after the change | CalendarEventDetails |

### Possible use case
<!-- MANUAL: use_case -->
**Reschedule Meetings**: Move a meeting to a time found by Suggest Meeting Times and email the guests.

**Add a Video Link**: Add a Google Meet link to an in-person meeting that has gone remote.

**Manage Attendance**: Invite new team members to a meeting and remove people who have left the team.
<!-- END MANUAL -->

---
