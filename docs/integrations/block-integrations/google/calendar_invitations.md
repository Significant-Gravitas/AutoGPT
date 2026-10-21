# Google Calendar Invitations
<!-- MANUAL: file_description -->
Blocks for answering Google Calendar invitations sent to the user. They use the `calendar.events` scope. The organizer sees a reply as soon as it is saved, so these blocks are marked as irreversible actions.
<!-- END MANUAL -->

## Google Calendar Respond To Event

### What it is
Accept, decline or tentatively accept a Google Calendar invitation, with an optional note to the organizer.

### How it works
<!-- MANUAL: how_it_works -->
Reads the event to find the connected account in its guest list, then saves the reply with `events.patch`. The patch sends only that one guest entry with `attendeesOmitted` set, so the rest of the guest list is left alone. The reply is accepted, declined or tentative, with an optional note, and Google emails the organizer unless `notify_organizer` is off.

If the account isn't on the guest list (for example, it was invited through a group), the block fails with a message saying so instead of guessing.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| event_id | ID of the event, e.g. from Search Events or Get Event | str | Yes |
| calendar_id | Calendar the event is on: 'primary' for the main calendar, or an ID from List Calendars | str | No |
| response | Your reply | "accepted" \| "declined" \| "tentative" | Yes |
| comment | Optional note for the organizer | str | No |
| notify_organizer | Email the organizer about your reply | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| event | The event with your reply | CalendarEventDetails |

### Possible use case
<!-- MANUAL: use_case -->
**Auto-Accept Team Meetings**: Accept invitations from teammates automatically so they land on the calendar.

**Holiday Declines**: Decline meetings during time off with a note saying when you're back.

**Tentative Holds**: Mark clashing invitations as tentative until someone decides which to keep.
<!-- END MANUAL -->

---
