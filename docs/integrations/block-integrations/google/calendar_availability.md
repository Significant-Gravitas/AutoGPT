# Google Calendar Availability
<!-- MANUAL: file_description -->
Blocks for finding free time in Google Calendar. They ask Google only for busy times, not event details, using the `calendar.readonly` scope the Read Events block already asks for.
<!-- END MANUAL -->

## Google Calendar Suggest Meeting Times

### What it is
Suggest meeting times when you and the given people are all free, using Google Calendar free/busy. By default only suggests weekday times between 09:00 and 17:00 in your time zone.

### How it works
<!-- MANUAL: how_it_works -->
Asks the Calendar API `freebusy.query` endpoint when you (your main calendar, unless `include_me` is off) and each attendee are busy during the window, then works out the free periods itself. By default it only considers weekdays from 09:00 to 17:00 in `time_zone`, or your profile time zone; clear `earliest_time` and `latest_time` to allow any hour. Each suggestion starts at the beginning of a free period, rounded up to the next quarter hour, lasts `duration_minutes`, and reports `free_until`, when that free period ends. The window defaults to the next seven days and never starts in the past.

Google only shares free/busy for calendars you can see, usually people in your organization. Calendars it refuses are listed in `unchecked_calendars` and ignored by the suggestions, and the block fails if none of the calendars can be read. It checks at most 50 calendars at once.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| attendee_emails | Email addresses of the people who need to be free. Google only shares free/busy for calendars you can see, usually people in your organization. | List[str] | No |
| include_me | Also check your own main calendar | bool | No |
| duration_minutes | Meeting length in minutes | int | No |
| window_start | Earliest time to suggest. Empty means now. | str (date-time) | No |
| window_end | Time by which the meeting must be over. Empty means 7 days after the start. | str (date-time) | No |
| time_zone | Time zone for the working hours, for window times without a UTC offset and for the suggestions, e.g. Europe/London. Empty uses your profile time zone. | str | No |
| earliest_time | Don't suggest meetings that start before this time of day (HH:MM, 24-hour). Empty allows any time. | str | No |
| latest_time | Meetings must be over by this time of day (HH:MM, 24-hour). Empty allows any time. | str | No |
| include_weekends | Also suggest times on Saturdays and Sundays | bool | No |
| max_suggestions | Maximum number of times to suggest | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| slots | Suggested meeting times, earliest first, at most one per free period | List[MeetingSlot] |
| slot | Each suggested meeting time | MeetingSlot |
| unchecked_calendars | Calendars whose free/busy Google wouldn't share, e.g. people outside your organization. The suggestions ignore them. | List[str] |
| time_zone | Time zone of the suggestions and the working hours | str |

### Possible use case
<!-- MANUAL: use_case -->
**Book a Meeting End to End**: Find the first time three colleagues are all free and pass it to Create Event.

**Offer Interview Slots**: Suggest several free times across the week to send to a candidate.

**Reschedule After a Clash**: Find the next free time for a meeting that clashed and move it with Update Event.
<!-- END MANUAL -->

---
