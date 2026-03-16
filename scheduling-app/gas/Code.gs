/**
 * ============================================================
 *  SCHEDULING APP — Google Apps Script Backend (Code.gs)
 * ============================================================
 *
 * This script acts as the API bridge between the React web app
 * and your Google Calendar. It runs entirely in Google's cloud
 * for free, with no server required.
 *
 * ── SETUP INSTRUCTIONS ──────────────────────────────────────
 *
 *  1. Go to https://script.google.com → click "New project"
 *  2. Delete any existing code and paste this entire file
 *  3. Set OWNER_CALENDAR_ID below (usually your Gmail address)
 *  4. Enable the Calendar Advanced Service:
 *       Extensions → Apps Script → Services (+)
 *       Find "Google Calendar API" → Add → OK
 *  5. Deploy as a Web App:
 *       Deploy → New deployment → Web app
 *       - Description: "Scheduling API v1"
 *       - Execute as: Me
 *       - Who has access: Anyone
 *       → Click Deploy → Authorize (grant Calendar + Gmail access)
 *  6. Copy the Web App URL
 *  7. Paste it as VITE_GAS_URL in your Railway environment variables
 *     AND as GAS_URL for the server-side Express proxy.
 *
 * ── RE-DEPLOYING AFTER CHANGES ──────────────────────────────
 *  Any code changes require a NEW deployment version:
 *  Deploy → Manage deployments → Edit (pencil icon) → New version → Deploy
 *
 * ── TESTING ─────────────────────────────────────────────────
 *  Open in browser (replace YOUR_DATE):
 *  YOUR_WEB_APP_URL?action=getBusySlots&date=2024-01-15&tz=UTC&duration=30
 *  You should get a JSON response with a `busySlots` array.
 *
 * ============================================================
 */

// ─────────────────────────────────────────────────────────────
//  CONFIGURATION — update these values
// ─────────────────────────────────────────────────────────────

/** Your Google Calendar ID. Usually your Gmail address. */
const OWNER_CALENDAR_ID = 'your-email@gmail.com'

/** Timezone for your working hours (IANA format). */
const OWNER_TZ = 'Asia/Jerusalem'

/** Working hours in OWNER_TZ (24-hour, inclusive start, exclusive end). */
const WORKING_HOURS = { start: 9, end: 18 }

/** Buffer added before and after each existing event (minutes). */
const BUFFER_MINS = 15

/** Minimum notice period — cannot book within this many hours from now. */
const MIN_NOTICE_HOURS = 2

// ─────────────────────────────────────────────────────────────
//  HTTP ROUTER
// ─────────────────────────────────────────────────────────────

function doGet(e) {
  return handleRequest(e, null)
}

function doPost(e) {
  let body = null
  try {
    body = JSON.parse(e.postData.contents)
  } catch (_) {
    return jsonResponse({ error: 'Invalid JSON body' }, 400)
  }
  return handleRequest(e, body)
}

function handleRequest(e, body) {
  const action = (e.parameter && e.parameter.action) || (body && body.action)

  try {
    switch (action) {
      case 'getBusySlots':
        return jsonResponse(getBusySlots(e.parameter))
      case 'createEvent':
        return jsonResponse(createEvent(body))
      default:
        return jsonResponse({ error: `Unknown action: ${action}` }, 400)
    }
  } catch (err) {
    Logger.log('Error in handleRequest: ' + err.message + '\n' + err.stack)
    return jsonResponse({ error: err.message }, 500)
  }
}

function jsonResponse(data, statusCode) {
  // GAS ContentService always returns 200; status codes are informational here
  return ContentService
    .createTextOutput(JSON.stringify(data))
    .setMimeType(ContentService.MimeType.JSON)
}

// ─────────────────────────────────────────────────────────────
//  GET BUSY SLOTS
// ─────────────────────────────────────────────────────────────

/**
 * Return the busy blocks for a given calendar date.
 *
 * Params:
 *   date     – 'YYYY-MM-DD'
 *   tz       – visitor's IANA timezone (used only for context; response is UTC ISO)
 *   duration – slot duration in minutes (30 | 60)
 *
 * Response:
 *   { busySlots: [{start: ISO, end: ISO}] }
 *   Each block is already expanded by BUFFER_MINS on both sides.
 */
function getBusySlots(params) {
  const dateStr  = params.date                // 'YYYY-MM-DD'
  const duration = Number(params.duration) || 30

  if (!dateStr || !/^\d{4}-\d{2}-\d{2}$/.test(dateStr)) {
    throw new Error('date param required (YYYY-MM-DD)')
  }

  // Build start/end of the day in OWNER_TZ
  const [year, mon, day] = dateStr.split('-').map(Number)
  const dayStart = new Date(Date.UTC(year, mon - 1, day, 0, 0, 0))
  const dayEnd   = new Date(Date.UTC(year, mon - 1, day, 23, 59, 59))

  // Shift from UTC to OWNER_TZ (approximate — GAS CalendarApp uses the script's tz)
  const calendar = CalendarApp.getCalendarById(OWNER_CALENDAR_ID)
  if (!calendar) throw new Error('Calendar not found. Check OWNER_CALENDAR_ID.')

  const events = calendar.getEvents(dayStart, dayEnd)

  const busySlots = events
    .filter(e => !e.isAllDayEvent())
    .map(e => {
      const bufMs = BUFFER_MINS * 60 * 1000
      return {
        start: new Date(e.getStartTime().getTime() - bufMs).toISOString(),
        end:   new Date(e.getEndTime().getTime()   + bufMs).toISOString(),
      }
    })

  return { busySlots }
}

// ─────────────────────────────────────────────────────────────
//  CREATE EVENT
// ─────────────────────────────────────────────────────────────

/**
 * Create a Google Calendar event with a Google Meet link.
 *
 * Body params:
 *   name       – attendee's full name
 *   email      – attendee's email (receives calendar invite)
 *   subject    – meeting title / description
 *   startISO   – UTC ISO string for meeting start
 *   duration   – meeting length in minutes (30 | 60)
 *   userTz     – attendee's IANA timezone (stored in event description)
 *   requestId  – idempotency key (prevents duplicate events on retry)
 *
 * Response (success):
 *   { ok: true, eventId, meetLink, startISO, endISO }
 *
 * Response (error):
 *   { ok: false, error: '...' }
 */
function createEvent(body) {
  const { name, email, subject, startISO, duration, userTz, requestId } = body

  if (!name || !email || !startISO || !duration) {
    throw new Error('Missing required fields: name, email, startISO, duration')
  }

  const emailRe = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  if (!emailRe.test(email)) throw new Error('Invalid email address')

  const startTime = new Date(startISO)
  const endTime   = new Date(startTime.getTime() + Number(duration) * 60 * 1000)

  // ── Minimum notice check ─────────────────────────────────
  const minNoticeMs = MIN_NOTICE_HOURS * 60 * 60 * 1000
  if (startTime.getTime() - Date.now() < minNoticeMs) {
    return { ok: false, error: `Must book at least ${MIN_NOTICE_HOURS} hours in advance` }
  }

  // ── Idempotency check ────────────────────────────────────
  if (requestId) {
    const existing = findEventByRequestId(requestId, startTime)
    if (existing) {
      const meetLink = getMeetLinkFromEvent(existing)
      return {
        ok:       true,
        eventId:  existing.getId(),
        meetLink: meetLink,
        startISO: existing.getStartTime().toISOString(),
        endISO:   existing.getEndTime().toISOString(),
      }
    }
  }

  // ── Double-booking check ──────────────────────────────────
  const calendar = CalendarApp.getCalendarById(OWNER_CALENDAR_ID)
  if (!calendar) throw new Error('Calendar not found. Check OWNER_CALENDAR_ID.')

  const bufMs     = BUFFER_MINS * 60 * 1000
  const checkFrom = new Date(startTime.getTime() - bufMs)
  const checkTo   = new Date(endTime.getTime()   + bufMs)
  const conflicts = calendar.getEvents(checkFrom, checkTo)

  if (conflicts.length > 0) {
    return { ok: false, error: 'Time slot is no longer available. Please pick another.' }
  }

  // ── Create the event via Advanced Calendar Service ────────
  // conferenceDataVersion:1 triggers automatic Google Meet link generation
  const eventResource = {
    summary:     subject || `Meeting with ${name}`,
    description: [
      `Booked via Scheduling App`,
      `Attendee: ${name} <${email}>`,
      `Timezone: ${userTz || 'UTC'}`,
      requestId ? `RequestId: ${requestId}` : '',
    ].filter(Boolean).join('\n'),
    start:  { dateTime: startTime.toISOString(), timeZone: OWNER_TZ },
    end:    { dateTime: endTime.toISOString(),   timeZone: OWNER_TZ },
    attendees: [
      { email: email, displayName: name },
    ],
    conferenceData: {
      createRequest: {
        requestId:             Utilities.getUuid(),
        conferenceSolutionKey: { type: 'hangoutsMeet' },
      },
    },
    reminders: {
      useDefault: false,
      overrides: [
        { method: 'email', minutes: 60 },
        { method: 'popup', minutes: 15 },
      ],
    },
  }

  const createdEvent = Calendar.Events.insert(
    eventResource,
    OWNER_CALENDAR_ID,
    { conferenceDataVersion: 1, sendUpdates: 'all' }
  )

  const meetLink = createdEvent.conferenceData
    && createdEvent.conferenceData.entryPoints
    && createdEvent.conferenceData.entryPoints.find(ep => ep.entryPointType === 'video')
      ? createdEvent.conferenceData.entryPoints.find(ep => ep.entryPointType === 'video').uri
      : 'https://meet.google.com'

  return {
    ok:       true,
    eventId:  createdEvent.id,
    meetLink: meetLink,
    startISO: createdEvent.start.dateTime,
    endISO:   createdEvent.end.dateTime,
  }
}

// ─────────────────────────────────────────────────────────────
//  HELPERS
// ─────────────────────────────────────────────────────────────

/**
 * Find an existing calendar event whose description contains a requestId.
 * Searches the day of startTime ± 1 day to handle tz edge cases.
 */
function findEventByRequestId(requestId, startTime) {
  const calendar = CalendarApp.getCalendarById(OWNER_CALENDAR_ID)
  const from = new Date(startTime.getTime() - 24 * 60 * 60 * 1000)
  const to   = new Date(startTime.getTime() + 24 * 60 * 60 * 1000)
  const events = calendar.getEvents(from, to)
  return events.find(e => (e.getDescription() || '').includes(`RequestId: ${requestId}`)) || null
}

/**
 * Extract a Google Meet video link from a CalendarApp Event object.
 */
function getMeetLinkFromEvent(event) {
  // CalendarApp Event objects don't expose conferenceData directly,
  // so we fall back to checking the location field (GAS sets it).
  const loc = event.getLocation() || ''
  if (loc.startsWith('https://meet.google.com')) return loc

  // Try parsing from description
  const desc = event.getDescription() || ''
  const match = desc.match(/https:\/\/meet\.google\.com\/[a-z-]+/)
  return match ? match[0] : 'https://meet.google.com'
}
