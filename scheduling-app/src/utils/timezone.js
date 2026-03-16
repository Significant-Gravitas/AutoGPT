import { OWNER_TZ } from '../config.js'

/**
 * Build an .ics file content string for a confirmed booking.
 * Returns a data: URI that can be used as an <a href> to trigger download.
 */
export function buildIcsDataUri(booking) {
  const { name, email, subject, startISO, endISO, meetLink } = booking

  const dtStart = toIcsDateTime(startISO, OWNER_TZ)
  const dtEnd   = toIcsDateTime(endISO,   OWNER_TZ)
  const uid     = `${Date.now()}-${Math.random().toString(36).slice(2)}@scheduler`
  const now     = toIcsDateTime(new Date().toISOString(), 'UTC').replace('Z', '')

  const ics = [
    'BEGIN:VCALENDAR',
    'VERSION:2.0',
    'PRODID:-//Scheduler//EN',
    'CALSCALE:GREGORIAN',
    'METHOD:REQUEST',
    'BEGIN:VEVENT',
    `UID:${uid}`,
    `DTSTAMP:${now}Z`,
    `DTSTART;TZID=${OWNER_TZ}:${dtStart}`,
    `DTEND;TZID=${OWNER_TZ}:${dtEnd}`,
    `SUMMARY:${escapeIcs(subject || 'Meeting')}`,
    `DESCRIPTION:Google Meet: ${meetLink}`,
    `LOCATION:${meetLink}`,
    `ORGANIZER;CN=${escapeIcs(name)}:MAILTO:${email}`,
    'BEGIN:VALARM',
    'TRIGGER:-PT15M',
    'ACTION:DISPLAY',
    'DESCRIPTION:Reminder',
    'END:VALARM',
    'END:VEVENT',
    'END:VCALENDAR',
  ].join('\r\n')

  const blob    = new Blob([ics], { type: 'text/calendar;charset=utf-8' })
  return URL.createObjectURL(blob)
}

/**
 * Get a short timezone label for display, e.g. "PST", "GMT+2"
 */
export function tzAbbr(tz) {
  try {
    return new Intl.DateTimeFormat('en-US', {
      timeZone:    tz,
      timeZoneName: 'short',
    }).formatToParts(new Date()).find(p => p.type === 'timeZoneName')?.value || tz
  } catch {
    return tz
  }
}

// ── internal helpers ─────────────────────────────────────────────────────────

/** Convert ISO string to iCal TZID format: 'YYYYMMDDTHHmmss' */
function toIcsDateTime(isoString, tz) {
  const d = new Date(isoString)
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: tz,
    year:    'numeric',
    month:   '2-digit',
    day:     '2-digit',
    hour:    '2-digit',
    minute:  '2-digit',
    second:  '2-digit',
    hour12:  false,
  }).formatToParts(d)

  const get = (t) => parts.find(p => p.type === t)?.value || '00'
  return `${get('year')}${get('month')}${get('day')}T${get('hour')}${get('minute')}${get('second')}`
}

function escapeIcs(str) {
  return String(str).replace(/[\\;,]/g, c => `\\${c}`).replace(/\n/g, '\\n')
}
