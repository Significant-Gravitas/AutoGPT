import { WORKING_HOURS, BUFFER_MINS, MIN_NOTICE_HOURS, OWNER_TZ } from '../config.js'

/**
 * Generate candidate time slots for a given date in the owner's timezone,
 * then filter out:
 *   - slots that overlap busy blocks (+ buffer padding)
 *   - slots within MIN_NOTICE_HOURS of right now
 *   - slots that end after working hours end
 *
 * @param {Date}   date       - The selected date (JS Date, any timezone)
 * @param {Array}  busySlots  - [{start: ISOstring, end: ISOstring}] from GAS
 * @param {string} userTz     - IANA timezone of the visitor
 * @param {number} duration   - Slot duration in minutes (30 | 60)
 * @returns {Array} [{start: ISOstring, end: ISOstring, label: string}]
 */
export function generateAvailableSlots(date, busySlots, userTz, duration = 30) {
  const slots = []
  const now   = new Date()
  const cutoff = new Date(now.getTime() + MIN_NOTICE_HOURS * 60 * 60 * 1000)

  // Build the working-hours window in the owner's timezone for the selected date
  const dateStr = toDateString(date)   // 'YYYY-MM-DD' in local calendar
  const workStart = parseInTz(`${dateStr}T${pad(WORKING_HOURS.start)}:00:00`, OWNER_TZ)
  const workEnd   = parseInTz(`${dateStr}T${pad(WORKING_HOURS.end  )}:00:00`, OWNER_TZ)

  // Expand busy slots with buffer
  const busy = busySlots.map(b => ({
    start: new Date(new Date(b.start).getTime() - BUFFER_MINS * 60 * 1000),
    end:   new Date(new Date(b.end  ).getTime() + BUFFER_MINS * 60 * 1000),
  }))

  let cursor = new Date(workStart)
  while (cursor < workEnd) {
    const slotEnd = new Date(cursor.getTime() + duration * 60 * 1000)

    // Must end within working hours
    if (slotEnd > workEnd) break

    // Must be far enough in the future
    if (cursor >= cutoff) {
      // Must not overlap any busy block
      const overlaps = busy.some(b => cursor < b.end && slotEnd > b.start)
      if (!overlaps) {
        slots.push({
          start: cursor.toISOString(),
          end:   slotEnd.toISOString(),
          label: formatInTz(cursor, userTz),
        })
      }
    }

    cursor = new Date(cursor.getTime() + 30 * 60 * 1000)  // iterate in 30-min steps
  }

  return slots
}

// ── helpers ──────────────────────────────────────────────────────────────────

function pad(n) { return String(n).padStart(2, '0') }

/** Format a JS Date in a given IANA timezone as "HH:MM" (12-hour with AM/PM) */
export function formatInTz(date, tz) {
  return new Intl.DateTimeFormat('en-US', {
    timeZone:    tz,
    hour:        'numeric',
    minute:      '2-digit',
    hour12:      true,
  }).format(date)
}

/** Format date + time in a given tz, e.g. "Mon, Jan 15 at 10:00 AM" */
export function formatDateTimeInTz(isoString, tz) {
  const date = new Date(isoString)
  return new Intl.DateTimeFormat('en-US', {
    timeZone:  tz,
    weekday:   'short',
    month:     'short',
    day:       'numeric',
    hour:      'numeric',
    minute:    '2-digit',
    hour12:    true,
  }).format(date)
}

/** Return 'YYYY-MM-DD' string from a JS Date (uses local calendar date). */
function toDateString(date) {
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`
}

/**
 * Parse a local datetime string ('YYYY-MM-DDTHH:MM:SS') as if it's in `tz`.
 * Works by finding the UTC instant that, when rendered in `tz`, gives that datetime.
 * Uses Intl.DateTimeFormat bisection — works in all modern browsers.
 */
function parseInTz(localStr, tz) {
  // First approximation: treat as UTC
  const approxUtc = new Date(localStr + 'Z')

  // Get what that UTC instant looks like in `tz`
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: tz,
    year:   'numeric',
    month:  '2-digit',
    day:    '2-digit',
    hour:   '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  }).formatToParts(approxUtc)

  const get = (type) => parts.find(p => p.type === type)?.value || '00'
  const tzStr = `${get('year')}-${get('month')}-${get('day')}T${get('hour')}:${get('minute')}:${get('second')}`

  // Compute the offset
  const tzDate   = new Date(tzStr + 'Z')
  const diff     = approxUtc - tzDate              // milliseconds offset
  return new Date(approxUtc.getTime() - diff)
}
