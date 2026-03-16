/**
 * Booking persistence via Google Apps Script / Google Sheet.
 * No external service required — uses the same GAS deployment as the calendar.
 *
 * The GAS endpoint must have a `saveBooking` action that appends a row
 * to the configured Google Sheet (BOOKINGS_SHEET_ID in Code.gs).
 *
 * If GAS_URL is not configured the call is silently skipped.
 */

const GAS_URL =
  import.meta.env.VITE_GAS_URL ||
  'https://script.google.com/macros/s/AKfycbxIt5jVoSmstOxBh2Ojej3hwSNPHxuWc-gu6CT5-A5iwJEO_8bJYFxg269UJaa0mt09/exec'

/**
 * Persist a confirmed booking record to the Google Sheet via GAS.
 * Silently skips if GAS_URL is not configured.
 */
export async function saveBooking(booking) {
  if (!GAS_URL) return

  try {
    await fetch(GAS_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'text/plain' }, // avoids CORS preflight
      body: JSON.stringify({
        action:   'saveBooking',
        name:     booking.name,
        email:    booking.email,
        subject:  booking.subject,
        startISO: booking.startISO,
        endISO:   booking.endISO,
        duration: booking.duration,
        meetLink: booking.meetLink,
        eventId:  booking.eventId,
        userTz:   booking.userTz,
      }),
    })
  } catch (err) {
    // Non-blocking: booking was already confirmed in Google Calendar
    console.warn('[Sheet] Failed to save booking record:', err.message)
  }
}
