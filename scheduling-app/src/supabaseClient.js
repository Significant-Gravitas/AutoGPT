import { createClient } from '@supabase/supabase-js'

const supabaseUrl  = import.meta.env.VITE_SUPABASE_URL  || ''
const supabaseKey  = import.meta.env.VITE_SUPABASE_ANON_KEY || ''

// Client is null-safe: if env vars are missing the app still works,
// but booking records won't be persisted to Supabase.
export const supabase =
  supabaseUrl && supabaseKey
    ? createClient(supabaseUrl, supabaseKey)
    : null

/**
 * Persist a confirmed booking record to Supabase.
 * Silently skips if Supabase is not configured.
 */
export async function saveBooking(booking) {
  if (!supabase) return

  const { error } = await supabase.from('bookings').insert({
    name:         booking.name,
    email:        booking.email,
    subject:      booking.subject,
    start_time:   booking.startISO,
    end_time:     booking.endISO,
    duration_mins: booking.duration,
    meet_link:    booking.meetLink,
    event_id:     booking.eventId,
    user_tz:      booking.userTz,
  })

  if (error) {
    console.warn('[Supabase] Failed to save booking:', error.message)
  }
}
