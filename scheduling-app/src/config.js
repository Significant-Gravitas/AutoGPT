/**
 * Scheduling App Configuration
 *
 * Update these values to match your setup:
 *  1. Set GAS_URL after deploying your Google Apps Script web app
 *  2. Adjust WORKING_HOURS for your availability
 *  3. Update OWNER_NAME / OWNER_TZ if needed
 */

// ── GAS Backend URL ──────────────────────────────────────────────────────────
// Override in production via the VITE_GAS_URL environment variable.
// The fallback below is the deployed Google Apps Script Web App.
export const GAS_URL =
  import.meta.env.VITE_GAS_URL ||
  'https://script.google.com/macros/s/AKfycbywOVVMWMPiTmTXzB7F8rm7LZFpItdVHjTWd3eDCbP24lZUwMwDrtENg0GbINmSUSF0/exec'

// ── Owner / Calendar settings ────────────────────────────────────────────────
export const OWNER_NAME = import.meta.env.VITE_OWNER_NAME || 'Your Name'
export const OWNER_TZ   = 'Asia/Jerusalem'   // owner's timezone (Israel)

// ── Working hours (in OWNER_TZ) ──────────────────────────────────────────────
export const WORKING_HOURS = {
  start: 9,   // 09:00
  end:   18,  // 18:00
}

// ── Slot options (minutes) ───────────────────────────────────────────────────
export const SLOT_DURATIONS = [30, 60]

// ── Booking constraints ───────────────────────────────────────────────────────
export const MIN_NOTICE_HOURS = 2   // cannot book within 2 hours of now
export const BUFFER_MINS      = 15  // padding added around each busy block

// ── Days ahead to allow booking ───────────────────────────────────────────────
export const MAX_DAYS_AHEAD = 60
