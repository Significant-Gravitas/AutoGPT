/**
 * Express server for the Scheduling App.
 *
 * Responsibilities:
 *  1. Serve the React production build (dist/)
 *  2. Expose a REST API so external apps (Telegram, Wix, other services)
 *     can query availability and create bookings without the browser UI.
 *
 * Environment variables:
 *  PORT           – HTTP port (Railway sets this automatically)
 *  GAS_URL        – Google Apps Script Web App URL (server-side calls)
 *  API_KEY        – Secret key for /api/* access (X-Api-Key header)
 *  ALLOWED_ORIGIN – Comma-separated origins for CORS (optional, defaults to *)
 */

import express        from 'express'
import { createServer }  from 'http'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
import fetch             from 'node-fetch'

const __dirname = dirname(fileURLToPath(import.meta.url))
const app       = express()
const PORT      = process.env.PORT || 3000
const GAS_URL   =
  process.env.GAS_URL ||
  process.env.VITE_GAS_URL ||
  'https://script.google.com/macros/s/AKfycbxIt5jVoSmstOxBh2Ojej3hwSNPHxuWc-gu6CT5-A5iwJEO_8bJYFxg269UJaa0mt09/exec'
const API_KEY   = process.env.API_KEY || ''

// ── Middleware ───────────────────────────────────────────────────────────────

app.use(express.json())

// CORS — allow all origins by default (required for Wix iframe + external apps)
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin',  '*')
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, X-Api-Key')
  if (req.method === 'OPTIONS') return res.sendStatus(204)
  next()
})

// Remove X-Frame-Options so the app can be embedded as an iframe in Wix
app.use((req, res, next) => {
  res.removeHeader('X-Frame-Options')
  res.setHeader(
    'Content-Security-Policy',
    "frame-ancestors 'self' *.wix.com *.wixsite.com *.editorx.com"
  )
  next()
})

// ── API key auth middleware (applied to /api/* routes) ──────────────────────

function requireApiKey(req, res, next) {
  if (!API_KEY) return next()  // auth disabled if no key set

  const key = req.headers['x-api-key'] || req.query.apiKey
  if (key !== API_KEY) {
    return res.status(401).json({ ok: false, error: 'Invalid or missing API key' })
  }
  next()
}

// ── REST API routes ──────────────────────────────────────────────────────────

/**
 * GET /api/health
 * Quick liveness check.
 */
app.get('/api/health', (req, res) => {
  res.json({ ok: true, status: 'running', ts: new Date().toISOString() })
})

/**
 * GET /api/slots?date=YYYY-MM-DD&tz=America/New_York&duration=30
 *
 * Returns available time slots by proxying to GAS.
 * Protected by X-Api-Key header.
 *
 * Response: { slots: [{start, end, label}] }
 */
app.get('/api/slots', requireApiKey, async (req, res) => {
  const { date, tz = 'UTC', duration = '30' } = req.query

  if (!date || !/^\d{4}-\d{2}-\d{2}$/.test(date)) {
    return res.status(400).json({ ok: false, error: 'date param required (YYYY-MM-DD)' })
  }
  if (!GAS_URL) {
    return res.status(503).json({ ok: false, error: 'GAS_URL not configured' })
  }

  try {
    const params = new URLSearchParams({ action: 'getBusySlots', date, tz, duration })
    const gasRes = await fetch(`${GAS_URL}?${params}`)
    const data   = await gasRes.json()
    if (data.error) return res.status(502).json({ ok: false, error: data.error })
    res.json({ ok: true, slots: data.busySlots || [] })
  } catch (err) {
    res.status(502).json({ ok: false, error: err.message })
  }
})

/**
 * POST /api/book
 *
 * Create a booking via GAS.
 * Protected by X-Api-Key header.
 *
 * Body: { name, email, subject, startISO, duration, userTz, requestId? }
 * Response: { ok: true, meetLink, eventId, startISO, endISO }
 */
app.post('/api/book', requireApiKey, async (req, res) => {
  const { name, email, subject, startISO, duration, userTz } = req.body

  if (!name || !email || !startISO || !duration) {
    return res.status(400).json({ ok: false, error: 'Missing required fields: name, email, startISO, duration' })
  }
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    return res.status(400).json({ ok: false, error: 'Invalid email address' })
  }
  if (!GAS_URL) {
    return res.status(503).json({ ok: false, error: 'GAS_URL not configured' })
  }

  const requestId = req.body.requestId || `${email}-${startISO}-${Date.now()}`

  try {
    const gasRes = await fetch(GAS_URL, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ action: 'createEvent', name, email, subject, startISO, duration, userTz, requestId }),
    })
    const data = await gasRes.json()
    if (!data.ok) return res.status(502).json({ ok: false, error: data.error || 'Booking failed' })
    res.json(data)
  } catch (err) {
    res.status(502).json({ ok: false, error: err.message })
  }
})

// ── Static file serving (React build) ───────────────────────────────────────

const distDir = join(__dirname, 'dist')
app.use(express.static(distDir))

// SPA fallback — serve index.html for all non-API routes
app.get('*', (req, res) => {
  res.sendFile(join(distDir, 'index.html'))
})

// ── Start ────────────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`Scheduling app running on port ${PORT}`)
  if (!GAS_URL) console.warn('⚠️  GAS_URL is not set — calendar integration will not work')
  if (!API_KEY) console.warn('⚠️  API_KEY is not set — /api/* endpoints are unprotected')
})
