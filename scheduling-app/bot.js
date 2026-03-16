/**
 * Telegram Bot for the Scheduling App
 *
 * Allows users to book meetings via Telegram using the same backend
 * as the web app (Express /api/* endpoints).
 *
 * Setup:
 *  1. Create a bot via @BotFather and get your token
 *  2. Set TELEGRAM_BOT_TOKEN env var
 *  3. Set SERVER_URL to your Railway app URL (e.g. https://my-app.railway.app)
 *  4. Set API_KEY to match your Express server's API_KEY env var
 *
 * Conversation flow:
 *  /start or /book  → show next 7 available dates
 *  Select date      → show duration options (30 / 60 min)
 *  Select duration  → show available time slots
 *  Select slot      → ask for Name | Email | Subject
 *  Submit details   → create booking → show confirmation with Meet link
 */

import { Telegraf, Markup } from 'telegraf'
import fetch from 'node-fetch'

const BOT_TOKEN  = process.env.TELEGRAM_BOT_TOKEN
const SERVER_URL = process.env.SERVER_URL || 'http://localhost:3000'
const API_KEY    = process.env.API_KEY    || ''

if (!BOT_TOKEN) {
  console.error('TELEGRAM_BOT_TOKEN is not set. Telegram bot will not start.')
  process.exit(1)
}

const bot = new Telegraf(BOT_TOKEN)

// ── In-memory session store ──────────────────────────────────────────────────
// (Replace with Redis for multi-instance Railway deployments)
const sessions = new Map()

function getSession(chatId) {
  if (!sessions.has(chatId)) sessions.set(chatId, {})
  return sessions.get(chatId)
}

function clearSession(chatId) {
  sessions.delete(chatId)
}

// ── Helpers ──────────────────────────────────────────────────────────────────

const HEADERS = { 'Content-Type': 'application/json', 'X-Api-Key': API_KEY }

async function fetchSlots(date, tz, duration) {
  const params = new URLSearchParams({ date, tz, duration })
  const res    = await fetch(`${SERVER_URL}/api/slots?${params}`, { headers: HEADERS })
  return res.json()
}

async function createBooking(payload) {
  const res = await fetch(`${SERVER_URL}/api/book`, {
    method:  'POST',
    headers: HEADERS,
    body:    JSON.stringify(payload),
  })
  return res.json()
}

/** Return next N working days as 'YYYY-MM-DD' strings */
function nextWorkingDays(n = 7) {
  const days = []
  const d    = new Date()
  d.setDate(d.getDate() + 1)   // start from tomorrow

  while (days.length < n) {
    const dow = d.getDay()
    if (dow !== 0 && dow !== 6) {  // skip weekends (0=Sun, 6=Sat)
      days.push(d.toISOString().slice(0, 10))
    }
    d.setDate(d.getDate() + 1)
  }
  return days
}

/** Format 'YYYY-MM-DD' to a human-readable label like 'Mon, Jan 15' */
function formatDate(dateStr) {
  return new Date(dateStr + 'T12:00:00Z').toLocaleDateString('en-US', {
    weekday: 'short',
    month:   'short',
    day:     'numeric',
    timeZone: 'UTC',
  })
}

// ── Bot handlers ─────────────────────────────────────────────────────────────

// /start or /book — show date picker
const showDatePicker = async (ctx) => {
  clearSession(ctx.chat.id)
  const dates   = nextWorkingDays(7)
  const buttons = dates.map(d =>
    Markup.button.callback(formatDate(d), `date:${d}`)
  )

  // Group into rows of 2
  const rows = []
  for (let i = 0; i < buttons.length; i += 2) rows.push(buttons.slice(i, i + 2))

  await ctx.reply(
    '📅 *Select a date for your meeting:*',
    { parse_mode: 'Markdown', ...Markup.inlineKeyboard(rows) }
  )
}

bot.start(showDatePicker)
bot.command('book', showDatePicker)

// Date selected → ask for duration
bot.action(/^date:(.+)$/, async (ctx) => {
  const date = ctx.match[1]
  const sess = getSession(ctx.chat.id)
  sess.date  = date

  await ctx.editMessageText(
    `📅 *${formatDate(date)}*\n\n⏱ How long should the meeting be?`,
    {
      parse_mode: 'Markdown',
      ...Markup.inlineKeyboard([
        [Markup.button.callback('30 minutes', `dur:30`), Markup.button.callback('60 minutes', `dur:60`)],
        [Markup.button.callback('← Back', 'back:dates')],
      ]),
    }
  )
})

// Back to dates
bot.action('back:dates', (ctx) => {
  clearSession(ctx.chat.id)
  return showDatePicker(ctx)
})

// Duration selected → load + show slots
bot.action(/^dur:(\d+)$/, async (ctx) => {
  const duration = Number(ctx.match[1])
  const sess     = getSession(ctx.chat.id)
  sess.duration  = duration

  // Use UTC as the default tz (user can refine via web app)
  const userTz = 'UTC'
  sess.userTz  = userTz

  await ctx.editMessageText(`⏳ Loading available times for *${formatDate(sess.date)}*…`, { parse_mode: 'Markdown' })

  try {
    const data = await fetchSlots(sess.date, userTz, duration)
    if (!data.ok || !data.slots?.length) {
      return ctx.editMessageText(
        `😔 No available slots on *${formatDate(sess.date)}* for ${duration} min.\n\nUse /book to try another date.`,
        { parse_mode: 'Markdown' }
      )
    }

    sess.slots = data.slots
    const buttons = data.slots.map((s, i) =>
      Markup.button.callback(s.label, `slot:${i}`)
    )

    const rows = []
    for (let i = 0; i < buttons.length; i += 3) rows.push(buttons.slice(i, i + 3))
    rows.push([Markup.button.callback('← Back', `date:${sess.date}`)])

    await ctx.editMessageText(
      `📅 *${formatDate(sess.date)}* · ${duration} min\n\nSelect a time slot:`,
      { parse_mode: 'Markdown', ...Markup.inlineKeyboard(rows) }
    )
  } catch (err) {
    await ctx.editMessageText(`❌ Could not load slots: ${err.message}`)
  }
})

// Slot selected → ask for contact details
bot.action(/^slot:(\d+)$/, async (ctx) => {
  const idx  = Number(ctx.match[1])
  const sess = getSession(ctx.chat.id)

  if (!sess.slots || !sess.slots[idx]) {
    return ctx.reply('Something went wrong. Please use /book to start over.')
  }

  sess.selectedSlot = sess.slots[idx]

  await ctx.editMessageText(
    `✅ *${sess.selectedSlot.label}* on *${formatDate(sess.date)}* — ${sess.duration} min\n\n` +
    `Please send your details in this format:\n\n` +
    `\`Name | email@example.com | Meeting Subject\``,
    { parse_mode: 'Markdown' }
  )

  sess.awaitingDetails = true
})

// Handle text input for contact details
bot.on('text', async (ctx) => {
  const sess = getSession(ctx.chat.id)
  if (!sess.awaitingDetails) return

  const parts = ctx.message.text.split('|').map(s => s.trim())
  if (parts.length < 3) {
    return ctx.reply(
      '⚠️ Please use the format:\n`Name | email@example.com | Meeting Subject`',
      { parse_mode: 'Markdown' }
    )
  }

  const [name, email, subject] = parts

  const emailRe = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  if (!emailRe.test(email)) {
    return ctx.reply('⚠️ That doesn\'t look like a valid email. Please try again.')
  }

  sess.awaitingDetails = false

  const processingMsg = await ctx.reply('⏳ Creating your booking…')

  try {
    const requestId = `tg-${ctx.chat.id}-${sess.selectedSlot.start}-${Date.now()}`
    const data = await createBooking({
      name, email, subject,
      startISO:  sess.selectedSlot.start,
      duration:  sess.duration,
      userTz:    sess.userTz,
      requestId,
    })

    if (!data.ok) throw new Error(data.error || 'Booking failed')

    // Success
    clearSession(ctx.chat.id)

    await ctx.telegram.editMessageText(
      ctx.chat.id, processingMsg.message_id, null,
      `✅ *Booking Confirmed!*\n\n` +
      `📅 *${formatDate(sess.date)}* at *${sess.selectedSlot.label}* (${sess.duration} min)\n` +
      `👤 ${name}\n` +
      `📧 ${email}\n` +
      `📝 ${subject}\n\n` +
      `🎥 *Google Meet:* ${data.meetLink}\n\n` +
      `_A calendar invite has been sent to your email._`,
      { parse_mode: 'Markdown' }
    )

    // Inline button to join Meet
    await ctx.reply('Tap below to join your meeting:', Markup.inlineKeyboard([
      [Markup.button.url('Join Google Meet', data.meetLink)],
      [Markup.button.callback('Book another meeting', 'back:dates')],
    ]))

  } catch (err) {
    await ctx.telegram.editMessageText(
      ctx.chat.id, processingMsg.message_id, null,
      `❌ Booking failed: ${err.message}\n\nUse /book to try again.`
    )
  }
})

// ── Error handling ────────────────────────────────────────────────────────────

bot.catch((err, ctx) => {
  console.error('[Telegram Bot Error]', err)
  ctx.reply('An unexpected error occurred. Please use /book to start over.').catch(() => {})
})

// ── Launch ────────────────────────────────────────────────────────────────────

bot.launch({ dropPendingUpdates: true })
console.log('Telegram scheduling bot started')

// Graceful stop
process.once('SIGINT',  () => bot.stop('SIGINT'))
process.once('SIGTERM', () => bot.stop('SIGTERM'))
