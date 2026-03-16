import { useState, useEffect, useCallback } from 'react'
import { CalendarCheck } from 'lucide-react'
import CalendarPicker     from './components/CalendarPicker.jsx'
import TimeSlotPicker     from './components/TimeSlotPicker.jsx'
import BookingForm        from './components/BookingForm.jsx'
import ConfirmationScreen from './components/ConfirmationScreen.jsx'
import { GAS_URL, OWNER_NAME } from './config.js'
import { saveBooking }    from './supabaseClient.js'

// Detect if we are embedded as an iframe (Wix or other)
const IS_EMBED = window.self !== window.top ||
  new URLSearchParams(window.location.search).get('embed') === 'true'

/**
 * App steps:
 *  'calendar'  → pick a date
 *  'slots'     → pick a time slot
 *  'form'      → enter name / email / subject
 *  'confirm'   → success screen
 */
export default function App() {
  const [step,         setStep]         = useState('calendar')
  const [selectedDate, setSelectedDate] = useState(null)   // JS Date
  const [busySlots,    setBusySlots]    = useState([])      // [{start,end} ISO strings]
  const [slotsLoading, setSlotsLoading] = useState(false)
  const [slotsError,   setSlotsError]   = useState(null)
  const [selectedSlot, setSelectedSlot] = useState(null)   // {start,end} ISO strings
  const [duration,     setDuration]     = useState(30)      // 30 | 60
  const [booking,      setBooking]      = useState(null)    // confirmed booking details
  const [userTz]                        = useState(
    () => Intl.DateTimeFormat().resolvedOptions().timeZone
  )

  // Fetch busy slots from GAS whenever the selected date changes
  const fetchBusySlots = useCallback(async (date) => {
    if (!date) return
    setSlotsLoading(true)
    setSlotsError(null)
    setBusySlots([])

    const yyyy = date.getFullYear()
    const mm   = String(date.getMonth() + 1).padStart(2, '0')
    const dd   = String(date.getDate()).padStart(2, '0')
    const dateStr = `${yyyy}-${mm}-${dd}`
    const params  = new URLSearchParams({
      action:   'getBusySlots',
      date:     dateStr,
      tz:       userTz,
      duration: String(duration),
    })

    try {
      const res  = await fetch(`${GAS_URL}?${params}`)
      const data = await res.json()
      if (data.error) throw new Error(data.error)
      setBusySlots(data.busySlots || [])
    } catch (err) {
      console.error('[GAS] getBusySlots failed:', err)
      setSlotsError('Could not load availability. Please try again.')
    } finally {
      setSlotsLoading(false)
    }
  }, [userTz, duration])

  useEffect(() => {
    if (selectedDate) fetchBusySlots(selectedDate)
  }, [selectedDate, fetchBusySlots])

  // Called by BookingForm after the user submits
  const handleBooking = useCallback(async (formData) => {
    const { name, email, subject } = formData
    const requestId = `${email}-${selectedSlot.start}-${Date.now()}`

    const body = {
      action:    'createEvent',
      name, email, subject,
      startISO:  selectedSlot.start,
      duration,
      userTz,
      requestId,
    }

    const res  = await fetch(GAS_URL, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    })
    const data = await res.json()
    if (!data.ok) throw new Error(data.error || 'Booking failed')

    const confirmed = {
      name, email, subject, duration, userTz,
      startISO: data.startISO,
      endISO:   data.endISO,
      meetLink: data.meetLink,
      eventId:  data.eventId,
    }

    // Persist to Supabase (non-blocking)
    saveBooking(confirmed)

    // Notify parent Wix page if embedded
    if (IS_EMBED) {
      window.parent.postMessage({ type: 'BOOKING_SUCCESS', booking: confirmed }, '*')
    }

    setBooking(confirmed)
    setStep('confirm')
  }, [selectedSlot, duration, userTz])

  const handleDateSelect = (date) => {
    setSelectedDate(date)
    setSelectedSlot(null)
    setStep('slots')
  }

  const handleSlotSelect = (slot) => {
    setSelectedSlot(slot)
    setStep('form')
  }

  const handleReset = () => {
    setStep('calendar')
    setSelectedDate(null)
    setSelectedSlot(null)
    setBusySlots([])
    setBooking(null)
  }

  return (
    <div className={`min-h-screen bg-gradient-to-br from-brand-50 to-blue-50 ${IS_EMBED ? 'p-2' : 'p-4'}`}>
      <div className={`mx-auto ${IS_EMBED ? 'max-w-full' : 'max-w-xl'}`}>

        {/* Header — hidden in embed mode */}
        {!IS_EMBED && (
          <header className="text-center mb-8 pt-6">
            <div className="inline-flex items-center justify-center w-14 h-14 bg-brand-600 rounded-2xl mb-4 shadow-lg">
              <CalendarCheck className="w-7 h-7 text-white" />
            </div>
            <h1 className="text-2xl font-bold text-gray-900">Book a Meeting</h1>
            <p className="text-gray-500 mt-1 text-sm">with {OWNER_NAME}</p>
          </header>
        )}

        {/* Step card */}
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">

          {step === 'calendar' && (
            <CalendarPicker
              onSelect={handleDateSelect}
              userTz={userTz}
            />
          )}

          {step === 'slots' && (
            <TimeSlotPicker
              selectedDate={selectedDate}
              busySlots={busySlots}
              loading={slotsLoading}
              error={slotsError}
              userTz={userTz}
              duration={duration}
              onDurationChange={(d) => { setDuration(d); setBusySlots([]); fetchBusySlots(selectedDate) }}
              onSelect={handleSlotSelect}
              onBack={() => setStep('calendar')}
            />
          )}

          {step === 'form' && (
            <BookingForm
              selectedSlot={selectedSlot}
              duration={duration}
              userTz={userTz}
              onSubmit={handleBooking}
              onBack={() => setStep('slots')}
            />
          )}

          {step === 'confirm' && (
            <ConfirmationScreen
              booking={booking}
              userTz={userTz}
              onReset={handleReset}
            />
          )}
        </div>

        {/* Footer — hidden in embed mode */}
        {!IS_EMBED && (
          <p className="text-center text-xs text-gray-400 mt-6 pb-4">
            Powered by Google Calendar · Times shown in your local timezone
          </p>
        )}
      </div>
    </div>
  )
}
