import { useState } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { MAX_DAYS_AHEAD } from '../config.js'

const DAYS   = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
const MONTHS = [
  'January','February','March','April','May','June',
  'July','August','September','October','November','December',
]

function startOfDay(date) {
  const d = new Date(date)
  d.setHours(0, 0, 0, 0)
  return d
}

export default function CalendarPicker({ onSelect }) {
  const today = startOfDay(new Date())
  const maxDate = new Date(today.getTime() + MAX_DAYS_AHEAD * 24 * 60 * 60 * 1000)

  const [viewDate, setViewDate] = useState(() => {
    const d = new Date()
    d.setDate(1)
    d.setHours(0, 0, 0, 0)
    return d
  })

  const year  = viewDate.getFullYear()
  const month = viewDate.getMonth()

  // First day of the month, and how many cells to offset
  const firstDay = new Date(year, month, 1).getDay()   // 0 = Sun
  const daysInMonth = new Date(year, month + 1, 0).getDate()

  const prevMonth = () => {
    const d = new Date(viewDate)
    d.setMonth(d.getMonth() - 1)
    // Don't go before current month
    const thisMonth = new Date(today)
    thisMonth.setDate(1)
    if (d >= thisMonth) setViewDate(d)
  }

  const nextMonth = () => {
    const d = new Date(viewDate)
    d.setMonth(d.getMonth() + 1)
    setViewDate(d)
  }

  const canGoPrev = () => {
    const prev = new Date(viewDate)
    prev.setMonth(prev.getMonth() - 1)
    const thisMonth = new Date(today)
    thisMonth.setDate(1)
    return prev >= thisMonth
  }

  const canGoNext = () => {
    const next = new Date(viewDate)
    next.setMonth(next.getMonth() + 1)
    return new Date(next.getFullYear(), next.getMonth(), 1) <= maxDate
  }

  const handleDayClick = (day) => {
    const selected = new Date(year, month, day)
    selected.setHours(0, 0, 0, 0)
    if (selected >= today && selected <= maxDate) {
      onSelect(selected)
    }
  }

  const isDisabled = (day) => {
    const d = new Date(year, month, day)
    return d < today || d > maxDate
  }

  const isToday = (day) => {
    const d = new Date(year, month, day)
    return d.toDateString() === today.toDateString()
  }

  // Build calendar grid cells
  const cells = []
  for (let i = 0; i < firstDay; i++) cells.push(null)
  for (let d = 1; d <= daysInMonth; d++) cells.push(d)

  return (
    <div className="p-6">
      <h2 className="text-lg font-semibold text-gray-800 mb-1">Select a Date</h2>
      <p className="text-sm text-gray-400 mb-5">Choose a day to see available times</p>

      {/* Month navigation */}
      <div className="flex items-center justify-between mb-4">
        <button
          onClick={prevMonth}
          disabled={!canGoPrev()}
          className="p-2 rounded-lg hover:bg-gray-100 disabled:opacity-30 disabled:cursor-not-allowed transition"
          aria-label="Previous month"
        >
          <ChevronLeft className="w-5 h-5 text-gray-600" />
        </button>
        <span className="font-semibold text-gray-800">
          {MONTHS[month]} {year}
        </span>
        <button
          onClick={nextMonth}
          disabled={!canGoNext()}
          className="p-2 rounded-lg hover:bg-gray-100 disabled:opacity-30 disabled:cursor-not-allowed transition"
          aria-label="Next month"
        >
          <ChevronRight className="w-5 h-5 text-gray-600" />
        </button>
      </div>

      {/* Day-of-week headers */}
      <div className="grid grid-cols-7 mb-2">
        {DAYS.map(d => (
          <div key={d} className="text-center text-xs font-medium text-gray-400 py-1">
            {d}
          </div>
        ))}
      </div>

      {/* Day cells */}
      <div className="grid grid-cols-7 gap-1">
        {cells.map((day, idx) => (
          <div key={idx} className="aspect-square flex items-center justify-center">
            {day !== null && (
              <button
                onClick={() => handleDayClick(day)}
                disabled={isDisabled(day)}
                className={`
                  w-9 h-9 rounded-xl text-sm font-medium transition-all
                  ${isDisabled(day)
                    ? 'text-gray-300 cursor-not-allowed'
                    : isToday(day)
                      ? 'bg-brand-600 text-white shadow-md hover:bg-brand-700'
                      : 'text-gray-700 hover:bg-brand-50 hover:text-brand-700'
                  }
                `}
                aria-label={`${MONTHS[month]} ${day}, ${year}`}
              >
                {day}
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
