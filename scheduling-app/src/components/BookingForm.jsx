import { useState } from 'react'
import { ChevronLeft, Loader2, User, Mail, MessageSquare, AlertCircle } from 'lucide-react'
import { formatDateTimeInTz } from '../utils/timeSlots.js'
import { OWNER_TZ } from '../config.js'

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/

export default function BookingForm({ selectedSlot, duration, userTz, onSubmit, onBack }) {
  const [form,       setForm]       = useState({ name: '', email: '', subject: '' })
  const [errors,     setErrors]     = useState({})
  const [submitting, setSubmitting] = useState(false)
  const [apiError,   setApiError]   = useState(null)

  const validate = () => {
    const e = {}
    if (!form.name.trim())               e.name    = 'Name is required'
    if (!form.email.trim())              e.email   = 'Email is required'
    else if (!EMAIL_RE.test(form.email)) e.email   = 'Enter a valid email address'
    if (!form.subject.trim())            e.subject = 'Meeting subject is required'
    setErrors(e)
    return Object.keys(e).length === 0
  }

  const handleChange = (field) => (e) => {
    setForm(prev => ({ ...prev, [field]: e.target.value }))
    if (errors[field]) setErrors(prev => ({ ...prev, [field]: undefined }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!validate()) return
    setSubmitting(true)
    setApiError(null)
    try {
      await onSubmit(form)
    } catch (err) {
      setApiError(err.message || 'Something went wrong. Please try again.')
    } finally {
      setSubmitting(false)
    }
  }

  const slotLabel = selectedSlot
    ? formatDateTimeInTz(selectedSlot.start, userTz)
    : ''

  const ownerLabel = selectedSlot
    ? formatDateTimeInTz(selectedSlot.start, OWNER_TZ)
    : ''

  return (
    <div className="p-6">
      {/* Back + title */}
      <div className="flex items-center gap-2 mb-1">
        <button
          onClick={onBack}
          className="p-1.5 rounded-lg hover:bg-gray-100 transition"
          aria-label="Back"
        >
          <ChevronLeft className="w-5 h-5 text-gray-500" />
        </button>
        <h2 className="text-lg font-semibold text-gray-800">Your Details</h2>
      </div>

      {/* Selected slot summary */}
      <div className="ml-8 mb-5 p-3 bg-brand-50 rounded-xl border border-brand-100">
        <p className="text-sm font-medium text-brand-800">{slotLabel} · {duration} min</p>
        {ownerLabel !== slotLabel && (
          <p className="text-xs text-brand-500 mt-0.5">Israel time: {ownerLabel}</p>
        )}
      </div>

      {/* API error */}
      {apiError && (
        <div className="flex items-start gap-2 mb-4 p-3 bg-red-50 rounded-xl text-sm text-red-600">
          <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
          <span>{apiError}</span>
        </div>
      )}

      <form onSubmit={handleSubmit} noValidate className="space-y-4">
        {/* Name */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Full Name
          </label>
          <div className="relative">
            <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              value={form.name}
              onChange={handleChange('name')}
              placeholder="Jane Smith"
              disabled={submitting}
              className={`w-full pl-9 pr-4 py-2.5 rounded-xl border text-sm transition
                focus:outline-none focus:ring-2 focus:ring-brand-300
                ${errors.name
                  ? 'border-red-300 bg-red-50 focus:ring-red-200'
                  : 'border-gray-200 focus:border-brand-400'
                }
              `}
            />
          </div>
          {errors.name && <p className="text-xs text-red-500 mt-1">{errors.name}</p>}
        </div>

        {/* Email */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Email Address
          </label>
          <div className="relative">
            <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="email"
              value={form.email}
              onChange={handleChange('email')}
              placeholder="jane@example.com"
              disabled={submitting}
              className={`w-full pl-9 pr-4 py-2.5 rounded-xl border text-sm transition
                focus:outline-none focus:ring-2 focus:ring-brand-300
                ${errors.email
                  ? 'border-red-300 bg-red-50 focus:ring-red-200'
                  : 'border-gray-200 focus:border-brand-400'
                }
              `}
            />
          </div>
          {errors.email && <p className="text-xs text-red-500 mt-1">{errors.email}</p>}
        </div>

        {/* Subject */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Meeting Subject
          </label>
          <div className="relative">
            <MessageSquare className="absolute left-3 top-3 w-4 h-4 text-gray-400" />
            <textarea
              value={form.subject}
              onChange={handleChange('subject')}
              placeholder="Brief description of what you'd like to discuss"
              rows={3}
              disabled={submitting}
              className={`w-full pl-9 pr-4 py-2.5 rounded-xl border text-sm transition resize-none
                focus:outline-none focus:ring-2 focus:ring-brand-300
                ${errors.subject
                  ? 'border-red-300 bg-red-50 focus:ring-red-200'
                  : 'border-gray-200 focus:border-brand-400'
                }
              `}
            />
          </div>
          {errors.subject && <p className="text-xs text-red-500 mt-1">{errors.subject}</p>}
        </div>

        {/* Submit */}
        <button
          type="submit"
          disabled={submitting}
          className="w-full py-3 rounded-xl bg-brand-600 text-white font-semibold text-sm
                     hover:bg-brand-700 active:scale-[0.98] transition-all
                     disabled:opacity-70 disabled:cursor-not-allowed
                     flex items-center justify-center gap-2 shadow-sm"
        >
          {submitting ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Booking…
            </>
          ) : (
            'Confirm Booking'
          )}
        </button>
      </form>
    </div>
  )
}
