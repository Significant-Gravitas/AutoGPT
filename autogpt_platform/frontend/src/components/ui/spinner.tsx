"use client";
import {
  type CSSProperties,
  type RefObject,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
} from 'react'

import { cn } from '@/lib/utils'

type SpinnerVariant = 'default' | 'gradient' | 'bars'

const ART = { default: 20, gradient: 21, bars: 18.4 } as const

const round = (n: number) => String(Math.round(n * 1000) / 1000)

function windowFor(art: number, cy = 12) {
  const size = 24 * (art / ART.default)
  return `${round(12 - size / 2)} ${round(cy - size / 2)} ${round(size)} ${round(size)}`
}

const ARC = { min: 18, max: 72 } as const

const CYCLE_EASES = [
  'cubic-bezier(0.37, 0, 0.63, 1)',
  'cubic-bezier(0.45, 0, 0.55, 1)',
  'cubic-bezier(0.65, 0, 0.35, 1)',
] as const

const between = (low: number, high: number) => low + Math.random() * (high - low)

function useDynamicArc(
  arcRef: RefObject<SVGCircleElement | null>,
  rest: number,
  on: boolean,
  paused = false,
) {
  const cycle = useRef<Animation | null>(null)

  useEffect(() => {
    const circle = arcRef.current
    if (!circle || !on || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    let angle = 0
    let length = rest
    const next = () => {
      const to = between(ARC.min, ARC.max)
      const ahead = angle + between(45, 180)
      const animation = circle.animate(
        {
          rotate: [`${String(angle - length * 1.8)}deg`, `${String(ahead - to * 1.8)}deg`],
          strokeDasharray: [`${String(length)} 100`, `${String(to)} 100`],
        },
        {
          duration: between(600, 1200),
          easing: CYCLE_EASES[Math.floor(Math.random() * CYCLE_EASES.length)],
          fill: 'forwards',
        },
      )
      animation.onfinish = next
      cycle.current = animation
      angle = ahead % 360
      length = to
    }
    next()

    return () => {
      cycle.current = null
      for (const animation of circle.getAnimations()) {
        animation.onfinish = null
        animation.cancel()
      }
    }
  }, [arcRef, rest, on])

  useEffect(() => {
    const animation = cycle.current
    if (paused) animation?.pause()
    else if (animation?.playState === 'paused') animation.play()
  }, [paused])
}

const SPINNER_REST = 75

function Spinner({
  className,
  variant = 'default',
  dynamic = true,
  ...props
}: React.ComponentProps<'svg'> & { variant?: SpinnerVariant; dynamic?: boolean }) {

  const id = useId()
  const arcRef = useRef<SVGCircleElement>(null)
  useDynamicArc(arcRef, SPINNER_REST, variant === 'default' && dynamic)

  const shared = {
    'data-slot': 'spinner',
    'data-variant': variant,
    role: 'status',
    'aria-label': 'Loading',
  } as const

  if (variant === 'gradient') {
    return (
      <svg
        viewBox={windowFor(ART.gradient, 12.06)}
        fill="none"
        {...shared}
        className={cn('size-4 animate-spin', className)}
        {...props}
      >
        <defs>
          <linearGradient id={`spinner-head-${id}`} x1="50%" x2="50%" y1="5.271%" y2="91.793%">
            <stop offset="0%" stopColor="currentColor" />
            <stop offset="100%" stopColor="currentColor" stopOpacity={0.55} />
          </linearGradient>
          <linearGradient id={`spinner-tail-${id}`} x1="50%" x2="50%" y1="15.24%" y2="87.15%">
            <stop offset="0%" stopColor="currentColor" stopOpacity={0} />
            <stop offset="100%" stopColor="currentColor" stopOpacity={0.55} />
          </linearGradient>
        </defs>
        <path
          d="M8.749.021a1.5 1.5 0 0 1 .497 2.958A7.5 7.5 0 0 0 3 10.375a7.5 7.5 0 0 0 7.5 7.5v3c-5.799 0-10.5-4.7-10.5-10.5C0 5.23 3.726.865 8.749.021"
          fill={`url(#spinner-head-${id})`}
          transform="translate(1.5 1.625)"
        />
        <path
          d="M15.392 2.673a1.5 1.5 0 0 1 2.119-.115A10.48 10.48 0 0 1 21 10.375c0 5.8-4.701 10.5-10.5 10.5v-3a7.5 7.5 0 0 0 5.007-13.084a1.5 1.5 0 0 1-.115-2.118"
          fill={`url(#spinner-tail-${id})`}
          transform="translate(1.5 1.625)"
        />
      </svg>
    )
  }

  if (variant === 'bars') {
    return (
      <svg viewBox={windowFor(ART.bars)} {...shared} className={cn('size-4', className)} {...props}>

        {Array.from({ length: 12 }, (_, i) => (
          <rect
            key={i}
            className="t-spinner-bar"
            x="16"
            y="11.1"
            width="5.2"
            height="1.8"
            rx="0.9"
            fill="currentColor"
            transform={`rotate(${i * 30} 12 12)`}
            style={{ animationDelay: `calc(var(--spinner-bars-dur) * ${(i - 11) / 12})` }}
          />
        ))}
      </svg>
    )
  }

  return (
    <svg
      viewBox={windowFor(ART.default)}
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      {...shared}
      data-dynamic={dynamic || undefined}
      className={cn('size-4 animate-spin', className)}
      {...props}
    >
      <circle
        ref={arcRef}
        cx="12"
        cy="12"
        r="9"
        pathLength={100}
        strokeDasharray={`${String(SPINNER_REST)} 100`}
        className="origin-center -rotate-135 transform-fill"
      />
    </svg>
  )
}

type CheckState = 'loading' | 'done'

function readNum(name: string, fallback: number): number {
  const raw = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  if (!raw) return fallback
  if (raw.endsWith('ms')) return Number.parseFloat(raw)
  if (raw.endsWith('s') && !raw.endsWith('ms')) return Number.parseFloat(raw) * 1000
  const n = Number.parseFloat(raw)
  return Number.isNaN(n) ? fallback : n
}

const BADGE_REST = 25

function StatusBadge({
  state = 'loading',
  label,
  chime = true,
  variant = 'solid',
  spinner = 'default',
  dynamic = true,
}: {
  state?: CheckState
  label?: string
  chime?: boolean
  variant?: 'solid' | 'outline'
  spinner?: SpinnerVariant
  dynamic?: boolean
}) {
  const markRef = useRef<SVGPathElement>(null)
  const arcRef = useRef<SVGCircleElement>(null)

  useDynamicArc(arcRef, BADGE_REST, spinner === 'default' && dynamic, state === 'done')
  const [len, setLen] = useState<number | null>(null)
  const [crossing, setCrossing] = useState(false)
  const mounted = useRef(false)

  useLayoutEffect(() => {
    const mark = markRef.current
    if (mark) setLen(Math.ceil(mark.getTotalLength()))
  }, [])

  useEffect(() => {
    if (!mounted.current) {
      mounted.current = true
      return
    }
    setCrossing(true)
    const t = window.setTimeout(() => setCrossing(false), readNum('--check-fill-dur', 220) * 0.45)
    return () => window.clearTimeout(t)

  }, [state])

  const style = len === null ? undefined : ({ '--check-mark-len': len } as CSSProperties)

  return (

    <span className={'t-check-blur-wrap' + (crossing ? ' is-crossing' : '')}>
      <span
        className={cn('t-check-badge', variant === 'outline' && 'is-outline')}
        data-state={state}

        data-success={(chime && state === 'done') || undefined}
        style={style}
        role="img"
        aria-label={label ?? (state === 'done' ? 'Done' : 'In progress')}
      >
        {spinner === 'default' ? (
          <>
            <span className="t-check-ring" aria-hidden="true" />
            <svg className="t-check-arc" viewBox="0 0 22 22" aria-hidden="true">
              <circle
                ref={arcRef}
                cx="11"
                cy="11"
                r="9.75"
                pathLength={100}
                strokeDasharray={`${String(BADGE_REST)} 100`}
                className="origin-center -rotate-45 transform-fill"
              />
            </svg>
          </>
        ) : (
          <span className="t-check-spinner" aria-hidden="true">
            <Spinner variant={spinner} className="size-full" aria-hidden />
          </span>
        )}
        <span className="t-check-fill" aria-hidden="true" />
        <span className="t-check-disc" aria-hidden="true">
          <svg viewBox="0 0 24 24">
            <path ref={markRef} className="t-check-mark" d="M8 12.5L10.8 15.5L16.4 9.5" />
          </svg>
        </span>
      </span>
    </span>
  )
}

export { Spinner, StatusBadge, type CheckState, type SpinnerVariant }
