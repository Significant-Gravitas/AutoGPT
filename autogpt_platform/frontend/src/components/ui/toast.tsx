"use client";
import { useDirection } from '@base-ui/react/direction-provider'
import {
  type ComponentProps,
  type CSSProperties,
  type ReactNode,
  type RefObject,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from 'react'
import {
  animate,
  AnimatePresence,
  cancelFrame,
  frame,
  motion,
  MotionConfig,
  useIsPresent,
  useMotionValue,
  usePresenceData,
  useReducedMotion,
} from 'motion/react'

import { ALERT_MARKS, type AlertTone } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { StatusBadge } from '@/components/ui/spinner'
import { joinsLetters } from '@/lib/joined-script'
import { dissolve } from '@/lib/smoky-dissolve'
import { cn } from '@/lib/utils'

const EVENT = 'kobra:toast'
const DISMISS = 'kobra:toast-dismiss'
const LIFETIME = 2400

const DETAIL_LIFETIME = 4000

const ACTION_LIFETIME = 6000
const GAP = 8

const PEEK = 10
const SHRINK = 0.03
const DEPTH = 3
const FADE = 0.15
const REACH = 10
const SWIPE = 44
const FLICK = 380
const MORPH = { type: 'spring', duration: 0.3, bounce: 0 } as const
const EXIT = { type: 'spring', duration: 0.2, bounce: 0 } as const

export const LINE_HEIGHT = 36

const CARD_WIDTH = 340

export type ToastSide = 'top' | 'bottom'

export type ToastAlign = 'start' | 'center' | 'end' | 'left' | 'right'
export type ToastPosition = `${ToastSide}-${ToastAlign}`
export const toastPositions = [
  'top-start',
  'top-center',
  'top-end',
  'bottom-start',
  'bottom-center',
  'bottom-end',
] as const satisfies readonly ToastPosition[]

type ToastEdge = 'start' | 'center' | 'end'

const JUSTIFY: Record<ToastEdge, string> = {
  start: 'justify-start',
  center: 'justify-center',
  end: 'justify-end',
}

export type ToastState = 'pending' | AlertTone
export type ToastAction = { label: string; run: () => void }
export type ToastInput = {
  id?: string
  message: string

  description?: string
  state?: ToastState
  action?: ToastAction
  lifetime?: number
}
export type Note = ToastInput & { id: string }
export type ToastClock = { waits: Map<string, number>; since: number | null }
export type StackSlot = { y: number; scale: number; opacity: number }
export type StackCard = {
  id: string
  render: (behind: boolean) => ReactNode
}

let nextId = 0

export function toast(input: string | ToastInput) {
  const detail = typeof input === 'string' ? { message: input } : input
  window.dispatchEvent(
    new CustomEvent(EVENT, {
      detail: { ...detail, id: detail.id ?? `toast-${++nextId}` },
    }),
  )
}

export function dismissToast(id: string) {
  window.dispatchEvent(new CustomEvent(DISMISS, { detail: id }))
}

export function upsertToast(notes: readonly Note[], note: Note): Note[] {
  const index = notes.findIndex((item) => item.id === note.id)
  return index === -1 ? [note, ...notes] : notes.map((item, at) => (at === index ? note : item))
}

export function dismissDelay(
  state: ToastState | undefined,
  hasAction: boolean,
  lifetime?: number,
  hasDetail = false,
): number | null {
  if (state === 'pending') return null
  if (lifetime !== undefined) return lifetime
  return hasAction ? ACTION_LIFETIME : hasDetail ? DETAIL_LIFETIME : LIFETIME
}

export function tick(clock: ToastClock, at: number) {
  if (clock.since === null) return

  const spent = at - clock.since
  clock.since = at
  for (const [id, left] of clock.waits) clock.waits.set(id, left - spent)
}

export function pileSlot(index: number): StackSlot {
  const depth = Math.min(index, DEPTH - 1)
  return {
    y: depth * PEEK,
    scale: 1 - depth * SHRINK,
    opacity: index < DEPTH ? 1 - depth * FADE : 0,
  }
}

export function fanSlot(index: number, heights: readonly number[]): StackSlot {
  let y = 0
  for (let at = 0; at < index; at++) y += (heights[at] ?? LINE_HEIGHT) + GAP
  return { y, scale: 1, opacity: 1 }
}

const TONE_INK: Record<Exclude<AlertTone, 'success'>, string> = {
  error: 'text-error',
  warning: 'text-warning',
  info: 'text-info',
}

const BADGE_SIZE = { '--check-size': '16px' } as CSSProperties

function ToastGlyph({ state }: { state: ToastState }) {
  switch (state) {

    case 'pending':
    case 'success':
      return (

        <span className="flex" style={BADGE_SIZE}>
          <StatusBadge state={state === 'success' ? 'done' : 'loading'} />
        </span>
      )
    case 'error':
    case 'warning':
    case 'info': {
      const Mark = ALERT_MARKS[state]

      return (
        <Mark
          className={cn('shrink-0', state === 'error' ? 'size-5' : 'size-4', TONE_INK[state])}
          aria-hidden
        />
      )
    }
    default: {
      const exhaustive: never = state
      return exhaustive
    }
  }
}

const GLYPH_POP = {
  initial: { opacity: 0, scale: 0.25, filter: 'blur(4px)' },
  animate: { opacity: 1, scale: 1, filter: 'blur(0px)' },
  exit: { opacity: 0, scale: 0.25, filter: 'blur(4px)', transition: EXIT },
} as const

const REVEAL = {

  delay: 0.04,

  sweep: 0.16,
  step: 0.02,
  duration: 0.24,

  after: 0.06,
  ease: [0.23, 1, 0.32, 1],
} as const

const graphemes = new Intl.Segmenter(undefined, { granularity: 'grapheme' })

function RevealText({
  text,
  words = false,
  after = 0,
}: {
  text: string
  words?: boolean
  after?: number
}) {
  const reduceMotion = useReducedMotion()

  const parts =
    words || joinsLetters(text)
      ? text.split(/(\s+)/).filter(Boolean)
      : [...graphemes.segment(text)].map(({ segment }) => segment)
  const step = Math.min(REVEAL.step, REVEAL.sweep / Math.max(parts.length - 1, 1))

  return (
    <>
      <span className="sr-only">{text}</span>
      <span aria-hidden>
        {parts.map((part, index) => (
          <motion.span
            key={index}
            initial={reduceMotion ? false : { opacity: 0, filter: 'blur(4px)' }}
            animate={{ opacity: 1, filter: 'blur(0px)' }}
            transition={{
              duration: REVEAL.duration,
              ease: REVEAL.ease,
              delay: REVEAL.delay + after + index * step,
            }}
          >
            {part}
          </motion.span>
        ))}
      </span>
    </>
  )
}

function ToastLine({
  message,
  reveal = false,
  from = 0,
}: {
  message: string
  reveal?: boolean
  from?: number
}) {

  const [revealed] = useState(reveal)
  const fading = !useIsPresent()
  return (
    <motion.div
      initial={{ width: from }}
      animate={{ width: 'auto' }}
      exit={{ opacity: 0, filter: 'blur(4px)', transition: EXIT }}
      transition={MORPH}
      className={cn('flex items-center overflow-hidden', fading && 'absolute inset-y-0 start-0')}
    >
      <span className="w-max max-w-lg shrink-0 truncate">
        {revealed ? <RevealText text={message} /> : message}
      </span>
    </motion.div>
  )
}

function ToastDetail({ text, reveal, from }: { text: string; reveal: boolean; from: number }) {
  const [revealed] = useState(reveal)
  const present = useIsPresent()
  const swapping = usePresenceData() === true
  const fading = !present && swapping
  return (
    <motion.div
      initial={{
        height: from,
        opacity: revealed ? 1 : 0,
        filter: revealed ? 'blur(0px)' : 'blur(4px)',
      }}
      animate={{ height: 'auto', opacity: 1, filter: 'blur(0px)' }}
      exit="leave"
      variants={{
        leave: (swap: boolean) =>
          swap
            ? { opacity: 0, filter: 'blur(4px)', transition: EXIT }
            : { height: 0, opacity: 0, filter: 'blur(4px)' },
      }}
      transition={MORPH}
      className={cn('overflow-hidden', fading && 'absolute inset-x-0 top-0')}
    >

      <p className="pb-0.75 text-pretty text-muted-foreground">
        {revealed ? <RevealText text={text} words after={REVEAL.after} /> : text}
      </p>
    </motion.div>
  )
}

const glyphKey = (state: ToastState) =>
  state === 'pending' || state === 'success' ? 'badge' : state

function ToastPill({
  state,
  message,
  description,
  action,
  onAction,
  onDismiss,
  behind,
}: {
  state?: ToastState
  message: string
  description?: string
  action?: ToastAction
  onAction: () => void
  onDismiss: () => void
  behind: boolean
}) {
  const pill = useRef<HTMLDivElement>(null)
  const lines = useRef<HTMLDivElement>(null)
  const details = useRef<HTMLDivElement>(null)
  const [lineWidth, setLineWidth] = useState(0)
  const [detailHeight, setDetailHeight] = useState(0)
  useLayoutEffect(() => {
    const line = lines.current
    const detail = details.current
    if (!line || !detail) return
    const observer = new ResizeObserver(() => {
      setLineWidth(line.offsetWidth)
      setDetailHeight(detail.offsetHeight)
    })
    observer.observe(line)
    observer.observe(detail)
    return () => observer.disconnect()
  }, [])

  const [last, setLast] = useState({ message, state, description })
  const [reveal, setReveal] = useState(false)
  const [revealDetail, setRevealDetail] = useState(false)
  const [detailSwap, setDetailSwap] = useState(false)
  if (last.message !== message || last.state !== state || last.description !== description) {
    setReveal(last.message !== message)
    setRevealDetail(last.description !== description)
    setDetailSwap(Boolean(last.description) && Boolean(description))
    setLast({ message, state, description })
  }

  return (
    <motion.div
      ref={pill}

      style={{ borderRadius: LINE_HEIGHT / 2 }}
      drag={behind ? false : true}
      dragSnapToOrigin
      dragElastic={0.6}
      dragMomentum={false}
      dragTransition={{ bounceStiffness: 520, bounceDamping: 42 }}
      onDragEnd={(_, info) => {
        const far = Math.hypot(info.offset.x, info.offset.y) > SWIPE
        const fast = Math.hypot(info.velocity.x, info.velocity.y) > FLICK
        if (!far && !fast) return

        if (pill.current) dissolve(pill.current, { onComplete: onDismiss })
        else onDismiss()
      }}
      className={cn(
        'toast-pill relative flex max-w-lg overflow-hidden bg-popover/80 text-sm text-foreground backdrop-blur-md',
        behind ? 'pointer-events-none' : 'pointer-events-auto cursor-grab active:cursor-grabbing',

        action ? 'pe-2' : 'pe-4',
        state ? 'ps-3' : 'ps-4',

        description && 'pt-0.5',
      )}
    >
      <motion.div
        animate={{ opacity: behind ? 0 : 1 }}
        transition={MORPH}

        className="grid min-w-max grid-cols-[auto_minmax(0,1fr)_auto] items-center"
      >

        <div className="col-start-1 row-start-1 flex">
          <AnimatePresence initial={false}>
            {state ? (
              <motion.div
                key="glyph"
                initial={{ width: 0 }}
                animate={{ width: 'auto' }}
                exit={{ width: 0 }}
                transition={MORPH}

                className="flex justify-end"
              >

                <div className="flex w-max items-center pe-3">

                  <AnimatePresence initial={false} mode="popLayout">
                    <motion.span
                      key={glyphKey(state)}
                      {...GLYPH_POP}
                      transition={MORPH}
                      className="flex size-4 shrink-0 items-center justify-center"
                    >
                      <ToastGlyph state={state} />
                    </motion.span>
                  </AnimatePresence>
                </div>
              </motion.div>
            ) : null}
          </AnimatePresence>
        </div>

        <div
          ref={lines}
          className={cn(
            'relative col-start-2 row-start-1 flex h-9 items-center justify-self-start',
            description && 'font-bold',
          )}
        >
          <AnimatePresence initial={false}>
            <ToastLine key={message} message={message} reveal={reveal} from={lineWidth} />
          </AnimatePresence>
        </div>

        <div className="col-start-3 row-start-1 flex">
          <AnimatePresence initial={false}>
            {action ? (
              <motion.div
                key="action"
                initial={{ width: 0 }}
                animate={{ width: 'auto' }}
                exit={{ width: 0 }}
                transition={MORPH}

                className="-my-1 -me-1 flex items-center overflow-hidden rounded-full py-1 pe-1"
              >
                <div className="w-max ps-7">
                  <Button type="button" size="sm" onClick={onAction} className="h-7 rounded-full">
                    {action.label}
                  </Button>
                </div>
              </motion.div>
            ) : null}
          </AnimatePresence>
        </div>

        <div
          ref={details}
          className={cn(
            'toast-detail relative -top-1.5 col-span-2 col-start-2 row-start-2 w-0 min-w-full',
            action ? 'pe-2' : 'pe-0',
          )}
        >
          <AnimatePresence initial={false} custom={detailSwap}>
            {description ? (
              <ToastDetail
                key={description}
                text={description}
                reveal={revealDetail}
                from={detailSwap ? detailHeight : 0}
              />
            ) : null}
          </AnimatePresence>
        </div>

        <motion.div
          aria-hidden
          initial={false}
          animate={{ width: description ? CARD_WIDTH - (state ? 12 : 16) - (action ? 8 : 16) : 0 }}
          transition={MORPH}
          className="col-span-3 col-start-1 row-start-3"
        />
      </motion.div>
    </motion.div>
  )
}

function sameBoxes(
  first: Record<string, { width: number; height: number }>,
  second: Record<string, { width: number; height: number }>,
) {
  const ids = Object.keys(second)
  return (
    ids.length === Object.keys(first).length &&
    ids.every(
      (id) => first[id]?.width === second[id]?.width && first[id]?.height === second[id]?.height,
    )
  )
}

type Side = 'maxWidth' | 'maxHeight'

function useCap(box: RefObject<HTMLDivElement | null>, side: Side, cap: number | null) {
  const limit = useMotionValue<number | string>('none')

  useLayoutEffect(() => {
    const element = box.current
    if (!element) return

    const natural = () => {
      const held = element.style[side]
      element.style[side] = ''
      const size = side === 'maxWidth' ? element.offsetWidth : element.offsetHeight
      element.style[side] = held
      return size
    }
    if (cap === null) {
      if (limit.get() === 'none') return
      const release = () => limit.set('none')
      const controls = animate(limit, natural(), {
        ...MORPH,

        onComplete: () => frame.update(release),
      })
      return () => {
        controls.stop()
        cancelFrame(release)
      }
    }

    if (limit.get() === 'none') limit.set(natural())
    const controls = animate(limit, cap, MORPH)
    return () => controls.stop()
  }, [box, side, cap, limit])

  return limit
}

function DeckCap({
  width,
  height,
  ref,
  children,
}: {
  width: number | null
  height: number | null
  ref: (element: HTMLDivElement | null) => void
  children: ReactNode
}) {
  const box = useRef<HTMLDivElement | null>(null)
  const maxWidth = useCap(box, 'maxWidth', width)
  const maxHeight = useCap(box, 'maxHeight', height)

  return (
    <motion.div
      ref={(element) => {
        box.current = element
        ref(element)
      }}
      style={{ maxWidth, maxHeight }}

      className="relative flex flex-col"
    >
      {children}
    </motion.div>
  )
}

function DeckSlot({
  layer,
  style,
  ...motionProps
}: ComponentProps<typeof motion.div> & { layer: number }) {
  const present = useIsPresent()
  return <motion.div {...motionProps} style={{ ...style, zIndex: present ? layer : 0 }} />
}

function ToastStack({
  cards,
  onOpen,
  position,
}: {
  cards: StackCard[]
  onOpen: (open: boolean) => void
  position: ToastPosition
}) {
  const [pointing, setPointing] = useState(false)
  const [focused, setFocused] = useState(false)
  const [holding, setHolding] = useState(false)
  const [boxes, setBoxes] = useState<Record<string, { width: number; height: number }>>({})
  const measured = useRef(new Map<string, HTMLElement>())
  const root = useRef<HTMLDivElement>(null)
  const textDirection = useDirection()

  const open = pointing || focused || holding
  const piled = cards.length > 1 && !open

  useEffect(() => onOpen(open), [onOpen, open])

  useEffect(() => {
    if (!holding) return
    const release = () => setHolding(false)

    const check = (event: PointerEvent) => {
      if (event.buttons === 0) release()
    }
    window.addEventListener('pointerup', release)
    window.addEventListener('pointercancel', release)
    window.addEventListener('pointermove', check)
    return () => {
      window.removeEventListener('pointerup', release)
      window.removeEventListener('pointercancel', release)
      window.removeEventListener('pointermove', check)
    }
  }, [holding])

  const at = useRef<{ x: number; y: number } | null>(null)
  const overCards = useCallback(
    () =>
      at.current !== null &&
      [...measured.current.values()].some((element) => {
        const box = element.getBoundingClientRect()
        return (
          at.current!.x >= box.left - REACH &&
          at.current!.x <= box.right + REACH &&
          at.current!.y >= box.top - REACH &&
          at.current!.y <= box.bottom + REACH
        )
      }),
    [],
  )

  useEffect(() => {
    const onMove = (event: PointerEvent) => {
      at.current = { x: event.clientX, y: event.clientY }
      setPointing(overCards())
    }

    const onLeave = () => {
      at.current = null
      setPointing(false)
    }

    window.addEventListener('pointermove', onMove)
    document.documentElement.addEventListener('pointerleave', onLeave)
    return () => {
      window.removeEventListener('pointermove', onMove)
      document.documentElement.removeEventListener('pointerleave', onLeave)
    }
  }, [overCards])

  useEffect(() => {
    if (!pointing) return
    let frame = requestAnimationFrame(function check() {
      if (!overCards()) {
        setPointing(false)
        return
      }
      frame = requestAnimationFrame(check)
    })
    return () => cancelAnimationFrame(frame)
  }, [pointing, overCards])

  useEffect(() => {
    if (!focused) return
    let frame = requestAnimationFrame(function check() {
      if (!root.current?.contains(document.activeElement)) {
        setFocused(false)
        return
      }
      frame = requestAnimationFrame(check)
    })
    return () => cancelAnimationFrame(frame)
  }, [focused])

  const measure = useCallback(() => {
    const next: Record<string, { width: number; height: number }> = {}
    for (const [id, element] of measured.current) {

      const { maxWidth, maxHeight } = element.style
      element.style.maxWidth = ''
      element.style.maxHeight = ''
      next[id] = { width: element.offsetWidth, height: element.offsetHeight }
      element.style.maxWidth = maxWidth
      element.style.maxHeight = maxHeight
    }
    setBoxes((current) => (sameBoxes(current, next) ? current : next))
  }, [])

  useLayoutEffect(measure, [cards, measure])

  const sizes = useRef<ResizeObserver | null>(null)
  useEffect(() => {
    const observer = new ResizeObserver(measure)
    for (const element of measured.current.values()) observer.observe(element)
    sizes.current = observer
    return () => observer.disconnect()
  }, [measure])

  const heights = cards.map((card) => boxes[card.id]?.height ?? LINE_HEIGHT)
  const deck = cards[0] ? boxes[cards[0].id] : undefined
  const [side, align] = position.split('-') as [ToastSide, ToastAlign]
  const rtl = textDirection === 'rtl'
  const edge: ToastEdge =
    align === 'left' ? (rtl ? 'end' : 'start') : align === 'right' ? (rtl ? 'start' : 'end') : align

  const corner = edge === 'center' ? 'center' : (edge === 'start') !== rtl ? 'left' : 'right'

  const fall = side === 'top' ? 1 : -1

  return (
    <motion.div
      ref={root}
      aria-live="polite"
      onPointerDown={() => setHolding(true)}
      onFocus={() => setFocused(true)}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) setFocused(false)
      }}
      className={cn(
        'pointer-events-none fixed inset-x-0 z-100',
        side === 'top' ? 'top-4' : 'bottom-4',
      )}
    >
      <AnimatePresence initial={false}>
        {cards.map((card, index) => {
          const slot = piled ? pileSlot(index) : fanSlot(index, heights)

          const drop =
            piled && index > 0
              ? Math.max(0, (heights[0] ?? LINE_HEIGHT) - (heights[index] ?? LINE_HEIGHT))
              : 0
          const y = (slot.y + drop) * fall
          const from = (slot.y + drop - 8) * fall
          return (
            <DeckSlot
              key={card.id}
              layer={cards.length - index}
              style={{

                transformOrigin: `${side} ${corner}`,
              }}
              initial={{ opacity: 0, y: from, scale: slot.scale * 0.98 }}
              animate={{ opacity: slot.opacity, y, scale: slot.scale }}
              exit={{ opacity: 0, y: from, scale: slot.scale * 0.96, transition: EXIT }}
              transition={MORPH}
              className={cn(
                'absolute inset-x-0 flex px-4',
                side === 'top' ? 'top-0' : 'bottom-0',
                JUSTIFY[edge],
              )}
              aria-hidden={slot.opacity === 0 || undefined}
            >
              <DeckCap
                width={piled && index > 0 ? (deck?.width ?? null) : null}
                height={piled && index > 0 ? (deck?.height ?? null) : null}
                ref={(element) => {
                  if (element) {
                    measured.current.set(card.id, element)
                    sizes.current?.observe(element)
                    return
                  }
                  const gone = measured.current.get(card.id)
                  if (gone) sizes.current?.unobserve(gone)
                  measured.current.delete(card.id)
                }}
              >
                {card.render(piled && index > 0)}
              </DeckCap>
            </DeckSlot>
          )
        })}
      </AnimatePresence>
    </motion.div>
  )
}

function subscribeVisibility(onChange: () => void) {
  document.addEventListener('visibilitychange', onChange)
  return () => document.removeEventListener('visibilitychange', onChange)
}

function useDocumentHidden() {
  return useSyncExternalStore(
    subscribeVisibility,
    () => document.hidden,
    () => false,
  )
}

export function Toasts({ position = 'top-center' }: { position?: ToastPosition }) {
  const [notes, setNotes] = useState<Note[]>([])
  const [reading, setReading] = useState(false)
  const hidden = useDocumentHidden()

  const paused = reading || hidden
  const clock = useRef<ToastClock>({ waits: new Map(), since: null })

  const issued = useRef<string | null>(null)

  useEffect(() => {
    clock.current.since = performance.now()

    const onToast = (event: Event) => {
      const note = (event as CustomEvent<Note>).detail
      issued.current = note.id
      setNotes((current) => upsertToast(current, note))
      tick(clock.current, performance.now())
      const delay = dismissDelay(
        note.state,
        note.action !== undefined,
        note.lifetime,
        note.description !== undefined,
      )
      if (delay === null) clock.current.waits.delete(note.id)
      else clock.current.waits.set(note.id, delay)
    }
    const onDismiss = (event: Event) => {
      const id = (event as CustomEvent<string>).detail
      clock.current.waits.delete(id)
      setNotes((current) => current.filter((note) => note.id !== id))
    }

    window.addEventListener(EVENT, onToast)
    window.addEventListener(DISMISS, onDismiss)
    return () => {
      window.removeEventListener(EVENT, onToast)
      window.removeEventListener(DISMISS, onDismiss)
    }
  }, [])

  useEffect(() => {
    tick(clock.current, performance.now())
    clock.current.since = paused ? null : performance.now()
  }, [paused])

  useEffect(() => {
    if (paused) return

    tick(clock.current, performance.now())
    const waits = [...clock.current.waits.values()]
    if (waits.length === 0) return

    const timer = window.setTimeout(
      () => {
        tick(clock.current, performance.now())
        const expired = new Set<string>()
        for (const [id, left] of clock.current.waits) {
          if (left <= 0) expired.add(id)
        }
        for (const id of expired) clock.current.waits.delete(id)
        setNotes((current) => current.filter((note) => !expired.has(note.id)))
      },
      Math.max(0, Math.min(...waits)),
    )
    return () => window.clearTimeout(timer)
  }, [notes, paused])

  return (
    <MotionConfig reducedMotion="user">
      <ToastStack
        position={position}
        onOpen={setReading}
        cards={notes.map((note) => {
          const drop = () => {
            clock.current.waits.delete(note.id)
            setNotes((current) => current.filter((item) => item.id !== note.id))
          }

          const act = () => {
            issued.current = null
            note.action?.run()
            if (issued.current !== note.id) drop()
          }
          return {
            id: note.id,
            render: (behind) => (
              <ToastPill
                state={note.state}
                message={note.message}
                description={note.description}
                action={note.action}
                onAction={act}
                onDismiss={drop}
                behind={behind}
              />
            ),
          }
        })}
      />
    </MotionConfig>
  )
}
