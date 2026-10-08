"use client";
import { useDirection } from '@base-ui/react/direction-provider'
import {
  IconArrowBackUp,
  IconChevronDown,
  IconChevronLeft,
  IconChevronRight,
} from '@tabler/icons-react'
import * as React from 'react'

// Patched on install: Next 15.5's bundled App Router React has no
// `useEffectEvent`, so the calendar carries a ref-based equivalent. Drop this
// block and restore `React.useEffectEvent` once Next ships React 19.2.
function useEffectEvent<T extends (...args: never[]) => unknown>(callback: T): T {
  const latest = React.useRef(callback)
  React.useInsertionEffect(() => {
    latest.current = callback
  })
  return React.useCallback(((...args) => latest.current(...args)) as T, [])
}
import {
  DateLib,
  DayPicker,
  dateMatchModifiers,
  defaultDateLib,
  defaultLocale,
  getDefaultClassNames,
  useDayPicker,
  type CustomComponents,
  type DateLibOptions,
  type DateRange,
  type DayButton,
  type Modifiers,
} from 'react-day-picker'
import { useReducedMotion } from 'motion/react'

import { joinsLetters } from '@/lib/joined-script'
import { listenForSwipes } from '@/lib/trackpad-swipe'
import { cn } from '@/lib/utils'
import { Button, buttonVariants } from '@/components/ui/button'
import {
  Caption,
  createTooltipHandle,
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipViewport,
} from '@/components/ui/tooltip'

function Calendar({
  className,
  classNames,
  showOutsideDays,
  captionLayout = 'label',
  buttonVariant = 'ghost',
  layout = 'default',
  periods = false,
  formatters,
  components,
  onMonthChange,
  ...props
}: React.ComponentProps<typeof DayPicker> & {
  buttonVariant?: React.ComponentProps<typeof Button>['variant']

  layout?: 'default' | 'compact'

  periods?: boolean
}) {
  const defaultClassNames = getDefaultClassNames()
  const compact = layout === 'compact'
  const periodic = periods && (props.numberOfMonths ?? 1) === 1
  const [view, setView] = React.useState<Period>('day')
  if (!periodic && view !== 'day') setView('day')

  const [tip] = React.useState(() => createTooltipHandle<string>())
  const textDirection = useDirection()

  const outsideDays = showOutsideDays ?? (props.numberOfMonths ?? 1) === 1

  const [picked, setPicked] = React.useState<DateRange | undefined>()
  const [aim, setAim] = React.useState<Aim | null>(null)
  const range = props.mode === 'range' ? ('selected' in props ? props.selected : picked) : undefined
  const start = range?.from && (!range.to || sameDay(range.from, range.to)) ? range.from : null
  const span = start && aim && !sameDay(aim.date, start) ? ordered(start, aim.date) : null
  const days = span ? dayNumber(span.to) - dayNumber(span.from) + 1 : 0

  const nights = days - 1
  const allowed =
    span !== null &&
    (props.mode !== 'range' ||
      ((!props.max || nights <= props.max) &&
        (!props.min || props.min <= 1 || nights >= props.min)))
  const preview = allowed ? span : null

  const [landed, setLanded] = React.useState<Landed | null>(null)
  if (preview && landed) setLanded(null)
  const forward = preview !== null && start !== null && sameDay(preview.from, start)

  const clear = React.useCallback(() => setAim(null), [])
  const settle = React.useCallback(() => setLanded(null), [])
  const count = React.useMemo(
    () => ({
      span: preview && aim ? { days, x: aim.x, y: aim.y, instant: aim.instant } : null,
      landed,
      clear,
      settle,
    }),
    [preview, aim, days, landed, clear, settle],
  )

  const aimAt = (date: Date, modifiers: Modifiers, target: Element, instant: boolean) => {
    if (!(target instanceof HTMLElement) || modifiers.disabled || modifiers.hidden) {
      setAim(null)
      return
    }
    let x = target.offsetWidth / 2
    let y = 0
    let node: HTMLElement | null = target
    while (node && node.dataset.slot !== 'calendar') {
      x += node.offsetLeft
      y += node.offsetTop
      node = node.offsetParent instanceof HTMLElement ? node.offsetParent : null
    }
    if (!node) return
    setAim({ date, x, y, instant })
  }

  const previewing =
    props.mode === 'range'
      ? {
          onDayMouseEnter: (date: Date, modifiers: Modifiers, event: React.MouseEvent) => {
            aimAt(date, modifiers, event.currentTarget, false)
            props.onDayMouseEnter?.(date, modifiers, event)
          },

          onDayFocus: (date: Date, modifiers: Modifiers, event: React.FocusEvent) => {
            const target = event.currentTarget
            aimAt(date, modifiers, target, target.matches(':focus-visible'))
            props.onDayFocus?.(date, modifiers, event)
          },
          onSelect: (next: DateRange | undefined, ...rest: RangeSelectRest) => {
            const chosen = restartedOutside(range, next, rest[0])
            const answered =
              preview !== null &&
              aim !== null &&
              !aim.instant &&
              sameDay(aim.date, rest[0]) &&
              chosen?.from !== undefined &&
              chosen.to !== undefined &&
              !sameDay(chosen.from, chosen.to)
            if (answered) setLanded({ days, x: aim.x, y: aim.y })
            setPicked(chosen)
            ;(props.onSelect as RangeSelect | undefined)?.(chosen, ...rest)
          },

          ...('selected' in props ? null : { selected: picked }),
        }
      : null

  const pick = (span: Span, event: React.MouseEvent) => {
    if (props.mode !== 'range') return
    setAim(null)
    setLanded(null)
    setPicked(span)
    ;(props.onSelect as RangeSelect | undefined)?.(span, span.from, {}, event)
  }

  const dayPickerProps = (previewing ? { ...props, ...previewing } : props) as React.ComponentProps<
    typeof DayPicker
  >

  const previousMonth = React.useRef<Date | null>(null)
  const [direction, setDirection] = React.useState<'forward' | 'back' | null>(null)

  const monthCount = props.numberOfMonths ?? 1
  const [drawnCount, setDrawnCount] = React.useState(monthCount)
  if (drawnCount !== monthCount) {
    setDrawnCount(monthCount)
    setDirection(null)
  }

  const handleMonthChange = (month: Date) => {
    setAim(null)
    setLanded(null)
    if (previousMonth.current) {
      setDirection(month > previousMonth.current ? 'forward' : 'back')
    }
    previousMonth.current = month
    onMonthChange?.(month)
  }

  const choose = (next: Period) => {
    const from = PERIODS.findIndex((period) => period.value === view)
    const to = PERIODS.findIndex((period) => period.value === next)
    if (from === to) return
    setAim(null)
    setLanded(null)
    setDirection(to > from ? 'forward' : 'back')
    setView(next)
  }
  const period = { periodic, compact, view, direction, choose, pick, tip, buttonVariant }

  const parts = React.useMemo(
    () =>
      ({
        Root: CalendarRoot,
        Chevron: ({ className, orientation, ...chevronProps }) => {

          if (orientation === 'left') {
            return (
              <IconChevronLeft
                className={cn('size-4 in-[.rdp-nav]:[&:dir(rtl)]:-scale-x-100', className)}
                {...chevronProps}
              />
            )
          }

          if (orientation === 'right') {
            return (
              <IconChevronRight
                className={cn('size-4 in-[.rdp-nav]:[&:dir(rtl)]:-scale-x-100', className)}
                {...chevronProps}
              />
            )
          }

          return <IconChevronDown className={cn('size-4', className)} {...chevronProps} />
        },
        DayButton: CalendarDayButton,
        WeekNumber: ({ children, ...weekProps }) => (
          <td {...weekProps}>
            <div className="flex size-(--cell-size) items-center justify-center text-center">
              {children}
            </div>
          </td>
        ),
      }) satisfies Partial<CustomComponents>,
    [],
  )

  const Month = React.useMemo(
    () =>
      (({ calendarMonth, displayIndex, className, ...monthProps }) => (
        <div
          key={calendarMonth.date.getTime()}
          ref={() => {
            previousMonth.current ??= calendarMonth.date
          }}
          data-direction={periodic ? undefined : direction}

          className={cn(!periodic && direction && 'calendar-month', className)}
          {...monthProps}
        />
      )) satisfies CustomComponents['Month'],

    [periodic, periodic ? null : direction],
  )

  const calendar = (
    <CalendarSpanContext.Provider value={count}>
      <DayPicker
        onMonthChange={handleMonthChange}
        showOutsideDays={outsideDays}

        dir={textDirection === 'rtl' ? 'rtl' : undefined}
        className={cn(
          'group/calendar bg-background p-(--calendar-inset) [--cell-radius:var(--radius-md)] in-data-[slot=card-content]:bg-transparent in-data-[slot=popover-content]:bg-transparent',

          periodic && 'pb-0',
          compact
            ? '[--calendar-inset:--spacing(2)] [--cell-size:--spacing(7)]'
            : '[--calendar-inset:--spacing(3)] [--cell-size:--spacing(9)]',
          className,
        )}
        captionLayout={periodic ? 'label' : captionLayout}
        formatters={{

          formatMonthDropdown: (date, dateLib = defaultDateLib) => dateLib.format(date, 'LLL'),

          ...(compact && periodic
            ? {
                formatCaption: (date: Date, options?: DateLibOptions, dateLib?: DateLib) =>
                  (dateLib ?? new DateLib(options)).format(date, 'LLL y'),
              }
            : props.dateLib
              ? {

                  formatCaption: (date: Date, options?: DateLibOptions, dateLib?: DateLib) =>
                    (dateLib ?? new DateLib(options, props.dateLib)).format(date, 'LLLL y'),
                }
              : null),

          formatWeekdayName: (date: Date, options?: DateLibOptions, dateLib?: DateLib) => {
            const lib = dateLib ?? new DateLib(options)
            const short = lib.format(date, compact ? 'cccccc' : 'ccc')
            return joinsLetters(short) ? lib.format(date, 'ccccc') : short
          },
          ...formatters,
        }}
        classNames={{
          root: cn('relative w-fit', defaultClassNames.root),

          months: cn(
            'relative flex flex-col gap-4 md:flex-row',
            periodic && 'gap-0 md:flex-col',
            defaultClassNames.months,
          ),
          month: cn('flex w-full flex-col gap-4', periodic && 'gap-0', defaultClassNames.month),
          nav: cn(
            periodic
              ? 'absolute end-0 top-0 flex h-(--cell-size) items-center gap-1'
              : 'absolute inset-x-0 top-0 flex w-full items-center justify-between gap-1',
            defaultClassNames.nav,
          ),
          button_previous: cn(
            buttonVariants({ variant: buttonVariant }),
            'size-(--cell-size) p-0 select-none aria-disabled:opacity-50',
            defaultClassNames.button_previous,
          ),
          button_next: cn(
            buttonVariants({ variant: buttonVariant }),
            'size-(--cell-size) p-0 select-none aria-disabled:opacity-50',
            defaultClassNames.button_next,
          ),
          month_caption: cn(
            'flex h-(--cell-size) w-full items-center',
            periodic
              ? 'justify-start pe-[calc(var(--cell-size)*2+--spacing(1))]'
              : 'justify-center px-(--cell-size)',
            defaultClassNames.month_caption,
          ),

          dropdowns: cn(
            'flex h-(--cell-size) w-full items-center gap-1.5 text-sm font-semibold',
            periodic ? 'justify-start' : 'justify-center',
            defaultClassNames.dropdowns,
          ),
          dropdown_root: cn(
            'relative rounded-(--cell-radius) has-focus-visible:ring-[3px] has-focus-visible:ring-ring/50',

            periodic && 'hover:bg-foreground/7',
            defaultClassNames.dropdown_root,
          ),
          dropdown: cn('absolute inset-0 bg-popover opacity-0', defaultClassNames.dropdown),
          caption_label: cn(
            'font-semibold select-none',
            captionLayout === 'label' && !periodic
              ? 'text-sm'
              : 'flex items-center gap-1 rounded-(--cell-radius) text-sm [&>svg]:size-3.5 [&>svg]:text-muted-foreground',

            periodic && 'px-2 py-1',
            defaultClassNames.caption_label,
          ),
          month_grid: cn('w-full border-collapse', defaultClassNames.month_grid),
          weekdays: cn('flex', defaultClassNames.weekdays),
          weekday: cn(
            'flex-1 rounded-(--cell-radius) font-normal text-muted-foreground select-none',
            compact ? 'text-[0.8rem]' : 'text-sm',
            defaultClassNames.weekday,
          ),
          week: cn('mt-2 flex w-full', defaultClassNames.week),
          week_number_header: cn(
            'w-(--cell-size) select-none',
            defaultClassNames.week_number_header,
          ),
          week_number: cn(
            'text-[0.8rem] text-muted-foreground select-none',
            defaultClassNames.week_number,
          ),

          day: cn(
            'group/day relative aspect-square h-full w-full rounded-(--cell-radius) p-0 text-center select-none [&:last-child[data-selected=true]_button:not([data-selected-single=true])]:rounded-e-(--cell-radius)',
            props.showWeekNumber
              ? '[&:nth-child(2)[data-selected=true]_button:not([data-selected-single=true])]:rounded-s-(--cell-radius)'
              : '[&:first-child[data-selected=true]_button:not([data-selected-single=true])]:rounded-s-(--cell-radius)',

            '[&:has(+[data-hidden=true])[data-selected=true]_button:not([data-selected-single=true])]:rounded-e-(--cell-radius) [[data-hidden=true]+&[data-selected=true]_button:not([data-selected-single=true])]:rounded-s-(--cell-radius)',
            defaultClassNames.day,
          ),

          range_start: cn(
            'relative isolate z-0 rounded-s-(--cell-radius) bg-(--calendar-range) after:absolute after:inset-y-0 after:end-0 after:w-4 after:bg-(--calendar-range) has-[+[data-hidden=true]]:rounded-e-(--cell-radius) has-[+[data-hidden=true]]:after:hidden',
            defaultClassNames.range_start,
          ),
          range_middle: cn('rounded-none', defaultClassNames.range_middle),
          range_end: cn(
            'relative isolate z-0 rounded-e-(--cell-radius) bg-(--calendar-range) after:absolute after:inset-y-0 after:start-0 after:w-4 after:bg-(--calendar-range) [[data-hidden=true]+&]:rounded-s-(--cell-radius) [[data-hidden=true]+&]:after:hidden',
            defaultClassNames.range_end,
          ),

          today: cn('rounded-none', defaultClassNames.today),
          outside: cn(
            'text-muted-foreground aria-selected:text-muted-foreground',
            defaultClassNames.outside,
          ),
          disabled: cn('text-muted-foreground opacity-50', defaultClassNames.disabled),
          hidden: cn('invisible', defaultClassNames.hidden),
          ...classNames,
        }}

        components={{
          ...parts,
          Month,
          ...(periodic
            ? {
                Nav: CalendarNav,
                Months: CalendarMonths,
                MonthGrid: CalendarMonthGrid,
                CaptionLabel: CalendarCaptionLabel,
              }
            : null),
          ...components,
        }}
        {...dayPickerProps}
        modifiers={{
          ...props.modifiers,
          preview_middle: preview ? { after: preview.from, before: preview.to } : false,
          preview_start_before: preview && forward ? preview.from : false,
          preview_start_after: preview && !forward ? preview.to : false,
          preview_end_after: preview && forward ? preview.to : false,
          preview_end_before: preview && !forward ? preview.from : false,
        }}
        modifiersClassNames={{
          preview_middle:
            'rounded-none bg-(--calendar-range) first:rounded-s-(--cell-radius) last:rounded-e-(--cell-radius) has-[+[data-hidden=true]]:rounded-e-(--cell-radius) [[data-hidden=true]+&]:rounded-s-(--cell-radius)',
          preview_start_before:
            'before:absolute before:inset-y-0 before:end-0 before:w-1/2 before:bg-(--calendar-range) last:before:hidden has-[+[data-hidden=true]]:before:hidden',
          preview_start_after:
            'before:absolute before:inset-y-0 before:start-0 before:w-1/2 before:bg-(--calendar-range) first:before:hidden [[data-hidden=true]+&]:before:hidden',
          preview_end_after:
            'bg-(--calendar-range) rounded-s-none first:rounded-s-(--cell-radius) [[data-hidden=true]+&]:rounded-s-(--cell-radius)',
          preview_end_before:
            'bg-(--calendar-range) rounded-e-none last:rounded-e-(--cell-radius) has-[+[data-hidden=true]]:rounded-e-(--cell-radius)',
          ...props.modifiersClassNames,
        }}
      />
    </CalendarSpanContext.Provider>
  )

  return <CalendarPeriodContext.Provider value={period}>{calendar}</CalendarPeriodContext.Provider>
}

type Aim = { date: Date; x: number; y: number; instant: boolean }

type Landed = { days: number; x: number; y: number }

type RangeSelect = (range: DateRange | undefined, ...rest: RangeSelectRest) => void
type RangeSelectRest = [Date, Modifiers, React.MouseEvent | React.KeyboardEvent]

const dayNumber = (date: Date) =>
  Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()) / 86_400_000

const sameDay = (a: Date, b: Date) => dayNumber(a) === dayNumber(b)

const ordered = (a: Date, b: Date) => (a < b ? { from: a, to: b } : { from: b, to: a })

function restartedOutside(
  range: DateRange | undefined,
  next: DateRange | undefined,
  day: Date,
): DateRange | undefined {
  if (!range?.from || !range.to || sameDay(range.from, range.to)) return next
  const at = dayNumber(day)
  return at < dayNumber(range.from) || at > dayNumber(range.to) ? { from: day, to: day } : next
}

const CalendarSpanContext = React.createContext<{
  span: { days: number; x: number; y: number; instant: boolean } | null
  landed: Landed | null
  clear: () => void
  settle: () => void
}>({ span: null, landed: null, clear: () => {}, settle: () => {} })

type Period = 'day' | 'month' | 'quarter' | 'year'

const PERIODS: ReadonlyArray<{ value: Period; label: string }> = [
  { value: 'day', label: 'Day' },
  { value: 'month', label: 'Month' },
  { value: 'quarter', label: 'Quarter' },
  { value: 'year', label: 'Year' },
]

const UNITS = ['month', 'quarter', 'year'] as const satisfies ReadonlyArray<Period>

type Span = { from: Date; to: Date }

function spanOf(lib: DateLib, unit: Exclude<Period, 'day'>, year: number, index: number): Span {
  const months = unit === 'year' ? 12 : unit === 'quarter' ? 3 : 1
  return {
    from: lib.newDate(year, index * months, 1),
    to: lib.addDays(lib.newDate(year, (index + 1) * months, 1), -1),
  }
}

const within = (date: Date, span: Span) =>
  dayNumber(date) >= dayNumber(span.from) && dayNumber(date) <= dayNumber(span.to)

const CalendarPeriodContext = React.createContext<{
  periodic: boolean
  compact: boolean
  view: Period
  direction: 'forward' | 'back' | null
  choose: (next: Period) => void
  pick: (span: Span, event: React.MouseEvent) => void
  tip: ReturnType<typeof createTooltipHandle<string>>
  buttonVariant: React.ComponentProps<typeof Button>['variant']
} | null>(null)

function CalendarRoot({
  className,
  rootRef,
  children,
  ...rootProps
}: React.ComponentProps<CustomComponents['Root']>) {
  const { span, landed, clear, settle } = React.useContext(CalendarSpanContext)
  const count = span ?? landed
  const { nextMonth, previousMonth, goToMonth, dayPickerProps } = useDayPicker()
  const box = React.useRef<HTMLDivElement | null>(null)

  const lang = dayPickerProps.locale?.code ?? defaultLocale.code
  const numerals = dayPickerProps.numerals ?? 'latn'
  const dayCount = React.useMemo(
    () =>
      new Intl.NumberFormat(lang, {
        style: 'unit',
        unit: 'day',
        unitDisplay: 'long',
        numberingSystem: numerals,
        useGrouping: false,
      }),
    [lang, numerals],
  )
  const period = React.useContext(CalendarPeriodContext)
  const view = period?.periodic ? period.view : 'day'

  const turn = useEffectEvent((step: 1 | -1) => {
    const rtl = box.current ? getComputedStyle(box.current).direction === 'rtl' : false
    const forward = rtl ? step < 0 : step > 0
    const month = forward ? nextMonth : previousMonth
    if (!month) return
    goToMonth(month)
    ;(forward ? dayPickerProps.onNextClick : dayPickerProps.onPrevClick)?.(month)
  })
  const pageable = useEffectEvent(() => view === 'day' && Boolean(nextMonth ?? previousMonth))

  React.useEffect(() => {
    const element = box.current
    if (!element) return
    return listenForSwipes(
      element,
      (step) => turn(step),
      () => pageable(),
    )
  }, [])

  const ref = React.useCallback(
    (element: HTMLDivElement | null) => {
      box.current = element
      if (typeof rootRef === 'function') rootRef(element)
      else if (rootRef) rootRef.current = element
    },
    [rootRef],
  )

  return (
    <div
      data-slot="calendar"
      ref={ref}
      data-period-view={view === 'day' ? undefined : view}
      className={cn(className)}
      onPointerLeave={clear}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) clear()
      }}
      {...rootProps}
    >

      <CalendarResize key={dayPickerProps.numberOfMonths ?? 1}>{children}</CalendarResize>
      {count ? (
        <span
          aria-hidden
          data-instant={span?.instant || undefined}
          data-landed={span ? undefined : true}
          className="calendar-span"
          style={
            {
              '--calendar-span-x': `${String(count.x)}px`,
              '--calendar-span-y': `${String(count.y)}px`,
            } as React.CSSProperties
          }
          onTransitionEnd={(event) => {
            if (!span && event.propertyName === 'opacity') settle()
          }}
        >
          {dayCount.format(count.days)}
        </span>
      ) : null}
    </div>
  )
}

function CalendarResize({ children }: { children?: React.ReactNode }) {
  const content = React.useRef<HTMLDivElement>(null)
  const measured = React.useRef<number | null>(null)
  const [height, setHeight] = React.useState<number>()
  const [resizing, setResizing] = React.useState(false)
  const reduceMotion = useReducedMotion()

  React.useLayoutEffect(() => {
    const element = content.current
    if (!element) return

    const observer = new ResizeObserver(() => {
      const next = element.offsetHeight

      if (next === 0) {
        measured.current = null
        setHeight(undefined)
        return
      }
      if (measured.current === next) return

      if (measured.current !== null && !reduceMotion) setResizing(true)
      measured.current = next
      setHeight(next)
    })

    observer.observe(element)
    return () => observer.disconnect()
  }, [reduceMotion])

  return (
    <div

      data-resizing={resizing || undefined}
      className="calendar-resize"
      style={{ height }}
      onTransitionEnd={(event) => {

        if (event.target === event.currentTarget && event.propertyName === 'height') {
          setResizing(false)
        }
      }}
    >
      <div ref={content}>{children}</div>
    </div>
  )
}

function CalendarCaptionLabel({
  children,
  ...labelProps
}: React.ComponentProps<CustomComponents['CaptionLabel']>) {
  const period = React.useContext(CalendarPeriodContext)
  const { classNames } = useDayPicker()
  if (!period) return <span {...labelProps}>{children}</span>

  return (
    <span className={classNames.dropdown_root}>
      <select
        aria-label="Period"
        value={period.view}
        onChange={(event) => period.choose(event.currentTarget.value as Period)}
        className={classNames.dropdown}
      >
        {PERIODS.map(({ value, label }) => (
          <option key={value} value={value}>
            {label}
          </option>
        ))}
      </select>
      <span {...labelProps}>

        {period.view === 'day'
          ? children
          : PERIODS.find(({ value }) => value === period.view)?.label}
        <IconChevronDown aria-hidden />
      </span>
    </span>
  )
}

function CalendarMonths({
  children,
  ...monthsProps
}: React.ComponentProps<CustomComponents['Months']>) {
  const period = React.useContext(CalendarPeriodContext)
  const view = period?.view ?? 'day'
  return (
    <div {...monthsProps}>
      {children}
      {view === 'day' ? null : <CalendarPeriods key={view} unit={view} />}
    </div>
  )
}

function CalendarMonthGrid(gridProps: React.ComponentProps<CustomComponents['MonthGrid']>) {
  const period = React.useContext(CalendarPeriodContext)
  const { months } = useDayPicker()

  const measure = React.useCallback((element: HTMLDivElement | null) => {
    const box = element?.closest<HTMLElement>('[data-slot="calendar"]')
    if (!element || !box) return
    const observer = new ResizeObserver(() => {
      box.style.setProperty('--calendar-days-width', `${String(element.offsetWidth)}px`)
      box.style.setProperty('--calendar-days-height', `${String(element.offsetHeight)}px`)
    })
    observer.observe(element)
    return () => observer.disconnect()
  }, [])

  if (!period) return <table {...gridProps} />
  if (period.view !== 'day') return <></>

  return (
    <div
      key={months[0]?.date.getTime()}
      ref={measure}
      data-direction={period.direction ?? undefined}
      className={cn('pt-4 pb-(--calendar-inset)', period.direction && 'calendar-month')}
    >
      <table {...gridProps} />
    </div>
  )
}

function CalendarNav({
  onPreviousClick,
  onNextClick,
  previousMonth,
  nextMonth,
  ...navProps
}: React.ComponentProps<CustomComponents['Nav']>) {
  const { classNames, labels } = useDayPicker()
  const period = React.useContext(CalendarPeriodContext)
  if (!period) return <nav {...navProps} />

  if (period.view !== 'day') return <></>
  const { tip } = period

  return (
    <nav {...navProps}>
      <TooltipTrigger
        handle={tip}
        payload="Previous month"
        closeOnClick={false}
        type="button"
        aria-label={labels.labelPrevious(previousMonth)}
        aria-disabled={previousMonth ? undefined : true}
        tabIndex={previousMonth ? undefined : -1}
        onClick={(event) => {
          if (previousMonth) onPreviousClick?.(event)
        }}
        className={classNames.button_previous}
      >
        <IconChevronLeft className="size-4 [&:dir(rtl)]:-scale-x-100" />
      </TooltipTrigger>
      <TooltipTrigger
        handle={tip}
        payload="Next month"
        closeOnClick={false}
        type="button"
        aria-label={labels.labelNext(nextMonth)}
        aria-disabled={nextMonth ? undefined : true}
        tabIndex={nextMonth ? undefined : -1}
        onClick={(event) => {
          if (nextMonth) onNextClick?.(event)
        }}
        className={classNames.button_next}
      >
        <IconChevronRight className="size-4 [&:dir(rtl)]:-scale-x-100" />
      </TooltipTrigger>
      <CalendarTip handle={tip} />
    </nav>
  )
}

function CalendarTip({ handle }: { handle: ReturnType<typeof createTooltipHandle<string>> }) {
  return (
    <Tooltip handle={handle}>
      {({ payload }) => (
        <TooltipContent>
          <TooltipViewport>
            <Caption>{payload ?? ''}</Caption>
          </TooltipViewport>
        </TooltipContent>
      )}
    </Tooltip>
  )
}

function CalendarPeriods({ unit }: { unit: (typeof UNITS)[number] }) {
  const { months, goToMonth, dayPickerProps, formatters, selected, classNames } = useDayPicker()
  const period = React.useContext(CalendarPeriodContext)
  const id = React.useId()
  const list = React.useRef<HTMLDivElement>(null)
  const refs = React.useRef<Array<HTMLButtonElement | null>>([])

  const { mode, disabled, startMonth, endMonth, dir } = dayPickerProps
  const shown = months[0]?.date ?? new Date()
  const today = dayPickerProps.today ?? new Date()

  const dateLib = React.useMemo(
    () =>
      new DateLib(
        {
          locale: { ...defaultLocale, ...dayPickerProps.locale },
          numerals: dayPickerProps.numerals,
        },
        dayPickerProps.dateLib,
      ),
    [dayPickerProps.locale, dayPickerProps.numerals, dayPickerProps.dateLib],
  )
  const [years] = React.useState(() => {
    const first = startMonth ? dateLib.getYear(startMonth) : dateLib.getYear(shown) - 10
    const last = endMonth ? dateLib.getYear(endMonth) : dateLib.getYear(shown) + 10
    return Array.from({ length: Math.max(1, last - first + 1) }, (_, offset) => first + offset)
  })
  const perYear = unit === 'month' ? 12 : unit === 'quarter' ? 4 : 1
  const columns = unit === 'quarter' ? 2 : 3

  const monthNames = React.useMemo(
    () =>
      Array.from({ length: 12 }, (_, month) =>
        formatters.formatMonthDropdown(dateLib.newDate(years[0]!, month, 1), dateLib),
      ),
    [formatters, dateLib, years],
  )

  const cells = years.flatMap((year) =>
    Array.from({ length: perYear }, (_, index) => {
      const span = spanOf(dateLib, unit, year, index)
      const label =
        unit === 'month'
          ? (monthNames[index] ?? '')
          : unit === 'quarter'
            ? `Q${dateLib.formatNumber(index + 1)}`
            : formatters.formatYearDropdown(span.from, dateLib)
      return { year, span, label }
    }),
  )

  const isPicked = (span: Span) => {
    if (mode === 'range') {
      const range = selected as DateRange | undefined
      return Boolean(
        range?.from && range.to && sameDay(range.from, span.from) && sameDay(range.to, span.to),
      )
    }
    const dates =
      mode === 'multiple'
        ? ((selected as Date[] | undefined) ?? [])
        : [selected as Date | undefined]
    return dates.some((date) => date !== undefined && within(date, span))
  }

  const isBlocked = (span: Span) => {
    if (startMonth && span.to < dateLib.startOfMonth(startMonth)) return true
    if (endMonth && span.from > dateLib.endOfMonth(endMonth)) return true
    if (!disabled) return false
    for (const day = new Date(span.from); day <= span.to; day.setDate(day.getDate() + 1)) {
      if (!dateMatchModifiers(day, disabled)) return false
    }
    return true
  }

  const opening = Math.max(
    0,
    cells.findIndex(({ span }) => within(shown, span)),
  )
  const current = cells.findIndex(({ span }) => within(today, span))
  const [focus, setFocus] = React.useState(opening)

  const reveal = (index: number, behavior: ScrollBehavior) => {
    const element = list.current
    const cell = refs.current[index]
    if (!element || !cell) return
    element.scrollTo({
      top:
        unit === 'year'
          ? cell.offsetTop - (element.clientHeight - cell.offsetHeight) / 2
          : (cell.closest<HTMLElement>('[data-year]')?.offsetTop ?? 0) -
            Number.parseFloat(getComputedStyle(element).paddingTop),
      behavior,
    })
  }

  React.useLayoutEffect(() => {
    const element = list.current
    const cell = refs.current[opening]
    if (!element || !cell) return

    if (unit === 'year' && cell.offsetTop + cell.offsetHeight <= element.clientHeight) return
    reveal(opening, 'instant')

  }, [])

  const jump = () => {
    setFocus(current)
    reveal(current, matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth')
  }

  const activate = (span: Span, event: React.MouseEvent) => {
    if (!period || isBlocked(span)) return
    goToMonth(span.from)
    if (mode === 'range') {
      period.pick(span, event)
      return
    }
    period.choose(unit === 'year' ? 'month' : 'day')
  }

  const move = (event: React.KeyboardEvent, index: number) => {
    const back = dir === 'rtl' ? 1 : -1
    const step =
      event.key === 'ArrowLeft'
        ? back
        : event.key === 'ArrowRight'
          ? -back
          : event.key === 'ArrowUp'
            ? -columns
            : event.key === 'ArrowDown'
              ? columns
              : event.key === 'Home'
                ? -(index % columns)
                : event.key === 'End'
                  ? columns - 1 - (index % columns)
                  : null
    if (step === null) return
    event.preventDefault()
    const next = Math.min(cells.length - 1, Math.max(0, index + step))
    setFocus(next)
    refs.current[next]?.focus()
  }

  const pill = ({ span, label, year }: (typeof cells)[number], index: number) => {
    const picked = isPicked(span)
    const blocked = isBlocked(span)
    return (
      <button
        key={`${String(year)}-${label}`}
        ref={(element) => {
          refs.current[index] = element
        }}
        type="button"
        tabIndex={index === focus ? 0 : -1}
        aria-label={
          unit === 'month'
            ? dateLib.format(span.from, 'LLLL y')
            : unit === 'quarter'
              ? `${label} ${formatters.formatYearDropdown(span.from, dateLib)}`
              : label
        }
        aria-pressed={mode === 'range' ? picked : undefined}
        aria-current={mode !== 'range' && picked ? 'date' : undefined}
        aria-disabled={blocked || undefined}
        data-picked={picked || undefined}
        data-now={within(today, span) || undefined}
        onClick={(event) => activate(span, event)}
        onKeyDown={(event) => move(event, index)}
        onFocus={() => setFocus(index)}
        className={cn(

          'flex h-(--cell-size) items-center justify-center rounded-full border border-input tabular-nums ring-0 transition-[box-shadow,scale] duration-(--ring-duration) ease-(--ease-out) outline-none select-none ring-inset not-data-picked:hover:bg-foreground/5 focus-visible:ring-[3px] focus-visible:ring-ring/50 active:scale-[0.97] aria-disabled:pointer-events-none aria-disabled:opacity-50 data-now:not-data-picked:text-(--calendar-today) data-picked:border-transparent data-picked:bg-(--calendar-active) data-picked:text-(--calendar-active-foreground) data-picked:hover:brightness-110 motion-reduce:transition-none',
          period?.compact ? 'text-xs' : 'text-sm',
        )}
      >
        {label}
      </button>
    )
  }

  return (
    <>
      {period && current >= 0 ? (
        <div className={classNames.nav}>
          <TooltipTrigger
            handle={period.tip}
            payload="Jump to current"
            closeOnClick={false}
            type="button"
            aria-label={`Jump to the current ${unit}`}
            onClick={jump}
            className={cn(
              buttonVariants({ variant: period.buttonVariant }),
              'size-(--cell-size) p-0 select-none',
            )}
          >
            <IconArrowBackUp className="size-4 [&:dir(rtl)]:-scale-x-100" />
          </TooltipTrigger>
          <CalendarTip handle={period.tip} />
        </div>
      ) : null}

      <div aria-hidden className="-mx-(--calendar-inset) mt-(--calendar-inset) border-b" />
      <div
        ref={list}
        data-direction={period?.direction ?? undefined}
        className={cn(

          'relative -me-(--calendar-inset) h-[calc(var(--calendar-days-height,calc(var(--cell-size)*7+--spacing(4)+var(--calendar-inset)))-var(--calendar-inset)-1px)] w-[calc(var(--calendar-days-width,calc(var(--cell-size)*7))+var(--calendar-inset))] overflow-y-auto overscroll-contain pt-4 pb-(--calendar-inset)',
          period?.direction && 'calendar-month',
        )}
      >

        <div className="max-w-[var(--calendar-days-width,calc(var(--cell-size)*7))]">
          {unit === 'year' ? (
            <div className="grid grid-cols-3 gap-2" role="group" aria-label="Years">
              {cells.map(pill)}
            </div>
          ) : (
            years.map((year, row) => (
              <div
                key={year}
                data-year={year}
                role="group"
                aria-labelledby={`${id}-${String(year)}`}

                className="pb-4 last:pb-0"
              >
                <div
                  id={`${id}-${String(year)}`}
                  className="px-1 pb-2 text-sm text-muted-foreground tabular-nums"
                >
                  {formatters.formatYearDropdown(dateLib.newDate(year, 0, 1), dateLib)}
                </div>
                <div className={cn('grid gap-2', columns === 2 ? 'grid-cols-2' : 'grid-cols-3')}>
                  {cells
                    .slice(row * perYear, (row + 1) * perYear)
                    .map((cell, offset) => pill(cell, row * perYear + offset))}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </>
  )
}

function CalendarDayButton({
  className,
  day,
  modifiers,
  ...props
}: React.ComponentProps<typeof DayButton>) {
  const defaultClassNames = getDefaultClassNames()

  const ref = React.useRef<HTMLButtonElement>(null)
  React.useEffect(() => {
    if (modifiers.focused) ref.current?.focus()
  }, [modifiers.focused])

  return (
    <Button
      ref={ref}
      variant="ghost"
      size="icon"

      data-day={day.date.toLocaleDateString('en-US')}
      data-selected-single={
        modifiers.selected &&
        !modifiers.range_start &&
        !modifiers.range_end &&
        !modifiers.range_middle
      }
      data-selected={modifiers.selected || undefined}
      data-range-start={modifiers.range_start}
      data-range-end={modifiers.range_end}
      data-range-middle={modifiers.range_middle}
      className={cn(

        'relative isolate z-10 flex aspect-square size-auto w-full min-w-(--cell-size) flex-col gap-1 border-0 leading-none font-normal ring-0 transition-[box-shadow] duration-(--ring-duration) ease-(--ease-out) ring-inset before:transition-[scale] focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none data-[range-end=true]:rounded-(--cell-radius) data-[range-end=true]:bg-(--calendar-selected) data-[range-end=true]:text-(--calendar-selected-foreground) data-[range-middle=true]:rounded-none data-[range-middle=true]:bg-(--calendar-range) data-[range-middle=true]:text-foreground data-[range-start=true]:rounded-(--cell-radius) data-[range-start=true]:bg-(--calendar-selected) data-[range-start=true]:text-(--calendar-selected-foreground) data-[selected-single=true]:rounded-(--cell-radius) data-[selected-single=true]:bg-(--calendar-selected) data-[selected-single=true]:text-(--calendar-selected-foreground) data-[selected=true]:hover:before:bg-foreground/5 motion-reduce:transition-none dark:not-data-[selected=true]:hover:text-foreground dark:data-[selected=true]:hover:before:bg-foreground/5 [&>span]:text-xs [&>span]:opacity-70',

        'group-data-[today=true]/day:rounded-(--cell-radius) group-data-[today=true]/day:bg-(--calendar-today) group-data-[today=true]/day:text-(--calendar-today-foreground) group-data-[today=true]/day:hover:brightness-110',

        modifiers.outside && !modifiers.today && 'text-muted-foreground',
        defaultClassNames.day,
        className,
      )}
      {...props}
    />
  )
}

export { Calendar, CalendarDayButton }
