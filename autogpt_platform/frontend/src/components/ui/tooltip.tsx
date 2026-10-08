'use client'

import { Tooltip as TooltipPrimitive } from '@base-ui/react/tooltip'
import { type RefObject, useLayoutEffect, useRef } from 'react'
import { TextMorph } from 'torph/react'

import { joinsLetters } from '@/lib/joined-script'
import { cn } from '@/lib/utils'

const TRAVEL = 'duration-200 ease-[cubic-bezier(0.22,1,0.36,1)]'

const CLOSE_DELAY = 140

function TooltipProvider({
  delay = 0,
  closeDelay = CLOSE_DELAY,
  ...props
}: TooltipPrimitive.Provider.Props) {
  return (
    <TooltipPrimitive.Provider
      data-slot="tooltip-provider"
      delay={delay}
      closeDelay={closeDelay}
      {...props}
    />
  )
}

function Tooltip<Payload>({ ...props }: TooltipPrimitive.Root.Props<Payload>) {
  return <TooltipPrimitive.Root data-slot="tooltip" {...props} />
}

function TooltipTrigger<Payload>({ ...props }: TooltipPrimitive.Trigger.Props<Payload>) {
  return <TooltipPrimitive.Trigger data-slot="tooltip-trigger" {...props} />
}

const createTooltipHandle = TooltipPrimitive.createHandle

function TooltipContent({
  className,
  side = 'top',
  sideOffset = 6,
  align = 'center',
  alignOffset = 0,
  children,
  ...props
}: TooltipPrimitive.Popup.Props &
  Pick<TooltipPrimitive.Positioner.Props, 'align' | 'alignOffset' | 'side' | 'sideOffset'>) {
  return (
    <TooltipPrimitive.Portal>
      <TooltipPrimitive.Positioner
        align={align}
        alignOffset={alignOffset}
        side={side}
        sideOffset={sideOffset}
        className={cn(
          'isolate z-[110] h-[var(--positioner-height)] w-[var(--positioner-width)] max-w-[var(--available-width)] transition-[top,right,bottom,left,transform] data-instant:transition-none motion-reduce:transition-none',
          TRAVEL,
        )}
      >
        <TooltipPrimitive.Popup
          data-slot="tooltip-content"
          className={cn(
            'z-50 inline-flex h-[var(--popup-height,auto)] w-[var(--popup-width,auto)] max-w-xs origin-(--transform-origin) items-center justify-center gap-1.5 rounded-lg bg-foreground px-[var(--tooltip-inset)] py-1.5 text-[0.8125rem] font-medium text-background transition-[width,height,transform,opacity] [--tooltip-inset:0.625rem] has-data-[slot=kbd]:pe-1.5 data-ending-style:scale-[0.98] data-ending-style:opacity-0 data-ending-style:duration-100 data-instant:transition-none data-starting-style:scale-95 data-starting-style:opacity-0 **:data-[slot=kbd]:h-[1.125rem] **:data-[slot=kbd]:min-w-[1.125rem] **:data-[slot=kbd]:rounded-sm **:data-[slot=kbd]:text-[0.6875rem] **:data-[slot=kbd-group]:gap-0.5 motion-reduce:transition-none dark:bg-popover dark:text-popover-foreground dark:shadow-[0_2px_8px_rgb(0_0_0/0.4)] dark:ring-1 dark:ring-white/15',
            TRAVEL,
            className,
          )}
          {...props}
        >
          {children}
        </TooltipPrimitive.Popup>
      </TooltipPrimitive.Positioner>
    </TooltipPrimitive.Portal>
  )
}

function TooltipArrow({ className, ...props }: TooltipPrimitive.Arrow.Props) {
  return (
    <TooltipPrimitive.Arrow
      data-slot="tooltip-arrow"
      className={cn(

        'absolute size-2 rotate-45 rounded-[1px] bg-foreground data-[side=bottom]:-top-1 data-[side=inline-end]:-start-1 data-[side=inline-start]:-end-1 data-[side=left]:-right-1 data-[side=right]:-left-1 data-[side=top]:-bottom-1 dark:bg-popover',
        className,
      )}
      {...props}
    />
  )
}

type Size = { width: number; height: number }

function sizeOf(element: HTMLElement): Size {
  const css = getComputedStyle(element)
  const width = parseFloat(css.width) || 0
  const height = parseFloat(css.height) || 0
  if (Math.round(width) === element.offsetWidth && Math.round(height) === element.offsetHeight) {
    return { width, height }
  }
  return { width: element.offsetWidth, height: element.offsetHeight }
}

function setSize(element: HTMLElement, name: 'popup' | 'positioner', size: Size | 'auto' | 'max-content') {
  element.style.setProperty(`--${name}-width`, typeof size === 'string' ? size : `${size.width}px`)
  element.style.setProperty(`--${name}-height`, typeof size === 'string' ? size : `${size.height}px`)
}

const ANCHOR = ['position', 'top', 'right', 'bottom', 'left'] as const

function useResizeWithCaption(caption: RefObject<HTMLElement | null>, text: string) {
  const settled = useRef<Size | null>(null)
  const moving = useRef(false)

  useLayoutEffect(() => {
    const popup = caption.current?.closest<HTMLElement>('[data-slot=tooltip-content]')
    const positioner = popup?.parentElement
    if (!popup || !positioner || caption.current?.closest('[data-slot=tooltip-viewport]')) return

    const release = () => {
      moving.current = false
      setSize(popup, 'popup', 'auto')
      positioner.style.removeProperty('--positioner-width')
      positioner.style.removeProperty('--positioner-height')
      for (const property of ANCHOR) popup.style.removeProperty(property)
    }

    const from = moving.current ? sizeOf(popup) : settled.current
    release()
    setSize(positioner, 'positioner', 'max-content')
    const to = sizeOf(popup)
    settled.current = to

    if (
      !from ||
      (from.width === to.width && from.height === to.height) ||
      matchMedia('(prefers-reduced-motion: reduce)').matches
    ) {
      release()
      return
    }

    moving.current = true
    setSize(positioner, 'positioner', to)
    setSize(popup, 'popup', from)

    const side = positioner.dataset.side
    const rtl = getComputedStyle(positioner).direction === 'rtl'
    const fromRight = side === 'left' || side === (rtl ? 'inline-end' : 'inline-start')
    popup.style.setProperty('position', 'absolute')
    popup.style.setProperty(side === 'top' ? 'bottom' : 'top', '0')
    popup.style.setProperty(fromRight ? 'right' : 'left', '0')

    let live = true
    const frame = requestAnimationFrame(() => {
      setSize(popup, 'popup', to)
      void Promise.allSettled(popup.getAnimations().map((animation) => animation.finished)).then(() => {
        if (live) release()
      })
    })
    return () => {
      live = false
      cancelAnimationFrame(frame)
    }
  }, [caption, text])
}

export function Caption({ children }: { children: string }) {
  const box = useRef<HTMLSpanElement>(null)
  useResizeWithCaption(box, children)

  if (joinsLetters(children)) {
    return (
      <span ref={box} dir="auto" className="inline-block whitespace-nowrap">
        {children}
      </span>
    )
  }
  return (

    <span ref={box} dir="auto" className="relative inline-block whitespace-nowrap">
      <span aria-hidden className="invisible">
        {children}
      </span>
      <span className="absolute top-0 left-1/2 -translate-x-1/2">
        <TextMorph
          duration={200}
          ease="cubic-bezier(0.22, 1, 0.36, 1)"
          numbers={false}
          scale={false}
        >
          {children}
        </TextMorph>
      </span>
    </span>
  )
}

function TooltipViewport({ className, ...props }: TooltipPrimitive.Viewport.Props) {
  return (
    <TooltipPrimitive.Viewport
      data-slot="tooltip-viewport"
      className={cn(
        'relative -mx-[var(--tooltip-inset)] flex h-full w-[calc(100%+2*var(--tooltip-inset))] items-center justify-center overflow-clip px-[var(--tooltip-inset)]',
        '[&_[data-current]]:w-max [&_[data-previous]]:inset-0 [&_[data-previous]]:flex [&_[data-previous]]:items-center [&_[data-previous]]:justify-center',
        '[&_[data-current]]:whitespace-nowrap [&_[data-previous]]:whitespace-nowrap',
        '[&_[data-current]]:transition-[translate,opacity] [&_[data-previous]]:transition-[translate,opacity]',
        '[&_[data-current]]:duration-200 [&_[data-previous]]:duration-200',
        '[&_[data-current]]:ease-[cubic-bezier(0.22,1,0.36,1)] [&_[data-previous]]:ease-[cubic-bezier(0.22,1,0.36,1)]',

        "data-[activation-direction~='left']:[&_[data-current][data-starting-style]]:-translate-x-2 data-[activation-direction~='left']:[&_[data-current][data-starting-style]]:opacity-0",
        "data-[activation-direction~='right']:[&_[data-current][data-starting-style]]:translate-x-2 data-[activation-direction~='right']:[&_[data-current][data-starting-style]]:opacity-0",
        "data-[activation-direction~='left']:[&_[data-previous][data-ending-style]]:translate-x-2 data-[activation-direction~='left']:[&_[data-previous][data-ending-style]]:opacity-0",
        "data-[activation-direction~='right']:[&_[data-previous][data-ending-style]]:-translate-x-2 data-[activation-direction~='right']:[&_[data-previous][data-ending-style]]:opacity-0",
        "data-[activation-direction~='down']:[&_[data-current][data-starting-style]]:translate-y-[70%] data-[activation-direction~='down']:[&_[data-current][data-starting-style]]:opacity-0",
        "data-[activation-direction~='up']:[&_[data-current][data-starting-style]]:-translate-y-[70%] data-[activation-direction~='up']:[&_[data-current][data-starting-style]]:opacity-0",
        "data-[activation-direction~='down']:[&_[data-previous][data-ending-style]]:-translate-y-[70%] data-[activation-direction~='down']:[&_[data-previous][data-ending-style]]:opacity-0",
        "data-[activation-direction~='up']:[&_[data-previous][data-ending-style]]:translate-y-[70%] data-[activation-direction~='up']:[&_[data-previous][data-ending-style]]:opacity-0",
        '[[data-instant]_&_[data-current]]:transition-none [[data-instant]_&_[data-previous]]:transition-none',

        'motion-reduce:[&_[data-current]]:transition-none motion-reduce:[&_[data-previous]]:transition-none',
        className,
      )}
      {...props}
    />
  )
}

export {
  createTooltipHandle,
  Tooltip,
  TooltipArrow,
  TooltipTrigger,
  TooltipContent,
  TooltipProvider,
  TooltipViewport,
}
