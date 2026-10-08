"use client";
import { IconChevronLeft, IconChevronRight, IconDots } from '@tabler/icons-react'
import * as React from 'react'

import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'

function Pagination({ className, ...props }: React.ComponentProps<'nav'>) {
  return (
    <nav
      role="navigation"
      aria-label="pagination"
      data-slot="pagination"
      className={cn('mx-auto flex w-full justify-center', className)}
      {...props}
    />
  )
}

type PaginationHighlight = { x: number; width: number; y: number; height: number }

const PaginationHighlightContext = React.createContext<PaginationHighlight | null>(null)

const sameCell = (a: PaginationHighlight | null, b: PaginationHighlight | null) =>
  a === b ||
  (!!a && !!b && a.x === b.x && a.width === b.width && a.y === b.y && a.height === b.height)

function PaginationContent({ className, children, ...props }: React.ComponentProps<'ul'>) {
  const list = React.useRef<HTMLUListElement>(null)
  const [highlight, setHighlight] = React.useState<PaginationHighlight | null>(null)

  const measure = React.useCallback(() => {
    const cell = list.current?.querySelector<HTMLElement>('[data-active=true]')
    const next = cell
      ? {
          x: cell.offsetLeft,
          width: cell.offsetWidth,
          y: cell.offsetTop,
          height: cell.offsetHeight,
        }
      : null

    setHighlight((current) => (sameCell(current, next) ? current : next))
  }, [])

  React.useLayoutEffect(measure)

  React.useEffect(() => {
    let canceled = false
    void document.fonts.ready.then(() => {
      if (!canceled) measure()
    })

    return () => {
      canceled = true
    }
  }, [measure])

  return (
    <PaginationHighlightContext.Provider value={highlight}>
      <ul
        ref={list}
        data-slot="pagination-content"
        className={cn('relative flex items-center gap-0.5', className)}
        {...props}
      >

        {highlight ? (
          <span
            aria-hidden
            data-slot="pagination-highlight"
            className="sliding-tab-pill pointer-events-none absolute left-0 rounded-lg border border-border dark:border-input"
            style={{
              transform: `translateX(${highlight.x}px)`,
              width: highlight.width,
              top: highlight.y,
              height: highlight.height,
            }}
          />
        ) : null}
        {children}
      </ul>
    </PaginationHighlightContext.Provider>
  )
}

function PaginationItem({ ...props }: React.ComponentProps<'li'>) {
  return <li data-slot="pagination-item" {...props} />
}

type PaginationLinkProps = {
  isActive?: boolean
} & Pick<React.ComponentProps<typeof Button>, 'size'> &
  React.ComponentProps<'a'>

function PaginationLink({
  className,
  isActive,
  size = 'icon',
  children,
  ...props
}: PaginationLinkProps) {
  const highlight = React.useContext(PaginationHighlightContext)
  const cell = React.useRef<HTMLAnchorElement>(null)
  const label = React.useRef<HTMLSpanElement>(null)

  const numbered = isActive !== undefined

  const reveal = () => {
    if (!highlight || !cell.current || !label.current) return undefined

    const left = cell.current.offsetLeft + label.current.offsetLeft
    const right = left + label.current.offsetWidth

    return `inset(0 ${right - (highlight.x + highlight.width)}px 0 ${highlight.x - left}px)`
  }

  return (
    <Button

      variant={isActive && highlight === null ? 'outline' : 'ghost'}
      size={size}
      className={cn(
        'aria-disabled:pointer-events-none aria-disabled:opacity-50',

        numbered && 'text-muted-foreground aria-[current=page]:hover:before:bg-transparent!',
        className,
      )}
      nativeButton={false}
      render={
        <a
          ref={cell}
          aria-current={isActive ? 'page' : undefined}
          data-slot="pagination-link"
          data-active={isActive}
          {...props}
        />
      }
    >
      {numbered ? (
        <span ref={label} className="relative grid">
          <span className="col-start-1 row-start-1">{children}</span>
          <span
            aria-hidden
            className="sliding-tab-active-label pointer-events-none col-start-1 row-start-1 text-foreground"
            style={{ clipPath: reveal() }}
          >
            {children}
          </span>
        </span>
      ) : (
        children
      )}
    </Button>
  )
}

function PaginationPrevious({
  className,
  text = 'Previous',
  ...props
}: React.ComponentProps<typeof PaginationLink> & { text?: string }) {
  return (
    <PaginationLink
      aria-label="Go to previous page"
      size="default"
      className={cn('ps-1.5!', className)}
      {...props}
    >
      <IconChevronLeft data-icon="inline-start" className="rtl:-scale-x-100" />
      <span className="hidden sm:block">{text}</span>
    </PaginationLink>
  )
}

function PaginationNext({
  className,
  text = 'Next',
  ...props
}: React.ComponentProps<typeof PaginationLink> & { text?: string }) {
  return (
    <PaginationLink
      aria-label="Go to next page"
      size="default"
      className={cn('pe-1.5!', className)}
      {...props}
    >
      <span className="hidden sm:block">{text}</span>
      <IconChevronRight data-icon="inline-end" className="rtl:-scale-x-100" />
    </PaginationLink>
  )
}

function PaginationEllipsis({ className, ...props }: React.ComponentProps<'span'>) {
  return (
    <span
      aria-hidden
      data-slot="pagination-ellipsis"
      className={cn(
        "flex size-8 items-center justify-center [&_svg:not([class*='size-'])]:size-4",
        className,
      )}
      {...props}
    >
      <IconDots />
      <span className="sr-only">More pages</span>
    </span>
  )
}

export {
  Pagination,
  PaginationContent,
  PaginationEllipsis,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
}
