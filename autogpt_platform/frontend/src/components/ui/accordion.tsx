"use client";
import { IconPlus } from '@tabler/icons-react'
import * as React from 'react'
import { Accordion as AccordionPrimitive } from '@base-ui/react/accordion'

import { cn } from '@/lib/utils'

type AccordionVariant = 'primary' | 'secondary'

const AccordionVariantContext = React.createContext<AccordionVariant>('primary')

function Accordion({
  className,
  variant = 'primary',
  ...props
}: AccordionPrimitive.Root.Props & { variant?: AccordionVariant }) {
  return (
    <AccordionVariantContext value={variant}>
      <AccordionPrimitive.Root
        data-slot="accordion"
        data-variant={variant}
        className={cn('flex w-full flex-col', variant === 'secondary' && 'gap-3', className)}
        {...props}
      />
    </AccordionVariantContext>
  )
}

function AccordionItem({ className, ...props }: AccordionPrimitive.Item.Props) {
  const variant = React.useContext(AccordionVariantContext)

  return (
    <AccordionPrimitive.Item
      data-slot="accordion-item"
      className={cn(
        'overflow-hidden',
        variant === 'primary'
          ?
            'border-t border-foreground/15 first:border-t-0'
          : 'rounded-2xl border bg-card shadow-sm',
        className,
      )}
      {...props}
    />
  )
}

function AccordionTrigger({ className, children, ...props }: AccordionPrimitive.Trigger.Props) {
  const variant = React.useContext(AccordionVariantContext)

  return (
    <AccordionPrimitive.Header className="flex">
      <AccordionPrimitive.Trigger
        data-slot="accordion-trigger"
        className={cn(
          'group/accordion-trigger relative flex w-full flex-1 items-center justify-between gap-4 text-start transition-ring outline-none focus-visible:ring-3 focus-visible:ring-ring/50 aria-disabled:pointer-events-none aria-disabled:opacity-50',

          variant === 'primary'
            ? 'py-5 text-base leading-6 font-semibold'
            : 'px-5 py-4 text-sm font-semibold',
          className,
        )}
        {...props}
      >
        {children}
        {variant === 'primary' ? <AccordionMark /> : <AccordionPlus />}
      </AccordionPrimitive.Trigger>
    </AccordionPrimitive.Header>
  )
}

const MARK_TRAVEL =
  'transition-[d] duration-300 ease-[cubic-bezier(0.16,1,0.3,1)] motion-reduce:transition-none'

function AccordionMark() {
  return (
    <svg
      aria-hidden
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      className="pointer-events-none size-[1.125rem] shrink-0 text-foreground/50 transition-colors duration-200 group-hover/accordion-trigger:text-foreground/80"
    >
      <path
        d="M 6 9 L 12 15"
        className={cn(
          MARK_TRAVEL,
          "group-aria-expanded/accordion-trigger:[d:path('M_6_6_L_18_18')]",
        )}
      />
      <path
        d="M 18 9 L 12 15"
        className={cn(
          MARK_TRAVEL,
          "group-aria-expanded/accordion-trigger:[d:path('M_18_6_L_6_18')]",
        )}
      />
    </svg>
  )
}

function AccordionPlus() {
  return (
    <IconPlus
      aria-hidden
      className="pointer-events-none size-5 shrink-0 text-muted-foreground transition-[rotate,color] duration-300 ease-[cubic-bezier(0.16,1,0.3,1)] group-hover/accordion-trigger:text-foreground group-aria-expanded/accordion-trigger:rotate-45 group-aria-expanded/accordion-trigger:text-foreground"
    />
  )
}

function AccordionContent({ className, children, ...props }: AccordionPrimitive.Panel.Props) {
  const variant = React.useContext(AccordionVariantContext)

  return (
    <AccordionPrimitive.Panel
      data-slot="accordion-content"

      className="h-(--accordion-panel-height) overflow-hidden transition-[height] duration-300 ease-[cubic-bezier(0.16,1,0.3,1)] data-ending-style:h-0 data-starting-style:h-0"
      {...props}
    >
      <div
        className={cn(
          variant === 'primary'
            ? 'pb-5 text-base leading-[1.625] text-foreground/70 transition-opacity duration-300 ease-out in-data-ending-style:opacity-0 in-data-starting-style:opacity-0'
            : 'px-5 pt-0 pb-4 text-[0.9375rem] leading-relaxed text-muted-foreground',
          '[&_a]:underline [&_a]:underline-offset-3 [&_a]:hover:text-foreground [&_p:not(:last-child)]:mb-4',
          className,
        )}
      >
        {children}
      </div>
    </AccordionPrimitive.Panel>
  )
}

export { Accordion, AccordionItem, AccordionTrigger, AccordionContent, type AccordionVariant }
