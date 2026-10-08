"use client";
import { mergeProps } from '@base-ui/react/merge-props'
import { useRender } from '@base-ui/react/use-render'
import { cva, type VariantProps } from 'class-variance-authority'

import { cn } from '@/lib/utils'

const badgeVariants = cva(

  'group/badge t-label relative isolate inline-flex w-fit shrink-0 items-center justify-center gap-1 rounded-full border border-transparent py-0.5 font-medium [--surface-calm:0.5] whitespace-nowrap transition-ring before:pointer-events-none before:absolute before:inset-0 before:-z-10 before:rounded-[inherit] before:border before:border-transparent before:transition-[scale,background-color,border-color] before:duration-150 before:ease-out focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 motion-reduce:before:transition-none has-data-[icon=inline-end]:pe-1.5 has-data-[icon=inline-start]:ps-1.5 aria-invalid:border-destructive aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 [a,button]:active:before:scale-[0.97] motion-reduce:[a,button]:active:before:scale-100 [&>svg]:pointer-events-none [&>svg]:size-3!',
  {
    variants: {
      variant: {
        default: 't-surface t-surface-primary text-primary-foreground',
        secondary: 't-surface t-surface-secondary text-secondary-foreground',

        destructive:
          't-surface t-surface-danger text-destructive focus-visible:ring-destructive/20 dark:focus-visible:ring-destructive/40',
        outline: 't-surface t-surface-outline text-foreground',
        ghost: 'hover:bg-muted hover:text-muted-foreground dark:hover:bg-muted/50',
        link: 'text-primary underline-offset-4 hover:underline',
        blue: 't-surface t-surface-info text-[var(--badge-blue-fg)]',
        green: 't-surface t-surface-success text-[var(--badge-green-fg)]',
        amber: 't-surface t-surface-warning text-[var(--badge-amber-fg)]',
        neutral: 't-surface t-surface-secondary text-muted-foreground',

        tint: 't-surface t-surface-tint',
      },
      size: {
        default: 'h-6 px-2 text-sm',
        sm: 'h-5 px-1.5 text-xs',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  },
)

function Badge({
  className,
  variant = 'default',
  size = 'default',
  render,
  ...props
}: useRender.ComponentProps<'span'> & VariantProps<typeof badgeVariants>) {
  return useRender({
    defaultTagName: 'span',
    props: mergeProps<'span'>(
      {
        className: cn(badgeVariants({ variant, size }), className),
      },
      props,
    ),
    render,
    state: {
      slot: 'badge',
      variant,
    },
  })
}

export { Badge, badgeVariants }
