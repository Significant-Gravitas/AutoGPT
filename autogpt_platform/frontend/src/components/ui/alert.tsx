import { IconAlertCircle, IconCircleCheck, IconInfoCircle } from '@tabler/icons-react'
import * as React from 'react'
import { cva, type VariantProps } from 'class-variance-authority'

import { cn } from '@/lib/utils'

const ALERT_MARKS = {
  error: IconAlertCircle,
  warning: IconAlertCircle,
  success: IconCircleCheck,
  info: IconInfoCircle,
} as const

type AlertTone = keyof typeof ALERT_MARKS

const ALERT_TONES: Record<AlertTone, string> = {
  error: 'bg-error/10 text-error *:data-[slot=alert-mark]:bg-error',
  warning: 'bg-warning/10 text-warning *:data-[slot=alert-mark]:bg-warning',
  success: 'bg-success/10 text-success *:data-[slot=alert-mark]:bg-success',
  info: 'bg-info/10 text-info *:data-[slot=alert-mark]:bg-info',
}

const alertVariants = cva(
  'group/alert relative grid w-full grid-cols-[auto_1fr] gap-x-2.5 gap-y-0 rounded-lg px-2.5 py-2 text-start text-sm has-data-[slot=alert-action]:pe-18',
  {
    variants: { variant: ALERT_TONES },
    defaultVariants: { variant: 'info' },
  },
)

function Alert({
  className,
  variant = 'info',
  children,
  ...props
}: React.ComponentProps<'div'> & VariantProps<typeof alertVariants>) {
  const Mark = ALERT_MARKS[variant ?? 'info']

  return (
    <div
      data-slot="alert"
      role="alert"
      className={cn(alertVariants({ variant }), className)}
      {...props}
    >
      <Mark
        data-slot="alert-mark"
        aria-hidden="true"

        className="row-span-2 size-8 self-center rounded-md p-1.5 text-card"
      />
      {children}
    </div>
  )
}

function AlertTitle({ className, ...props }: React.ComponentProps<'div'>) {
  return (
    <div
      data-slot="alert-title"
      className={cn(
        'col-start-2 font-bold [&_a]:underline [&_a]:underline-offset-3 [&_a]:hover:text-foreground',
        className,
      )}
      {...props}
    />
  )
}

function AlertDescription({ className, ...props }: React.ComponentProps<'div'>) {
  return (
    <div
      data-slot="alert-description"

      className={cn(
        'col-start-2 text-sm text-balance md:text-pretty [&_a]:underline [&_a]:underline-offset-3 [&_a]:hover:text-foreground [&_p:not(:last-child)]:mb-4',
        className,
      )}
      {...props}
    />
  )
}

function AlertAction({ className, ...props }: React.ComponentProps<'div'>) {
  return (
    <div data-slot="alert-action" className={cn('absolute end-2 top-2', className)} {...props} />
  )
}

export { Alert, AlertTitle, AlertDescription, AlertAction, ALERT_MARKS, type AlertTone }
