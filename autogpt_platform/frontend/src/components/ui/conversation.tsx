import * as React from 'react'
import { mergeProps } from '@base-ui/react/merge-props'
import { useRender } from '@base-ui/react/use-render'
import { cva, type VariantProps } from 'class-variance-authority'

import { cn } from '@/lib/utils'

function Conversation({ className, ...props }: React.ComponentProps<'div'>) {
  return (
    <div
      data-slot="conversation"
      className={cn('flex min-w-0 flex-col gap-2', className)}
      {...props}
    />
  )
}

const bubbleVariants = cva(
  'group/bubble relative flex w-fit max-w-[80%] min-w-0 flex-col gap-1 group-data-[align=end]/message:self-end data-[align=end]:self-end data-[variant=ghost]:max-w-full',
  {
    variants: {
      variant: {
        default:
          '*:data-[slot=conversation-content]:bg-bubble-sent *:data-[slot=conversation-content]:text-bubble-sent-foreground [&>[data-slot=conversation-content]:is(button,a):hover]:bg-bubble-sent/80',
        secondary:
          '*:data-[slot=conversation-content]:bg-secondary *:data-[slot=conversation-content]:text-secondary-foreground [&>[data-slot=conversation-content]:is(button,a):hover]:bg-[color-mix(in_oklch,var(--secondary),var(--foreground)_5%)]',
        muted:
          '*:data-[slot=conversation-content]:bg-bubble-received *:data-[slot=conversation-content]:text-bubble-received-foreground [&>[data-slot=conversation-content]:is(button,a):hover]:bg-[color-mix(in_oklch,var(--bubble-received),var(--foreground)_5%)]',
        tinted:
          '*:data-[slot=conversation-content]:bg-[oklch(from_var(--primary)_0.93_calc(c*0.4)_h)] *:data-[slot=conversation-content]:text-foreground dark:*:data-[slot=conversation-content]:bg-[oklch(from_var(--primary)_0.3_calc(c*0.4)_h)] [&>[data-slot=conversation-content]:is(button,a):hover]:bg-[oklch(from_var(--primary)_0.88_calc(c*0.5)_h)] dark:[&>[data-slot=conversation-content]:is(button,a):hover]:bg-[oklch(from_var(--primary)_0.35_calc(c*0.5)_h)]',
        outline:
          '*:data-[slot=conversation-content]:border-border *:data-[slot=conversation-content]:bg-background [&>[data-slot=conversation-content]:is(button,a):hover]:bg-muted [&>[data-slot=conversation-content]:is(button,a):hover]:text-foreground dark:[&>[data-slot=conversation-content]:is(button,a):hover]:bg-input/30',
        ghost:
          'border-none *:data-[slot=conversation-content]:rounded-none *:data-[slot=conversation-content]:bg-transparent *:data-[slot=conversation-content]:p-0 [&>[data-slot=conversation-content]:is(button,a):hover]:bg-muted [&>[data-slot=conversation-content]:is(button,a):hover]:text-foreground dark:[&>[data-slot=conversation-content]:is(button,a):hover]:bg-muted/50',
        destructive:
          '*:data-[slot=conversation-content]:bg-destructive/10 *:data-[slot=conversation-content]:text-destructive dark:*:data-[slot=conversation-content]:bg-destructive/20 [&>[data-slot=conversation-content]:is(button,a):hover]:bg-destructive/20 dark:[&>[data-slot=conversation-content]:is(button,a):hover]:bg-destructive/30',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  },
)

function ConversationBubble({
  variant = 'default',
  align = 'start',
  sentAt,
  className,
  children,
  onClick,
  ...props
}: React.ComponentProps<'div'> &
  VariantProps<typeof bubbleVariants> & {
    align?: 'start' | 'end'
    sentAt?: React.ReactNode
  }) {
  const [revealed, setRevealed] = React.useState(false)

  return (
    <div
      data-slot="conversation-bubble"
      data-variant={variant}
      data-align={align}
      data-revealed={sentAt && revealed ? '' : undefined}
      className={cn(bubbleVariants({ variant }), sentAt ? 'cursor-pointer' : undefined, className)}
      onClick={(event) => {
        onClick?.(event)
        if (!sentAt) return

        if ((event.target as Element).closest('a,button,[role="button"]')) return
        setRevealed((open) => !open)
      }}
      {...props}
    >
      {children}
      {sentAt ? (

        <div
          data-slot="conversation-time"
          className="-mt-1 grid grid-rows-[0fr] transition-[grid-template-rows] duration-200 ease-out group-data-revealed/bubble:grid-rows-[1fr] motion-reduce:transition-none"
        >
          <div className="min-h-0 overflow-hidden">
            <time className="block px-3.5 pt-1 text-xs font-medium text-muted-foreground group-data-[align=end]/bubble:text-end">
              {sentAt}
            </time>
          </div>
        </div>
      ) : null}
    </div>
  )
}

function ConversationContent({ className, render, ...props }: useRender.ComponentProps<'div'>) {
  return useRender({
    defaultTagName: 'div',
    props: mergeProps<'div'>(
      {
        className: cn(
          'relative w-fit max-w-full min-w-0 rounded-[18px] border border-transparent px-3.5 py-2 text-sm leading-snug wrap-break-word group-data-[align=end]/bubble:self-end [button]:text-start [button,a]:transition-ring [button,a]:outline-none [button,a]:focus-visible:border-ring [button,a]:focus-visible:ring-3 [button,a]:focus-visible:ring-ring/50',

          '[unicode-bidi:plaintext]',

          'group-data-[align=end]/bubble:rounded-ee-[8px] group-data-[align=start]/bubble:rounded-es-[8px]',

          '[[data-slot=conversation-bubble][data-align=start]+[data-slot=conversation-bubble][data-align=start]>&]:rounded-ss-[8px]',
          '[[data-slot=conversation-bubble][data-align=end]+[data-slot=conversation-bubble][data-align=end]>&]:rounded-se-[8px]',
          className,
        ),
      },
      props,
    ),
    render,
    state: {
      slot: 'conversation-content',
    },
  })
}

const bubbleReactionsVariants = cva(
  'absolute z-10 flex w-fit shrink-0 items-center justify-center gap-1 rounded-full bg-muted px-1.5 py-0.5 text-sm ring-3 ring-card has-[button]:p-0',
  {
    variants: {
      side: {
        top: 'top-0 -translate-y-3/4',
        bottom: 'bottom-0 translate-y-3/4',
      },
      align: {
        start: 'start-3',
        end: 'end-3',
      },
    },
    defaultVariants: {
      side: 'bottom',
      align: 'end',
    },
  },
)

function ConversationReactions({
  side = 'bottom',
  align = 'end',
  className,
  ...props
}: React.ComponentProps<'div'> & {
  align?: 'start' | 'end'
  side?: 'top' | 'bottom'
}) {
  return (
    <div
      data-slot="conversation-reactions"
      data-align={align}
      data-side={side}
      className={cn(bubbleReactionsVariants({ side, align }), className)}
      {...props}
    />
  )
}

export { Conversation, ConversationBubble, ConversationContent, ConversationReactions }
