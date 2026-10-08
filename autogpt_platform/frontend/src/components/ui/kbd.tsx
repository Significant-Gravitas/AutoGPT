import { cn } from '@/lib/utils'

const kbdSizes = {
  sm: "h-4 min-w-4 rounded-sm px-1 text-[0.625rem] [&_svg:not([class*='size-'])]:size-2.5",
  md: "h-5 min-w-5 rounded-sm px-1 text-xs [&_svg:not([class*='size-'])]:size-3",
  lg: "h-6 min-w-6 rounded-md px-1.5 text-[0.8125rem] [&_svg:not([class*='size-'])]:size-3.5",
  xl: "h-7 min-w-7 rounded-lg px-2 text-sm [&_svg:not([class*='size-'])]:size-4",
} as const

type KbdSize = keyof typeof kbdSizes

function Kbd({
  className,
  size = 'md',
  ...props
}: React.ComponentProps<'kbd'> & { size?: KbdSize }) {
  return (
    <kbd
      data-slot="kbd"
      className={cn(
        'pointer-events-none inline-flex w-fit items-center justify-center gap-1 border border-border bg-muted font-sans font-medium text-foreground select-none in-data-[slot=context-menu-shortcut]:border-transparent in-data-[slot=dropdown-menu-item]:border-transparent in-data-[slot=tooltip-content]:border-transparent in-data-[slot=tooltip-content]:bg-background/20 in-data-[slot=tooltip-content]:text-background dark:in-data-[slot=tooltip-content]:bg-white/10 dark:in-data-[slot=tooltip-content]:text-popover-foreground',
        kbdSizes[size],
        className,
      )}
      {...props}
    />
  )
}

function KbdGroup({ className, ...props }: React.ComponentProps<'div'>) {
  return (
    <kbd
      data-slot="kbd-group"
      className={cn('inline-flex items-center gap-1', className)}
      {...props}
    />
  )
}

export { Kbd, KbdGroup, kbdSizes }
export type { KbdSize }
