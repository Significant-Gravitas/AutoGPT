import * as React from 'react'
import { Input as InputPrimitive } from '@base-ui/react/input'

import { cn } from '@/lib/utils'

const AUTO_DIRECTION = new Set(['text', 'search', 'email', 'url'])

const FOLLOWS_PAGE_WHEN_EMPTY = new Set(['text', 'search'])

function Input({ className, type, ...props }: React.ComponentProps<'input'>) {
  const kind = type ?? 'text'
  return (
    <InputPrimitive
      type={type}
      data-slot="input"
      dir={AUTO_DIRECTION.has(kind) ? 'auto' : undefined}
      className={cn(
        'h-8 w-full min-w-0 rounded-lg border focus-field border-input bg-transparent px-2.5 py-1 text-base file:inline-flex file:h-6 file:border-0 file:bg-transparent file:text-sm file:font-medium file:text-foreground placeholder:text-muted-foreground disabled:pointer-events-none disabled:cursor-not-allowed disabled:bg-input/50 disabled:opacity-50 aria-invalid:border-destructive aria-invalid:ring-3 aria-invalid:ring-destructive/20 md:text-sm dark:bg-input/30 dark:disabled:bg-input/80 dark:aria-invalid:border-destructive/50 dark:aria-invalid:ring-destructive/40',
        FOLLOWS_PAGE_WHEN_EMPTY.has(kind) && '[&[dir=auto]]:placeholder-shown:[direction:inherit]',
        className,
      )}
      {...props}
    />
  )
}

export { Input }
