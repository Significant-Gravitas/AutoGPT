import * as React from 'react'

import { cn } from '@/lib/utils'

function Textarea({ className, ...props }: React.ComponentProps<'textarea'>) {
  return (
    <textarea
      data-slot="textarea"
      dir="auto"
      className={cn(
        'flex field-sizing-content min-h-16 w-full rounded-lg border focus-field border-input bg-transparent px-2.5 py-2 text-base placeholder:text-muted-foreground disabled:cursor-not-allowed disabled:bg-input/50 disabled:opacity-50 aria-invalid:border-destructive aria-invalid:ring-3 aria-invalid:ring-destructive/20 md:text-sm dark:bg-input/30 dark:disabled:bg-input/80 dark:aria-invalid:border-destructive/50 dark:aria-invalid:ring-destructive/40 [&[dir=auto]]:placeholder-shown:[direction:inherit]',
        className,
      )}
      {...props}
    />
  )
}

export { Textarea }
