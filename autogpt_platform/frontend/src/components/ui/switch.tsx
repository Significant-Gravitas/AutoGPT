'use client'

import { type ComponentProps, useState } from 'react'

import { cn } from '@/lib/utils'

type SwitchProps = Omit<ComponentProps<'button'>, 'checked' | 'defaultChecked' | 'onChange'> & {
  checked?: boolean
  defaultChecked?: boolean
  onCheckedChange?: (checked: boolean) => void
}

function Switch({
  className,
  checked,
  defaultChecked = false,
  onCheckedChange,
  onClick,
  ...props
}: SwitchProps) {
  const [uncontrolled, setUncontrolled] = useState(defaultChecked)
  const on = checked ?? uncontrolled

  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      data-slot="switch"
      data-on={on ? 'true' : 'false'}
      className={cn(
        't-toggle',

        'peer group/switch relative inline-flex h-[18.4px] w-[32.66px] shrink-0 items-center rounded-full border border-transparent outline-none after:absolute after:-inset-x-3 after:-inset-y-2 focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 disabled:cursor-not-allowed disabled:opacity-50 data-[on=false]:bg-input data-[on=true]:bg-primary dark:data-[on=false]:bg-input/80',
        className,
      )}
      onClick={(event) => {
        setUncontrolled(!on)
        onCheckedChange?.(!on)
        onClick?.(event)
      }}
      {...props}
    >
      <span
        aria-hidden="true"
        data-slot="switch-thumb"

        className="t-toggle-thumb pointer-events-none ms-px block h-3.5 rounded-full bg-background rtl:-scale-x-100 dark:group-data-[on=false]/switch:bg-foreground dark:group-data-[on=true]/switch:bg-primary-foreground"
      />
    </button>
  )
}

export { Switch }
