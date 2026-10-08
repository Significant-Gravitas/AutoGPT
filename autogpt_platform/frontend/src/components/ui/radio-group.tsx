'use client'

import * as React from 'react'
import { Radio as RadioPrimitive } from '@base-ui/react/radio'
import { RadioGroup as RadioGroupPrimitive } from '@base-ui/react/radio-group'
import { arc, LayoutGroup, motion, useReducedMotion } from 'motion/react'

import { cn } from '@/lib/utils'

const TRAVEL = { type: 'spring', duration: 0.3, bounce: 0.18 } as const

const ARCH = arc({ strength: 0.6 })

const RadioAnimated = React.createContext(true)

function RadioGroup({
  className,
  animated = true,
  ...props
}: RadioGroupPrimitive.Props & { animated?: boolean }) {
  const group = React.useId()

  return (
    <RadioAnimated.Provider value={animated}>
      <LayoutGroup id={group}>
        <RadioGroupPrimitive
          data-slot="radio-group"
          className={cn('grid w-full gap-2', className)}
          {...props}
        />
      </LayoutGroup>
    </RadioAnimated.Provider>
  )
}

function RadioGroupItem({ className, ...props }: RadioPrimitive.Root.Props) {
  const reduceMotion = useReducedMotion()
  const animated = React.useContext(RadioAnimated)

  return (
    <RadioPrimitive.Root
      data-slot="radio-group-item"
      className={cn(
        'group/radio-group-item peer relative flex aspect-square size-4 shrink-0 rounded-full border border-input transition-ring outline-none group-has-[:focus-visible]/field-label:ring-0 after:absolute after:-inset-x-3 after:-inset-y-2 focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 aria-disabled:cursor-not-allowed aria-disabled:opacity-50 aria-invalid:border-destructive aria-invalid:ring-3 aria-invalid:ring-destructive/20 dark:bg-input/30 dark:aria-invalid:border-destructive/50 dark:aria-invalid:ring-destructive/40',
        className,
      )}
      {...props}
    >

      <RadioPrimitive.Indicator
        data-slot="radio-group-indicator"
        render={
          animated ? (
            <motion.span
              layoutId="radio-group-indicator"

              initial={false}
              transition={{ layout: reduceMotion ? { duration: 0 } : { ...TRAVEL, path: ARCH } }}
            />
          ) : (
            <span />
          )
        }
        className="pointer-events-none absolute inset-[3px] z-10 rounded-full bg-primary"
      />
    </RadioPrimitive.Root>
  )
}

export { RadioGroup, RadioGroupItem }
