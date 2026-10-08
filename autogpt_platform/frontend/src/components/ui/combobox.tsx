'use client'

import { IconCheck, IconChevronDown, IconSearchOff, IconX } from '@tabler/icons-react'
import * as React from 'react'
import { Combobox as ComboboxPrimitive } from '@base-ui/react'
import { AnimatePresence, motion, MotionConfig } from 'motion/react'

import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupInput,
} from '@/components/ui/input-group'

const ComboboxMultipleContext = React.createContext(false)

function Combobox<Value, Multiple extends boolean | undefined = false>({
  multiple,
  ...props
}: ComboboxPrimitive.Root.Props<Value, Multiple>) {
  return (
    <ComboboxMultipleContext.Provider value={multiple === true}>
      <ComboboxPrimitive.Root multiple={multiple} {...props} />
    </ComboboxMultipleContext.Provider>
  )
}

function ComboboxValue({ ...props }: ComboboxPrimitive.Value.Props) {
  return <ComboboxPrimitive.Value data-slot="combobox-value" {...props} />
}

function ComboboxTrigger({ className, children, ...props }: ComboboxPrimitive.Trigger.Props) {
  return (
    <ComboboxPrimitive.Trigger
      data-slot="combobox-trigger"
      className={cn("[&_svg:not([class*='size-'])]:size-4", className)}
      {...props}
    >
      {children}
      <IconChevronDown className="pointer-events-none size-4 text-muted-foreground" />
    </ComboboxPrimitive.Trigger>
  )
}

function ComboboxClear({ className, ...props }: ComboboxPrimitive.Clear.Props) {
  return (
    <ComboboxPrimitive.Clear
      data-slot="combobox-clear"
      render={<InputGroupButton variant="ghost" size="icon-xs" />}
      className={cn(
        'transition-[opacity,scale] duration-150 ease-out data-starting-style:scale-75 data-starting-style:opacity-0',
        'data-ending-style:scale-75 data-ending-style:opacity-0 data-ending-style:ease-in',
        'motion-reduce:transition-none',
        className,
      )}
      {...props}
    >
      <IconX className="pointer-events-none" />
    </ComboboxPrimitive.Clear>
  )
}

type Erasing = { text: string; style: React.CSSProperties }

const ERASE_MS = 130

function ComboboxInput({
  className,
  children,
  disabled = false,
  showTrigger = true,
  showClear = false,
  ...props
}: ComboboxPrimitive.Input.Props & {
  showTrigger?: boolean
  showClear?: boolean
}) {
  const [erasing, setErasing] = React.useState<Erasing | null>(null)

  const erase = (event: React.MouseEvent<HTMLElement>) => {
    const group = event.currentTarget.closest('[data-slot="input-group"]')
    const field = group?.querySelector('input')
    if (!group || !field?.value) return
    const from = field.getBoundingClientRect()
    const to = group.getBoundingClientRect()
    const type = getComputedStyle(field)

    const edge = getComputedStyle(group)

    const rtl = type.direction === 'rtl'
    setErasing({
      text: field.value,
      style: {
        ...(rtl
          ? {
              right:
                to.right -
                from.right -
                Number.parseFloat(edge.borderRightWidth) +
                Number.parseFloat(type.paddingRight),
            }
          : {
              left:
                from.left -
                to.left -
                Number.parseFloat(edge.borderLeftWidth) +
                Number.parseFloat(type.paddingLeft),
            }),
        top: from.top - to.top - Number.parseFloat(edge.borderTopWidth),
        height: from.height,
        font: type.font,
        letterSpacing: type.letterSpacing,
        color: type.color,
        direction: rtl ? 'rtl' : 'ltr',
        transformOrigin: rtl ? 'right' : 'left',
      },
    })
  }

  return (
    <InputGroup className={cn('w-auto', className)}>
      <ComboboxPrimitive.Input
        render={
          <InputGroupInput
            disabled={disabled}

            className={erasing ? 'combobox-erasing' : undefined}
          />
        }
        {...props}
      />
      <InputGroupAddon align="inline-end">
        {showTrigger && (
          <InputGroupButton
            size="icon-xs"
            variant="ghost"
            render={<ComboboxTrigger />}
            data-slot="input-group-button"
            className="group-has-data-[slot=combobox-clear]/input-group:hidden data-pressed:bg-transparent"
            disabled={disabled}
          />
        )}
        {showClear && <ComboboxClear disabled={disabled} onClick={erase} />}
      </InputGroupAddon>
      {erasing && (

        <MotionConfig reducedMotion="user">
          <motion.span

            key={erasing.text}
            aria-hidden
            initial={{ opacity: 1, scale: 1 }}
            animate={{ opacity: 0, scale: 0.96 }}
            transition={{ duration: ERASE_MS / 1000, ease: [0.4, 0, 1, 1] }}
            onAnimationComplete={() => setErasing(null)}
            style={erasing.style}
            className="pointer-events-none absolute flex items-center whitespace-nowrap"
          >
            {erasing.text}
          </motion.span>
        </MotionConfig>
      )}
      {children}
    </InputGroup>
  )
}

function ComboboxContent({
  className,
  side = 'bottom',
  sideOffset = 6,
  align = 'start',
  alignOffset = 0,
  anchor,
  ...props
}: ComboboxPrimitive.Popup.Props &
  Pick<
    ComboboxPrimitive.Positioner.Props,
    'side' | 'align' | 'sideOffset' | 'alignOffset' | 'anchor'
  >) {
  return (
    <ComboboxPrimitive.Portal>
      <ComboboxPrimitive.Positioner
        side={side}
        sideOffset={sideOffset}
        align={align}
        alignOffset={alignOffset}
        anchor={anchor}
        className="isolate z-50"
      >
        <ComboboxPrimitive.Popup
          data-slot="combobox-content"
          data-chips={!!anchor}
          className={cn(
            'group/combobox-content relative max-h-(--available-height) w-(--anchor-width) max-w-(--available-width) min-w-[calc(var(--anchor-width)+--spacing(7))] origin-(--transform-origin) overflow-hidden rounded-lg bg-popover text-popover-foreground shadow-md ring-1 ring-foreground/10 duration-150 ease-out data-closed:animate-out data-closed:fade-out-0 data-closed:zoom-out-95 data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95 data-[chips=true]:min-w-(--anchor-width) *:data-[slot=input-group]:m-1 *:data-[slot=input-group]:mb-0 *:data-[slot=input-group]:h-8 *:data-[slot=input-group]:border-input/30 *:data-[slot=input-group]:bg-input/30 *:data-[slot=input-group]:shadow-none',
            className,
          )}
          {...props}
        />
      </ComboboxPrimitive.Positioner>
    </ComboboxPrimitive.Portal>
  )
}

function ComboboxList({ className, ...props }: ComboboxPrimitive.List.Props) {
  return (
    <ComboboxPrimitive.List
      data-slot="combobox-list"
      className={cn(

        'max-h-[min(calc(--spacing(72)---spacing(9)),calc(var(--available-height)---spacing(9)))] scroll-py-1 overflow-x-hidden overflow-y-auto overscroll-contain p-1 data-empty:p-0',
        className,
      )}
      {...props}
    />
  )
}

function ComboboxItem({ className, children, ...props }: ComboboxPrimitive.Item.Props) {
  const multiple = React.useContext(ComboboxMultipleContext)

  return (
    <ComboboxPrimitive.Item
      data-slot="combobox-item"
      className={cn(
        "group/combobox-item relative flex w-full cursor-default items-center gap-2 rounded-md py-1 ps-1.5 text-sm outline-hidden select-none data-disabled:pointer-events-none data-disabled:opacity-50 data-highlighted:bg-accent data-highlighted:text-accent-foreground not-data-[variant=destructive]:data-highlighted:**:text-accent-foreground [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",

        multiple ? 'pe-2' : 'pe-8',
        className,
      )}
      {...props}
    >
      {multiple ? (

        <span
          aria-hidden

          className="flex size-4 shrink-0 items-center justify-center rounded-[4px] border border-input transition-colors group-data-selected/combobox-item:border-primary group-data-selected/combobox-item:bg-primary dark:bg-input/30 dark:group-data-selected/combobox-item:bg-primary"
        >
          <ComboboxPrimitive.ItemIndicator>

            <IconCheck className="size-3 stroke-primary-foreground" />
          </ComboboxPrimitive.ItemIndicator>
        </span>
      ) : null}
      {children}
      {multiple ? null : (
        <ComboboxPrimitive.ItemIndicator
          render={
            <span className="pointer-events-none absolute end-2 flex size-4 items-center justify-center" />
          }
        >
          <IconCheck className="pointer-events-none" />
        </ComboboxPrimitive.ItemIndicator>
      )}
    </ComboboxPrimitive.Item>
  )
}

function ComboboxGroup({ className, ...props }: ComboboxPrimitive.Group.Props) {
  return <ComboboxPrimitive.Group data-slot="combobox-group" className={cn(className)} {...props} />
}

function ComboboxLabel({ className, ...props }: ComboboxPrimitive.GroupLabel.Props) {
  return (
    <ComboboxPrimitive.GroupLabel
      data-slot="combobox-label"
      className={cn('px-2 py-1.5 text-xs text-muted-foreground', className)}
      {...props}
    />
  )
}

function ComboboxCollection({ ...props }: ComboboxPrimitive.Collection.Props) {
  return <ComboboxPrimitive.Collection data-slot="combobox-collection" {...props} />
}

function ComboboxEmpty({ className, children, ...props }: ComboboxPrimitive.Empty.Props) {
  return (
    <ComboboxPrimitive.Empty
      data-slot="combobox-empty"
      className={cn(
        'hidden w-full flex-col items-center justify-center gap-2 px-4 py-6 text-center text-sm text-muted-foreground group-data-empty/combobox-content:flex',
        className,
      )}
      {...props}
    >
      <IconSearchOff aria-hidden className="size-5 text-muted-foreground/60" />
      {children}
    </ComboboxPrimitive.Empty>
  )
}

function ComboboxSeparator({ className, ...props }: ComboboxPrimitive.Separator.Props) {
  return (
    <ComboboxPrimitive.Separator
      data-slot="combobox-separator"
      className={cn('-mx-1 my-1 h-px bg-border', className)}
      {...props}
    />
  )
}

function ComboboxChips({
  className,
  children,
  ...props
}: React.ComponentPropsWithRef<typeof ComboboxPrimitive.Chips> & ComboboxPrimitive.Chips.Props) {
  return (
    <ComboboxPrimitive.Chips
      data-slot="combobox-chips"
      className={cn(
        'flex min-h-8 flex-wrap items-center gap-1 rounded-lg border focus-field border-input bg-transparent bg-clip-padding px-2.5 py-1 text-sm has-aria-invalid:border-destructive has-aria-invalid:ring-3 has-aria-invalid:ring-destructive/20 has-data-[slot=combobox-chip]:px-1 dark:bg-input/30 dark:has-aria-invalid:border-destructive/50 dark:has-aria-invalid:ring-destructive/40',
        className,
      )}
      {...props}
    >
      <MotionConfig reducedMotion="user">
        <AnimatePresence initial={false}>{children}</AnimatePresence>
      </MotionConfig>
    </ComboboxPrimitive.Chips>
  )
}

function ComboboxChip({
  className,
  children,
  showRemove = true,
  ...props
}: ComboboxPrimitive.Chip.Props & {
  showRemove?: boolean
}) {
  return (
    <ComboboxPrimitive.Chip
      data-slot="combobox-chip"
      render={

        <motion.div
          layout
          initial={{ opacity: 0, scale: 0.94 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.8, transition: { duration: 0.14, ease: [0.4, 0, 1, 1] } }}
          transition={{ duration: 0.18, ease: [0.23, 1, 0.32, 1] }}
        />
      }
      className={cn(
        'origin-left rtl:origin-right',

        'isolate flex h-[calc(--spacing(5.25))] w-fit items-center justify-center gap-1 rounded-full bg-foreground/15 px-2 text-xs font-medium whitespace-nowrap text-foreground has-disabled:pointer-events-none has-disabled:cursor-not-allowed has-disabled:opacity-50 has-data-[slot=combobox-chip-remove]:pe-0',
        className,
      )}
      {...props}
    >
      {children}
      {showRemove && (
        <ComboboxPrimitive.ChipRemove
          render={<Button variant="ghost" size="icon-xs" />}

          className="isolation-auto -ms-1 size-[calc(--spacing(5.25))] rounded-full opacity-50 before:-start-2.5 before:[mask-image:linear-gradient(to_right,transparent,black_62%)] hover:opacity-100 rtl:before:[mask-image:linear-gradient(to_left,transparent,black_62%)]"
          data-slot="combobox-chip-remove"
        >
          <IconX className="pointer-events-none" />
        </ComboboxPrimitive.ChipRemove>
      )}
    </ComboboxPrimitive.Chip>
  )
}

function ComboboxChipsInput({ className, ...props }: ComboboxPrimitive.Input.Props) {
  return (
    <ComboboxPrimitive.Input
      data-slot="combobox-chip-input"
      className={cn('min-w-16 flex-1 outline-none', className)}
      {...props}
    />
  )
}

function useComboboxAnchor() {
  return React.useRef<HTMLDivElement | null>(null)
}

export {
  Combobox,
  ComboboxInput,
  ComboboxContent,
  ComboboxList,
  ComboboxItem,
  ComboboxGroup,
  ComboboxLabel,
  ComboboxCollection,
  ComboboxEmpty,
  ComboboxSeparator,
  ComboboxChips,
  ComboboxChip,
  ComboboxChipsInput,
  ComboboxTrigger,
  ComboboxValue,
  useComboboxAnchor,
}
