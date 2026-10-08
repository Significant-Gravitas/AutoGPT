"use client";
import { IconCheck, IconChevronRight } from '@tabler/icons-react'
import * as React from 'react'
import { useDirection } from '@base-ui/react/direction-provider'
import { Menu as MenuPrimitive } from '@base-ui/react/menu'

import { cn } from '@/lib/utils'

type DropdownMenuSize = 'sm' | 'default'

const SIZES = {

  sm: {
    panel: 'min-w-36 rounded-lg p-1',
    row: 'gap-1.5 rounded-md px-1.5 py-1 text-sm',
    check: 'gap-1.5 rounded-md py-1 ps-1.5 pe-8 text-sm',
    mark: 'end-2',
    inset: 'data-inset:ps-6.5',
    icon: "[&_svg:not([class*='size-'])]:size-3.5",
    label: 'px-1.5 py-1 text-xs',
    separator: '-mx-1 my-1',
    shortcut: 'text-xs',
  },
  default: {
    panel: 'min-w-48 rounded-xl p-1.5',
    row: 'gap-2 rounded-lg px-2.5 py-1.5 text-sm',
    check: 'gap-2 rounded-lg py-1.5 ps-2.5 pe-9 text-sm',
    mark: 'end-2.5',

    inset: 'data-inset:ps-8.5',
    icon: "[&_svg:not([class*='size-'])]:size-4",
    label: 'px-2.5 py-1 text-xs',
    separator: '-mx-1.5 my-1.5',
    shortcut: 'text-xs',
  },
} as const

const SizeContext = React.createContext<DropdownMenuSize>('default')

function useSize() {
  return SIZES[React.useContext(SizeContext)]
}

const ROW =
  'flex cursor-default items-center outline-hidden select-none focus:bg-accent focus:text-accent-foreground [&_svg]:pointer-events-none [&_svg]:shrink-0'

function DropdownMenu({ ...props }: MenuPrimitive.Root.Props) {
  return <MenuPrimitive.Root data-slot="dropdown-menu" {...props} />
}

function DropdownMenuPortal({ ...props }: MenuPrimitive.Portal.Props) {
  return <MenuPrimitive.Portal data-slot="dropdown-menu-portal" {...props} />
}

function DropdownMenuTrigger({ className, onPointerDown, ...props }: MenuPrimitive.Trigger.Props) {
  const [pressed, setPressed] = React.useState(false)

  React.useEffect(() => {
    if (!pressed) return

    const release = () => {
      setPressed(false)
    }
    window.addEventListener('pointerup', release)
    window.addEventListener('pointercancel', release)
    return () => {
      window.removeEventListener('pointerup', release)
      window.removeEventListener('pointercancel', release)
    }
  }, [pressed])

  return (
    <MenuPrimitive.Trigger
      data-slot="dropdown-menu-trigger"
      data-pressed={pressed || undefined}
      className={cn('select-none', className)}
      onPointerDown={(event) => {
        onPointerDown?.(event)
        if (event.button === 0) setPressed(true)
      }}
      {...props}
    />
  )
}

function DropdownMenuContent({
  className,
  align = 'start',
  alignOffset = 0,
  side = 'bottom',
  sideOffset = 6,
  size,
  ...props
}: MenuPrimitive.Popup.Props &
  Pick<MenuPrimitive.Positioner.Props, 'align' | 'alignOffset' | 'side' | 'sideOffset'> & {
    size?: DropdownMenuSize
  }) {
  const inherited = React.useContext(SizeContext)
  const resolved = size ?? inherited

  return (
    <SizeContext.Provider value={resolved}>
      <MenuPrimitive.Portal>
        <MenuPrimitive.Positioner
          className="isolate z-50 outline-none"
          align={align}
          alignOffset={alignOffset}
          side={side}
          sideOffset={sideOffset}
        >
          <MenuPrimitive.Popup
            data-slot="dropdown-menu-content"
            data-size={resolved}

            className={cn(
              'z-50 max-h-(--available-height) w-(--anchor-width) origin-(--transform-origin) overflow-x-hidden overflow-y-auto bg-popover text-popover-foreground shadow-md ring-1 ring-foreground/10 duration-150 ease-[cubic-bezier(0.16,1,0.3,1)] outline-none data-closed:animate-out data-closed:overflow-hidden data-closed:fade-out-0 data-closed:zoom-out-95 data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95',
              SIZES[resolved].panel,
              className,
            )}
            {...props}
          />
        </MenuPrimitive.Positioner>
      </MenuPrimitive.Portal>
    </SizeContext.Provider>
  )
}

function DropdownMenuGroup({ ...props }: MenuPrimitive.Group.Props) {
  return <MenuPrimitive.Group data-slot="dropdown-menu-group" {...props} />
}

function DropdownMenuLabel({
  className,
  inset,
  ...props
}: MenuPrimitive.GroupLabel.Props & {
  inset?: boolean
}) {
  const size = useSize()

  return (
    <MenuPrimitive.GroupLabel
      data-slot="dropdown-menu-label"
      data-inset={inset}
      className={cn('font-medium text-muted-foreground', size.label, size.inset, className)}
      {...props}
    />
  )
}

function DropdownMenuItem({
  className,
  inset,
  variant = 'default',
  highlighted,
  ...props
}: MenuPrimitive.Item.Props & {
  inset?: boolean
  variant?: 'default' | 'destructive'

  highlighted?: boolean
}) {
  const size = useSize()

  return (
    <MenuPrimitive.Item
      data-slot="dropdown-menu-item"
      data-inset={inset}
      data-variant={variant}
      className={cn(
        ROW,
        size.row,
        size.inset,
        size.icon,
        'group/dropdown-menu-item relative not-data-[variant=destructive]:focus:**:text-accent-foreground data-disabled:pointer-events-none data-disabled:opacity-50 data-[variant=destructive]:text-destructive data-[variant=destructive]:focus:bg-destructive/10 data-[variant=destructive]:focus:text-destructive dark:data-[variant=destructive]:focus:bg-destructive/20 data-[variant=destructive]:*:[svg]:text-destructive',
        highlighted && 'bg-accent text-accent-foreground',
        className,
      )}
      {...props}
    />
  )
}

function DropdownMenuSub({ ...props }: MenuPrimitive.SubmenuRoot.Props) {
  return <MenuPrimitive.SubmenuRoot data-slot="dropdown-menu-sub" {...props} />
}

function DropdownMenuSubTrigger({
  className,
  inset,
  children,
  highlighted,
  ...props
}: MenuPrimitive.SubmenuTrigger.Props & {
  inset?: boolean
  highlighted?: boolean
}) {
  const size = useSize()

  return (
    <MenuPrimitive.SubmenuTrigger
      data-slot="dropdown-menu-sub-trigger"
      data-inset={inset}
      className={cn(
        ROW,
        size.row,
        size.inset,
        size.icon,
        'not-data-[variant=destructive]:focus:**:text-accent-foreground',
        highlighted === undefined
          ? 'data-open:bg-accent data-open:text-accent-foreground data-popup-open:bg-accent data-popup-open:text-accent-foreground'
          : highlighted && 'bg-accent text-accent-foreground',
        className,
      )}
      {...props}
    >
      {children}
      <IconChevronRight className="ms-auto rtl:-scale-x-100" />
    </MenuPrimitive.SubmenuTrigger>
  )
}

function DropdownMenuSubContent({
  className,
  align = 'start',
  alignOffset = -5,
  side,
  sideOffset = 1,
  ...props
}: React.ComponentProps<typeof DropdownMenuContent>) {
  const direction = useDirection()

  return (
    <DropdownMenuContent
      data-slot="dropdown-menu-sub-content"
      align={align}
      alignOffset={alignOffset}
      side={side ?? (direction === 'rtl' ? 'left' : 'right')}
      sideOffset={sideOffset}
      className={cn('w-auto min-w-44 shadow-lg', className)}
      {...props}
    />
  )
}

function DropdownMenuCheckboxItem({
  className,
  children,
  checked,
  inset,
  marked = true,
  ...props
}: MenuPrimitive.CheckboxItem.Props & {
  inset?: boolean
  marked?: boolean
}) {
  const size = useSize()

  return (
    <MenuPrimitive.CheckboxItem
      data-slot="dropdown-menu-checkbox-item"
      data-inset={inset}
      className={cn(
        ROW,
        marked ? size.check : size.row,
        size.inset,
        size.icon,
        'relative focus:**:text-accent-foreground data-disabled:pointer-events-none data-disabled:opacity-50',
        className,
      )}
      checked={checked}
      {...props}
    >
      {marked ? (
        <span className={cn('pointer-events-none absolute', size.mark)}>
          <MenuPrimitive.CheckboxItemIndicator>
            <IconCheck />
          </MenuPrimitive.CheckboxItemIndicator>
        </span>
      ) : null}
      {children}
    </MenuPrimitive.CheckboxItem>
  )
}

function DropdownMenuRadioGroup({ ...props }: MenuPrimitive.RadioGroup.Props) {
  return <MenuPrimitive.RadioGroup data-slot="dropdown-menu-radio-group" {...props} />
}

function DropdownMenuRadioItem({
  className,
  children,
  inset,
  marked = true,
  ...props
}: MenuPrimitive.RadioItem.Props & {
  inset?: boolean

  marked?: boolean
}) {
  const size = useSize()

  return (
    <MenuPrimitive.RadioItem
      data-slot="dropdown-menu-radio-item"
      data-inset={inset}
      className={cn(
        ROW,
        marked ? size.check : size.row,
        size.inset,
        size.icon,
        'relative focus:**:text-accent-foreground data-disabled:pointer-events-none data-disabled:opacity-50',
        className,
      )}
      {...props}
    >
      {marked ? (
        <span className={cn('pointer-events-none absolute', size.mark)}>
          <MenuPrimitive.RadioItemIndicator>
            <IconCheck />
          </MenuPrimitive.RadioItemIndicator>
        </span>
      ) : null}
      {children}
    </MenuPrimitive.RadioItem>
  )
}

function DropdownMenuSeparator({ className, ...props }: MenuPrimitive.Separator.Props) {
  const size = useSize()

  return (
    <MenuPrimitive.Separator
      data-slot="dropdown-menu-separator"

      className={cn('h-px bg-border', size.separator, className)}
      {...props}
    />
  )
}

function DropdownMenuShortcut({ className, ...props }: React.ComponentProps<'span'>) {
  const size = useSize()

  return (
    <span
      data-slot="dropdown-menu-shortcut"

      className={cn(
        'ms-auto tracking-widest text-muted-foreground rtl:tracking-normal',
        size.shortcut,
        className,
      )}
      {...props}
    />
  )
}

export type { DropdownMenuSize }
export {
  DropdownMenu,
  DropdownMenuPortal,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuLabel,
  DropdownMenuItem,
  DropdownMenuCheckboxItem,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuShortcut,
  DropdownMenuSub,
  DropdownMenuSubTrigger,
  DropdownMenuSubContent,
}
