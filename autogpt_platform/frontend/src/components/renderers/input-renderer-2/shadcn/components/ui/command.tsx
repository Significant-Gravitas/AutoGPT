'use client';

import { Command as CommandPrimitive } from 'cmdk';
import { Search } from 'lucide-react';
import { ComponentProps, ComponentPropsWithoutRef, ElementRef, forwardRef } from 'react';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from './dialog';
import { cn } from '../../lib/utils';

/**
 * The root Command component that provides command menu functionality
 * @see https://ui.shadcn.com/docs/components/command
 * @returns A command menu component
 */

const Command = forwardRef<ElementRef<typeof CommandPrimitive>, ComponentPropsWithoutRef<typeof CommandPrimitive>>(
  ({ className, ...props }, ref) => {
    return (
      <CommandPrimitive
        ref={ref}
        data-slot='command'
        className={cn(
          'bg-popover text-popover-foreground flex h-full w-full flex-col overflow-hidden rounded-md',
          className,
        )}
        {...props}
      />
    );
  },
);

/**
 * A dialog wrapper for the Command component to display it in a modal
 *
 * @param title
 * @param description
 * @param className
 * @param showCloseButton
 * @param props - The props for the CommandDialog component
 * @param props.children - The content of the command dialog
 * @returns A dialog component containing the command menu
 */
function CommandDialog({
  title = 'Command Palette',
  description = 'Search for a command to run...',
  children,
  className,
  showCloseButton = true,
  ...props
}: ComponentProps<typeof Dialog> & {
  title?: string;
  description?: string;
  className?: string;
  showCloseButton?: boolean;
}) {
  return (
    <Dialog {...props}>
      <DialogHeader className='sr-only'>
        <DialogTitle>{title}</DialogTitle>
        <DialogDescription>{description}</DialogDescription>
      </DialogHeader>
      <DialogContent className={cn('overflow-hidden p-0', className)} showCloseButton={showCloseButton}>
        <Command className='[&_[cmdk-group-heading]]:text-muted-foreground **:data-[slot=command-input-wrapper]:h-12 [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:font-medium [&_[cmdk-group]]:px-2 [&_[cmdk-group]:not([hidden])_~[cmdk-group]]:pt-0 [&_[cmdk-input-wrapper]_svg]:h-5 [&_[cmdk-input-wrapper]_svg]:w-5 [&_[cmdk-input]]:h-12 [&_[cmdk-item]]:px-2 [&_[cmdk-item]]:py-3 [&_[cmdk-item]_svg]:h-5 [&_[cmdk-item]_svg]:w-5'>
          {children}
        </Command>
      </DialogContent>
    </Dialog>
  );
}

/**
 * The search input field for the command menu
 *
 * @param props - The props for the CommandInput component
 * @param props.className - Additional CSS classes to apply
 * @returns A search input component for the command menu
 */
const CommandInput = forwardRef<
  ElementRef<typeof CommandPrimitive.Input>,
  ComponentPropsWithoutRef<typeof CommandPrimitive.Input>
>(({ className, ...props }, ref) => (
  <div className='flex items-center border-b px-3' {...{ 'cmdk-input-wrapper': '' }}>
    <Search className='mr-2 h-4 w-4 shrink-0 opacity-50' />
    <CommandPrimitive.Input
      ref={ref}
      className={cn(
        'flex h-10 w-full rounded-md bg-transparent py-3 text-sm outline-none placeholder:text-muted-foreground disabled:cursor-not-allowed disabled:opacity-50',
        className,
      )}
      {...props}
    />
  </div>
));

CommandInput.displayName = CommandPrimitive.Input.displayName;

/**
 * Container for the list of command items
 *
 * @param props - The props for the CommandList component
 * @param props.className - Additional CSS classes to apply
 * @returns A container component for command items
 */
function CommandList({ className, ...props }: ComponentProps<typeof CommandPrimitive.List>) {
  return (
    <CommandPrimitive.List
      data-slot='command-list'
      className={cn('max-h-[300px] scroll-py-1 overflow-x-hidden overflow-y-auto', className)}
      {...props}
    />
  );
}

/**
 * Displayed when no results are found
 */
function CommandEmpty({ ...props }: ComponentProps<typeof CommandPrimitive.Empty>) {
  return <CommandPrimitive.Empty data-slot='command-empty' className='py-6 text-center text-sm' {...props} />;
}

/**
 * Groups related command items together
 *
 * @param props - The props for the CommandGroup component
 * @param props.className - Additional CSS classes to apply
 * @returns A group container for related command items
 */
function CommandGroup({ className, ...props }: ComponentProps<typeof CommandPrimitive.Group>) {
  return (
    <CommandPrimitive.Group
      data-slot='command-group'
      className={cn(
        'text-foreground [&_[cmdk-group-heading]]:text-muted-foreground overflow-hidden p-1 [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5 [&_[cmdk-group-heading]]:text-xs [&_[cmdk-group-heading]]:font-medium',
        className,
      )}
      {...props}
    />
  );
}

/**
 * Visual separator between command groups or items
 * @returns A separator component for the command menu
 */
function CommandSeparator({ className, ...props }: ComponentProps<typeof CommandPrimitive.Separator>) {
  return (
    <CommandPrimitive.Separator
      data-slot='command-separator'
      className={cn('bg-border -mx-1 h-px', className)}
      {...props}
    />
  );
}

/**
 * Individual command item that can be selected
 *
 * @param props - The props for the CommandItem component
 * @param props.className - Additional CSS classes to apply
 * @returns A selectable command item component
 */
function CommandItem({ className, ...props }: ComponentProps<typeof CommandPrimitive.Item>) {
  return (
    <CommandPrimitive.Item
      data-slot='command-item'
      className={cn(
        "data-[selected=true]:bg-accent data-[selected=true]:text-accent-foreground [&_svg:not([class*='text-'])]:text-muted-foreground relative flex cursor-default items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-hidden select-none data-[disabled=true]:pointer-events-none data-[disabled=true]:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",
        className,
      )}
      {...props}
    />
  );
}

/**
 * Displays keyboard shortcut hints for command items
 *
 * @param props - The props for the CommandShortcut component
 * @param props.className - Additional CSS classes to apply
 * @returns A component displaying keyboard shortcuts
 */

function CommandShortcut({ className, ...props }: ComponentProps<'span'>) {
  return (
    <span
      data-slot='command-shortcut'
      className={cn('text-muted-foreground ml-auto text-xs tracking-widest', className)}
      {...props}
    />
  );
}

export {
  Command,
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
  CommandShortcut,
};
