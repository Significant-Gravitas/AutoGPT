"use client";

import * as React from "react";
import {
  CaretSortIcon,
  CheckIcon,
  ChevronDownIcon,
  ChevronUpIcon,
} from "@radix-ui/react-icons";
import * as SelectPrimitive from "@radix-ui/react-select";

import { cn } from "@/lib/utils";

const Select = SelectPrimitive.Root;

// With `asChild` the child is the element and keeps its own slot.
const SelectGroup = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Group>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Group>
>((props, ref) => (
  <SelectPrimitive.Group
    ref={ref}
    {...(props.asChild ? {} : { "data-slot": "select-group" })}
    {...props}
  />
));
SelectGroup.displayName = SelectPrimitive.Group.displayName;

const SelectValue = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Value>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Value>
>((props, ref) => (
  <SelectPrimitive.Value
    ref={ref}
    {...(props.asChild ? {} : { "data-slot": "select-value" })}
    {...props}
  />
));
SelectValue.displayName = SelectPrimitive.Value.displayName;

// TODO(tokens): no role holds neutral-500 (#737373) or gray-400 (#9CA3AF).
// The dark: classes never apply (forcedTheme="light") and are DS6's
// (SECRT-3116) to delete.
const SelectTrigger = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Trigger>
>(({ className, children, ...props }, ref) => (
  <SelectPrimitive.Trigger
    ref={ref}
    data-slot="select-trigger"
    className={cn(
      "agpt-border-input agpt-shadow-input flex h-9 w-full items-center justify-between whitespace-nowrap rounded-md border bg-transparent px-3 py-2 text-body leading-5 ring-offset-surface placeholder:text-neutral-500 focus:outline-none disabled:cursor-not-allowed disabled:opacity-50 data-[state=open]:border-gray-400 dark:placeholder:text-neutral-400 [&>span]:line-clamp-1",
      className,
    )}
    {...props}
  >
    {children}
    <SelectPrimitive.Icon asChild>
      <CaretSortIcon data-slot="select-icon" className="h-4 w-4 opacity-50" />
    </SelectPrimitive.Icon>
  </SelectPrimitive.Trigger>
));
SelectTrigger.displayName = SelectPrimitive.Trigger.displayName;

const SelectScrollUpButton = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.ScrollUpButton>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.ScrollUpButton>
>(({ className, ...props }, ref) => (
  <SelectPrimitive.ScrollUpButton
    ref={ref}
    data-slot="select-scroll-up"
    className={cn(
      "flex cursor-default items-center justify-center py-1",
      className,
    )}
    {...props}
  >
    <ChevronUpIcon data-slot="select-scroll-up-icon" />
  </SelectPrimitive.ScrollUpButton>
));
SelectScrollUpButton.displayName = SelectPrimitive.ScrollUpButton.displayName;

const SelectScrollDownButton = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.ScrollDownButton>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.ScrollDownButton>
>(({ className, ...props }, ref) => (
  <SelectPrimitive.ScrollDownButton
    ref={ref}
    data-slot="select-scroll-down"
    className={cn(
      "flex cursor-default items-center justify-center py-1",
      className,
    )}
    {...props}
  >
    <ChevronDownIcon data-slot="select-scroll-down-icon" />
  </SelectPrimitive.ScrollDownButton>
));
SelectScrollDownButton.displayName =
  SelectPrimitive.ScrollDownButton.displayName;

// TODO(tokens): no role holds neutral-200 (#E5E5E5). neutral-950 (#0A0A0A) is
// held only by the `focus` role, the focus-ring colour, which is not a text
// colour (SECRT-3154).
const SelectContent = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Content>
>(({ className, children, position = "popper", ...props }, ref) => (
  <SelectPrimitive.Portal>
    <SelectPrimitive.Content
      ref={ref}
      data-slot="select-popup"
      className={cn(
        "relative z-50 max-h-96 min-w-[8rem] overflow-hidden rounded-md border border-neutral-200 bg-surface text-neutral-950 shadow-md data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2 dark:border-neutral-800 dark:bg-neutral-950 dark:text-neutral-50",
        position === "popper" &&
          "data-[side=bottom]:translate-y-1 data-[side=left]:-translate-x-1 data-[side=right]:translate-x-1 data-[side=top]:-translate-y-1",
        className,
      )}
      position={position}
      {...props}
    >
      <SelectScrollUpButton />
      <SelectPrimitive.Viewport
        data-slot="select-viewport"
        className={cn(
          "p-1",
          position === "popper" &&
            "h-[var(--radix-select-trigger-height)] w-full min-w-[var(--radix-select-trigger-width)]",
        )}
      >
        {children}
      </SelectPrimitive.Viewport>
      <SelectScrollDownButton />
    </SelectPrimitive.Content>
  </SelectPrimitive.Portal>
));
SelectContent.displayName = SelectPrimitive.Content.displayName;

// TODO(tokens): no role holds neutral-100 (#F5F5F5) or neutral-900 (#171717).
const SelectItem = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Item>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Item>
>(({ className, children, ...props }, ref) => (
  <SelectPrimitive.Item
    ref={ref}
    data-slot="select-item"
    className={cn(
      "relative flex w-full cursor-default select-none items-center rounded-sm py-1.5 pl-2 pr-8 text-body leading-5 outline-none focus:bg-neutral-100 focus:text-neutral-900 data-[disabled]:pointer-events-none data-[disabled]:opacity-50 dark:focus:bg-neutral-800 dark:focus:text-neutral-50",
      className,
    )}
    {...props}
  >
    <span
      data-slot="select-item-indicator-wrap"
      className="absolute right-2 flex h-3.5 w-3.5 items-center justify-center"
    >
      <SelectPrimitive.ItemIndicator data-slot="select-item-indicator">
        <CheckIcon data-slot="select-item-check-icon" className="h-4 w-4" />
      </SelectPrimitive.ItemIndicator>
    </span>
    <SelectPrimitive.ItemText data-slot="select-item-text">
      {children}
    </SelectPrimitive.ItemText>
  </SelectPrimitive.Item>
));
SelectItem.displayName = SelectPrimitive.Item.displayName;

// TODO(tokens): no role holds neutral-100 (#F5F5F5).
const SelectSeparator = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Separator>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Separator>
>(({ className, ...props }, ref) => (
  <SelectPrimitive.Separator
    ref={ref}
    data-slot="select-separator"
    className={cn(
      "-mx-1 my-1 h-px bg-neutral-100 dark:bg-neutral-800",
      className,
    )}
    {...props}
  />
));
SelectSeparator.displayName = SelectPrimitive.Separator.displayName;

export {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
};
