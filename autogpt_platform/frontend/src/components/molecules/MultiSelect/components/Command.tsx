"use client";

import * as React from "react";
import { Command as CommandPrimitive } from "cmdk";

import { cn } from "@/lib/utils";

const Command = React.forwardRef<
  React.ElementRef<typeof CommandPrimitive>,
  React.ComponentPropsWithoutRef<typeof CommandPrimitive>
>(function Command({ className, ...props }, ref) {
  return (
    <CommandPrimitive
      ref={ref}
      data-slot="command"
      className={cn(
        // TODO(tokens): text-neutral-950 #0A0A0A is the `focus` role, a ring
        // colour, not a text colour (SECRT-3154).
        "flex h-full w-full flex-col overflow-hidden rounded-md bg-surface text-neutral-950 dark:bg-neutral-950 dark:text-neutral-50",
        className,
      )}
      {...props}
    />
  );
});
Command.displayName = CommandPrimitive.displayName;

const CommandList = React.forwardRef<
  React.ElementRef<typeof CommandPrimitive.List>,
  React.ComponentPropsWithoutRef<typeof CommandPrimitive.List>
>(function CommandList({ className, ...props }, ref) {
  return (
    <CommandPrimitive.List
      ref={ref}
      data-slot="command-list"
      className={cn(
        "max-h-[300px] overflow-y-auto overflow-x-hidden",
        className,
      )}
      {...props}
    />
  );
});
CommandList.displayName = CommandPrimitive.List.displayName;

const CommandEmpty = React.forwardRef<
  React.ElementRef<typeof CommandPrimitive.Empty>,
  React.ComponentPropsWithoutRef<typeof CommandPrimitive.Empty>
>(function CommandEmpty(props, ref) {
  return (
    <CommandPrimitive.Empty
      ref={ref}
      data-slot="command-empty"
      className="py-6 text-center"
      {...props}
    />
  );
});
CommandEmpty.displayName = CommandPrimitive.Empty.displayName;

const CommandItem = React.forwardRef<
  React.ElementRef<typeof CommandPrimitive.Item>,
  React.ComponentPropsWithoutRef<typeof CommandPrimitive.Item>
>(function CommandItem({ className, ...props }, ref) {
  return (
    <CommandPrimitive.Item
      ref={ref}
      data-slot="command-item"
      className={cn(
        // TODO(tokens): no exact role: data-[selected=true]:bg-neutral-100
        // #F5F5F5, data-[selected=true]:text-neutral-900 #171717.
        "relative flex cursor-default select-none items-center rounded-sm px-2 py-1.5 outline-none data-[disabled=true]:pointer-events-none data-[selected=true]:bg-neutral-100 data-[selected=true]:text-neutral-900 data-[disabled=true]:opacity-50 dark:data-[selected=true]:bg-neutral-800 dark:data-[selected=true]:text-neutral-50",
        className,
      )}
      {...props}
    />
  );
});
CommandItem.displayName = CommandPrimitive.Item.displayName;

export { Command, CommandEmpty, CommandItem, CommandList };
