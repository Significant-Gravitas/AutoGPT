"use client";

import * as React from "react";
import * as TabsPrimitive from "@radix-ui/react-tabs";

import { cn } from "@/lib/utils";

const Tabs = TabsPrimitive.Root;

const TabsList = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.List>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.List>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.List
    ref={ref}
    className={cn(
      "inline-flex h-10 w-80 items-center justify-center rounded-3xl bg-gray-100 p-[5px] text-neutral-500 dark:bg-neutral-800 dark:text-neutral-400",
      className,
    )}
    {...props}
  />
));
TabsList.displayName = TabsPrimitive.List.displayName;

const TabsTrigger = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Trigger>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Trigger
    ref={ref}
    className={cn(
      "flex flex-1 items-start justify-center gap-2.5 whitespace-nowrap rounded-2xl px-3 py-2 text-center font-sans text-xs font-medium leading-tight text-gray-500 ring-offset-white transition-all",
      "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-950 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
      "data-[state=active]:bg-background data-[state=active]:text-foreground data-[state=active]:shadow-subtle",
      className,
    )}
    {...props}
  />
));
TabsTrigger.displayName = TabsPrimitive.Trigger.displayName;

const TabsContent = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Content>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Content
    ref={ref}
    className={cn(
      "mt-2 ring-offset-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-950 focus-visible:ring-offset-2 dark:ring-offset-neutral-950 dark:focus-visible:ring-neutral-300",
      className,
    )}
    {...props}
  />
));
TabsContent.displayName = TabsPrimitive.Content.displayName;

export { Tabs, TabsList, TabsTrigger, TabsContent };

<div className="inline-flex h-10 w-80 items-start justify-start rounded-3xl bg-gray-100 p-[5px]">
  <div
    data-state="selected"
    className="flex flex-1 items-start justify-start gap-2.5 rounded-2xl bg-background px-3 py-1.5 shadow-subtle"
  >
    <div className="flex-1 justify-start text-center font-['Geist'] text-xs font-medium leading-tight text-foreground">
      \ One-time top up
    </div>
  </div>
  <div
    data-state="unselected"
    className="flex flex-1 items-start justify-start gap-2.5 rounded-sm px-3 py-1.5"
  >
    <div className="flex-1 justify-start text-center font-['Geist'] text-xs font-medium leading-tight text-gray-500">
      Auto-refill
    </div>
  </div>
</div>;
