"use client"

import * as React from "react"
import * as TabsPrimitive from "@radix-ui/react-tabs"

import { cn } from "@/lib/utils"

const Tabs = TabsPrimitive.Root

const TabsList = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.List>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.List>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.List
    ref={ref}
    className={cn(
      "w-80 h-10 p-[5px] bg-gray-100 rounded-3xl inline-flex justify-center items-center text-neutral-500 dark:bg-neutral-800 dark:text-neutral-400",
      className
    )}
    {...props}
  />
))
TabsList.displayName = TabsPrimitive.List.displayName

const TabsTrigger = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Trigger>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Trigger
    ref={ref}
    className={cn(
      "flex-1 text-center justify-center rounded-2xl text-gray-500 text-xs font-medium font-sans leading-tight px-3 py-2 flex items-start gap-2.5 whitespace-nowrap ring-offset-white transition-all",
      "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-950 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
      "data-[state=active]:bg-background data-[state=active]:text-foreground data-[state=active]:shadow-[0px_1px_2px_0px_rgba(0,0,0,0.05)]",
      className
    )}
    {...props}
  />
))
TabsTrigger.displayName = TabsPrimitive.Trigger.displayName

const TabsContent = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Content>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Content
    ref={ref}
    className={cn(
      "mt-2 ring-offset-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-950 focus-visible:ring-offset-2 dark:ring-offset-neutral-950 dark:focus-visible:ring-neutral-300",
      className
    )}
    {...props}
  />
))
TabsContent.displayName = TabsPrimitive.Content.displayName

export { Tabs, TabsList, TabsTrigger, TabsContent }

<div className="w-80 h-10 p-[5px] bg-gray-100 rounded-3xl inline-flex justify-start items-start">
  <div data-state="selected" className="flex-1 px-3 py-1.5 bg-background rounded-2xl shadow-[0px_1px_2px_0px_rgba(0,0,0,0.05)] flex justify-start items-start gap-2.5">
    <div className="flex-1 text-center justify-start text-foreground text-xs font-medium font-['Geist'] leading-tight">\
      One-time top up
    </div>
  </div>
  <div data-state="unselected" className="flex-1 px-3 py-1.5 rounded-sm flex justify-start items-start gap-2.5">
    <div className="flex-1 text-center justify-start text-gray-500 text-xs font-medium font-['Geist'] leading-tight">
      Auto-refill
    </div>
  </div>
</div>