"use client";

import * as React from "react";
import * as TooltipPrimitive from "@radix-ui/react-tooltip";

import { cn } from "@/lib/utils";

function TooltipProvider(
  props: React.ComponentProps<typeof TooltipPrimitive.Provider>,
) {
  return <TooltipPrimitive.Provider disableHoverableContent {...props} />;
}

interface Props {
  children: React.ReactNode;
  delayDuration?: number;
  /** Take the delay from the nearest TooltipProvider instead, as the deleted
   *  components/ui/tooltip did (700ms at the app root, 0 in the sidebar). */
  inheritDelay?: boolean;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}

function Tooltip({
  children,
  delayDuration = 10,
  inheritDelay = false,
  open,
  onOpenChange,
}: Props) {
  return (
    <TooltipPrimitive.Root
      delayDuration={inheritDelay ? undefined : delayDuration}
      open={open}
      onOpenChange={onOpenChange}
    >
      {children}
    </TooltipPrimitive.Root>
  );
}

// With `asChild` the child is the trigger element and keeps its own slot.
const TooltipTrigger = React.forwardRef<
  React.ElementRef<typeof TooltipPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof TooltipPrimitive.Trigger>
>((props, ref) => (
  <TooltipPrimitive.Trigger
    ref={ref}
    {...(props.asChild ? {} : { "data-slot": "tooltip-trigger" })}
    {...props}
  />
));
TooltipTrigger.displayName = TooltipPrimitive.Trigger.displayName;

// Opt-in: content renders inline by default, so a trigger inside an
// `overflow-hidden` ancestor clips its own tooltip. Wrap in this to escape.
const TooltipPortal = TooltipPrimitive.Portal;

// Shared with every tooltip in the app so they look the same: dark chip,
// light text.
export const tooltipContentClassName =
  "z-50 max-w-xs origin-(--radix-tooltip-content-transform-origin) overflow-hidden rounded-md bg-action-hover px-3 py-1.5 text-caption font-normal leading-normal text-surface-muted animate-in fade-in-0 zoom-in-95 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2";

const TooltipContent = React.forwardRef<
  React.ElementRef<typeof TooltipPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TooltipPrimitive.Content>
>(({ className, sideOffset = 4, ...props }, ref) => (
  <TooltipPrimitive.Content
    ref={ref}
    data-slot="tooltip-content"
    sideOffset={sideOffset}
    className={cn(
      tooltipContentClassName,
      // Important beats Radix's inline pointer style in dialogs; an explicit
      // override must likewise use !pointer-events-auto.
      "!pointer-events-none [[data-radix-popper-content-wrapper]:has(>&)]:pointer-events-none",
      className,
    )}
    {...props}
  />
));
TooltipContent.displayName = TooltipPrimitive.Content.displayName;

export {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
  TooltipPortal,
  TooltipProvider,
};
