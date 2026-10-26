"use client";

import * as React from "react";
import * as TooltipPrimitive from "@radix-ui/react-tooltip";

import { tooltipContentClassName } from "@/components/atoms/Tooltip/BaseTooltip";
import { cn } from "@/lib/utils";

function TooltipProvider(
  props: React.ComponentProps<typeof TooltipPrimitive.Provider>,
) {
  return <TooltipPrimitive.Provider disableHoverableContent {...props} />;
}

const Tooltip = TooltipPrimitive.Root;

const TooltipTrigger = TooltipPrimitive.Trigger;

const TooltipContent = React.forwardRef<
  React.ElementRef<typeof TooltipPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TooltipPrimitive.Content>
>(({ className, sideOffset = 4, ...props }, ref) => (
  <TooltipPrimitive.Portal>
    <TooltipPrimitive.Content
      ref={ref}
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
  </TooltipPrimitive.Portal>
));
TooltipContent.displayName = TooltipPrimitive.Content.displayName;

export { Tooltip, TooltipTrigger, TooltipContent, TooltipProvider };
