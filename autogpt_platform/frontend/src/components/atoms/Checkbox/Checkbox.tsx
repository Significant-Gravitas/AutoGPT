"use client";

import * as CheckboxPrimitive from "@radix-ui/react-checkbox";
import { Tick02Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import * as React from "react";

import { cn } from "@/lib/utils";

const Checkbox = React.forwardRef<
  React.ElementRef<typeof CheckboxPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof CheckboxPrimitive.Root>
>(({ className, ...props }, ref) => (
  <CheckboxPrimitive.Root
    ref={ref}
    data-slot="checkbox"
    className={cn(
      "peer h-4 w-4 shrink-0 rounded-sm border border-action-hover shadow focus-visible:ring-1 focus-visible:ring-focus focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50 data-[state=checked]:border-action data-[state=checked]:bg-action data-[state=checked]:text-action-foreground",
      className,
    )}
    {...props}
  >
    <CheckboxPrimitive.Indicator
      data-slot="checkbox-indicator"
      className="flex items-center justify-center text-current"
    >
      <Icon
        aria-hidden
        icon={Tick02Icon}
        data-slot="checkbox-icon"
        className="h-4 w-4"
      />
    </CheckboxPrimitive.Indicator>
  </CheckboxPrimitive.Root>
));
Checkbox.displayName = CheckboxPrimitive.Root.displayName;

export { Checkbox };
