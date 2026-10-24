"use client";

import React from "react";
import { cn } from "@/lib/utils";
import * as CollapsiblePrimitive from "@radix-ui/react-collapsible";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  trigger: React.ReactNode;
  children: React.ReactNode;
  defaultOpen?: boolean;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  className?: string;
  triggerClassName?: string;
  contentClassName?: string;
}

export function Collapsible({
  trigger,
  children,
  defaultOpen = false,
  open,
  onOpenChange,
  className,
  triggerClassName,
  contentClassName,
}: Props) {
  const [isOpen, setIsOpen] = React.useState(defaultOpen);
  const isControlled = open !== undefined;
  const openState = isControlled ? open : isOpen;

  const handleOpenChange = (newOpen: boolean) => {
    if (!isControlled) {
      setIsOpen(newOpen);
    }
    onOpenChange?.(newOpen);
  };

  return (
    <CollapsiblePrimitive.Root
      open={openState}
      onOpenChange={handleOpenChange}
      data-slot="collapsible"
      className={cn("w-full", className)}
    >
      <CollapsiblePrimitive.Trigger
        data-slot="collapsible-trigger"
        className={cn(
          "flex w-full items-center justify-between text-left transition-all duration-200 hover:opacity-80",
          triggerClassName,
        )}
      >
        <div
          data-slot="collapsible-trigger-content"
          className="flex-end flex flex-wrap items-center gap-2"
        >
          {trigger}
          <Icon
            icon={ArrowDown01Icon}
            data-slot="collapsible-icon"
            className={cn(
              "inline-flex h-4 w-4 transition-transform duration-200",
              openState && "rotate-180",
            )}
          />
        </div>
      </CollapsiblePrimitive.Trigger>
      <CollapsiblePrimitive.Content
        data-slot="collapsible-panel"
        className={cn("overflow-hidden", contentClassName)}
      >
        <div data-slot="collapsible-panel-content" className="pt-2">
          {children}
        </div>
      </CollapsiblePrimitive.Content>
    </CollapsiblePrimitive.Root>
  );
}
