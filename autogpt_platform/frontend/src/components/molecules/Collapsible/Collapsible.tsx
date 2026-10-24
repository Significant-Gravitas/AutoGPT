"use client";

import * as CollapsiblePrimitive from "@radix-ui/react-collapsible";
import React from "react";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export const CollapsibleRoot = CollapsiblePrimitive.Root;

export const CollapsibleTrigger = CollapsiblePrimitive.CollapsibleTrigger;

export const CollapsibleContent = CollapsiblePrimitive.CollapsibleContent;

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
    <CollapsibleRoot
      open={openState}
      onOpenChange={handleOpenChange}
      className={cn("w-full", className)}
    >
      <CollapsibleTrigger
        className={cn(
          "flex w-full items-center justify-between text-left transition-all duration-200 hover:opacity-80",
          triggerClassName,
        )}
      >
        <div className="flex-end flex flex-wrap items-center gap-2">
          {trigger}
          <Icon
            icon={ArrowDown01Icon}
            className={cn(
              "inline-flex h-4 w-4 transition-transform duration-200",
              openState && "rotate-180",
            )}
          />
        </div>
      </CollapsibleTrigger>
      <CollapsibleContent className={cn("overflow-hidden", contentClassName)}>
        <div className="pt-2">{children}</div>
      </CollapsibleContent>
    </CollapsibleRoot>
  );
}
