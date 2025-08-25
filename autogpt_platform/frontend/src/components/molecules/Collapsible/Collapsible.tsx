"use client";

import React from "react";
import { cn } from "@/lib/utils";
import {
  Collapsible as BaseCollapsible,
  CollapsibleTrigger as BaseCollapsibleTrigger,
  CollapsibleContent as BaseCollapsibleContent,
} from "@/components/ui/collapsible";
import { CaretDownIcon } from "@phosphor-icons/react";

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
    <BaseCollapsible
      open={openState}
      onOpenChange={handleOpenChange}
      className={cn("w-full", className)}
    >
      <BaseCollapsibleTrigger
        className={cn(
          "flex w-full items-center justify-between text-left transition-all duration-200 hover:opacity-80",
          triggerClassName,
        )}
      >
        <div className="flex-end flex flex-wrap items-center gap-2">
          {trigger}
          <CaretDownIcon
            className={cn(
              "inline-flex h-4 w-4 transition-transform duration-200",
              openState && "rotate-180",
            )}
          />
        </div>
      </BaseCollapsibleTrigger>
      <BaseCollapsibleContent
        className={cn("overflow-hidden", contentClassName)}
      >
        <div className="pt-2">{children}</div>
      </BaseCollapsibleContent>
    </BaseCollapsible>
  );
}
