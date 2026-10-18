"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { cn } from "@/lib/utils";
import { MoreHorizontalIcon } from "@hugeicons/core-free-icons";
import { ReactNode, useState } from "react";

interface Props {
  /** The secondary controls (connection, approvals, test mode, voice...). */
  children: ReactNode;
  /** Keeps the panel shut, e.g. while voice mode takes over the composer. */
  forceClosed?: boolean;
  disabled?: boolean;
  className?: string;
}

/**
 * The "..." button that keeps the composer to its essentials: the controls
 * that rarely change between messages live in a panel instead of a row that
 * overflows a phone screen. A popover rather than a menu, because the
 * controls inside open menus and pickers of their own.
 */
export function ComposerOverflowMenu({
  children,
  forceClosed = false,
  disabled,
  className,
}: Props) {
  const [isOpen, setIsOpen] = useState(false);

  // Forget the open state while forced shut, or the panel pops back open
  // on its own once voice mode ends.
  if (forceClosed && isOpen) setIsOpen(false);

  return (
    <Popover open={isOpen && !forceClosed} onOpenChange={setIsOpen}>
      <PopoverTrigger asChild>
        <Button
          type="button"
          variant="icon"
          size="icon"
          aria-label="More options"
          data-testid="composer-overflow-button"
          disabled={disabled}
          className={cn(
            "border-transparent bg-transparent text-black shadow-none hover:border-transparent hover:bg-zinc-100 hover:text-black",
            className,
            disabled && "opacity-40",
          )}
        >
          <Icon icon={MoreHorizontalIcon} className="h-4 w-4" />
        </Button>
      </PopoverTrigger>
      <PopoverContent
        side="top"
        align="end"
        data-testid="composer-overflow-panel"
        className="flex w-auto max-w-[calc(100vw-1.5rem)] flex-wrap items-center gap-1 p-1.5"
      >
        {children}
      </PopoverContent>
    </Popover>
  );
}
