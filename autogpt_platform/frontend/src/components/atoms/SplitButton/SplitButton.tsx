"use client";

import { Button } from "@/components/atoms/Button/Button";
import {
  type ButtonProps,
  extendedButtonVariants,
} from "@/components/atoms/Button/helpers";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { cn } from "@/lib/utils";
import { CaretDownIcon } from "@phosphor-icons/react/dist/ssr";
import React from "react";

export interface SplitButtonItem {
  key: string;
  label: React.ReactNode;
  onSelect: () => void;
}

interface Props {
  primaryLabel: React.ReactNode;
  onPrimaryClick: React.MouseEventHandler<HTMLButtonElement>;
  items: SplitButtonItem[];
  variant?: ButtonProps["variant"];
  size?: ButtonProps["size"];
  leftIcon?: React.ReactNode;
  loading?: boolean;
  disabled?: boolean;
  primaryAriaLabel?: string;
  menuAriaLabel?: string;
  align?: "start" | "center" | "end";
  className?: string;
  buttonClassName?: string;
}

// A primary action segment (the Button atom) joined to a caret segment that
// opens a DropdownMenu of alternate actions, styled to read as one control.
// The caret is a raw button (styled with the shared Button variants) because
// Radix's asChild Slot needs a forwardRef child and the Button atom has none.
export function SplitButton({
  primaryLabel,
  onPrimaryClick,
  items,
  variant = "primary",
  size = "large",
  leftIcon,
  loading = false,
  disabled = false,
  primaryAriaLabel,
  menuAriaLabel = "More options",
  align = "end",
  className,
  buttonClassName,
}: Props) {
  return (
    <div className={cn("inline-flex items-stretch", className)}>
      <Button
        variant={variant}
        size={size}
        loading={loading}
        disabled={disabled}
        leftIcon={leftIcon}
        onClick={onPrimaryClick}
        aria-label={primaryAriaLabel}
        withTooltip={false}
        className={cn("min-w-0 rounded-r-none border-r-0", buttonClassName)}
      >
        {primaryLabel}
      </Button>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button
            type="button"
            disabled={disabled || loading}
            aria-label={menuAriaLabel}
            className={cn(
              extendedButtonVariants({ variant, size }),
              "min-w-0 rounded-l-none border-l border-l-zinc-300 px-2",
              buttonClassName,
            )}
          >
            <CaretDownIcon size={14} weight="bold" />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align={align}>
          {items.map((item) => (
            <DropdownMenuItem
              key={item.key}
              onSelect={item.onSelect}
              data-testid={`split-button-item-${item.key}`}
            >
              {item.label}
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}
