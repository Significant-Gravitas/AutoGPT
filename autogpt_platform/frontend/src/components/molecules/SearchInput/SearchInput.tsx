"use client";

import { forwardRef } from "react";
import {
  CircleNotchIcon,
  MagnifyingGlassIcon,
  XIcon,
} from "@phosphor-icons/react";

import { cn } from "@/lib/utils";

interface Props {
  value: string;
  onChange: (next: string) => void;
  placeholder?: string;
  "aria-label"?: string;
  disabled?: boolean;
  loading?: boolean;
  maxLength?: number;
  size?: "small" | "medium";
  className?: string;
}

const sizeStyles = {
  small: "h-[36px] pl-10 pr-9 text-sm leading-[22px]",
  medium: "h-[46px] pl-12 pr-10 text-sm leading-[22px]",
} as const;

const iconOffset = {
  small: { left: "left-3", right: "right-2" },
  medium: { left: "left-4", right: "right-3" },
} as const;

export const SearchInput = forwardRef<HTMLInputElement, Props>(
  function SearchInput(
    {
      value,
      onChange,
      placeholder = "Search",
      "aria-label": ariaLabel,
      disabled,
      loading,
      maxLength,
      size = "medium",
      className,
    },
    ref,
  ) {
    const hasValue = value.length > 0;
    return (
      <div className={cn("relative w-full", className)}>
        <MagnifyingGlassIcon
          size={size === "small" ? 16 : 20}
          className={cn(
            "pointer-events-none absolute top-1/2 -translate-y-1/2 text-muted-foreground",
            iconOffset[size].left,
          )}
        />
        <input
          ref={ref}
          type="search"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          aria-label={ariaLabel ?? placeholder}
          disabled={disabled}
          maxLength={maxLength}
          className={cn(
            "w-full rounded-xl border border-input bg-background text-foreground placeholder:text-muted-foreground focus:border-ring focus:outline-none focus:ring-1 focus:ring-ring disabled:cursor-not-allowed disabled:opacity-60",
            sizeStyles[size],
            "[&::-webkit-search-cancel-button]:appearance-none",
          )}
        />
        {loading ? (
          <span
            role="status"
            aria-label="Searching"
            className={cn(
              "absolute top-1/2 flex size-6 -translate-y-1/2 items-center justify-center text-muted-foreground",
              iconOffset[size].right,
            )}
          >
            <CircleNotchIcon
              size={size === "small" ? 14 : 16}
              weight="bold"
              className="animate-spin"
            />
          </span>
        ) : hasValue && !disabled ? (
          <button
            type="button"
            onClick={() => onChange("")}
            aria-label="Clear search"
            className={cn(
              "absolute top-1/2 flex size-6 -translate-y-1/2 items-center justify-center rounded-full text-muted-foreground transition hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
              iconOffset[size].right,
            )}
          >
            <XIcon size={size === "small" ? 12 : 14} weight="bold" />
          </button>
        ) : null}
      </div>
    );
  },
);
