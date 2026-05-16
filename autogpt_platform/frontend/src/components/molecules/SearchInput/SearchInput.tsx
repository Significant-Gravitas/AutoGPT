"use client";

import { forwardRef } from "react";
import { MagnifyingGlassIcon, XIcon } from "@phosphor-icons/react";

import { cn } from "@/lib/utils";

interface Props {
  value: string;
  onChange: (next: string) => void;
  placeholder?: string;
  "aria-label"?: string;
  disabled?: boolean;
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
            "pointer-events-none absolute top-1/2 -translate-y-1/2 text-zinc-400",
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
          className={cn(
            "w-full rounded-full border border-zinc-200 bg-white text-textBlack placeholder:text-zinc-400 focus:border-purple-400 focus:outline-none focus:ring-1 focus:ring-purple-400 disabled:cursor-not-allowed disabled:opacity-60",
            sizeStyles[size],
            "[&::-webkit-search-cancel-button]:appearance-none",
          )}
        />
        {hasValue && !disabled ? (
          <button
            type="button"
            onClick={() => onChange("")}
            aria-label="Clear search"
            className={cn(
              "absolute top-1/2 flex size-6 -translate-y-1/2 items-center justify-center rounded-full text-zinc-500 transition hover:bg-zinc-100 hover:text-zinc-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-purple-400",
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
