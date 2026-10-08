"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import {
  fieldSizeClasses,
  type FieldSize,
} from "@/components/atoms/Input/fieldVariants";
import { Input as KobraInput } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import { Cancel01Icon, Search01Icon } from "@hugeicons/core-free-icons";
import { forwardRef } from "react";

interface Props {
  value: string;
  onChange: (next: string) => void;
  placeholder?: string;
  "aria-label"?: string;
  disabled?: boolean;
  loading?: boolean;
  maxLength?: number;
  size?: FieldSize;
  className?: string;
}

const sizeStyles = {
  sm: "pr-7 pl-8",
  md: "pr-9 pl-10",
  lg: "pr-10 pl-11",
} as const;

const iconOffset = {
  sm: { left: "left-2.5", right: "right-1" },
  md: { left: "left-3", right: "right-2" },
  lg: { left: "left-3.5", right: "right-2.5" },
} as const;

const iconSize = { sm: 14, md: 16, lg: 18 } as const;

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
      size = "lg",
      className,
    },
    ref,
  ) {
    const hasValue = value.length > 0;
    return (
      <div className={cn("relative w-full", className)}>
        <Icon
          icon={Search01Icon}
          size={iconSize[size]}
          className={cn(
            "pointer-events-none absolute top-1/2 -translate-y-1/2 text-muted-foreground",
            iconOffset[size].left,
          )}
        />
        <KobraInput
          ref={ref}
          type="search"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          aria-label={ariaLabel ?? placeholder}
          disabled={disabled}
          maxLength={maxLength}
          className={cn(
            fieldSizeClasses[size],
            sizeStyles[size],
            "[&::-webkit-search-cancel-button]:appearance-none",
          )}
        />
        {loading ? (
          <span
            className={cn(
              "absolute top-1/2 flex size-6 -translate-y-1/2 items-center justify-center text-muted-foreground",
              iconOffset[size].right,
            )}
          >
            <Spinner
              aria-label="Searching"
              className={size === "lg" ? "size-4" : "size-3.5"}
            />
          </span>
        ) : hasValue && !disabled ? (
          <button
            type="button"
            onClick={() => onChange("")}
            aria-label="Clear search"
            className={cn(
              "absolute top-1/2 flex size-6 -translate-y-1/2 items-center justify-center rounded-full text-muted-foreground focus-ring transition hover:bg-muted hover:text-foreground",
              iconOffset[size].right,
            )}
          >
            <Icon icon={Cancel01Icon} size={size === "lg" ? 14 : 12} />
          </button>
        ) : null}
      </div>
    );
  },
);
