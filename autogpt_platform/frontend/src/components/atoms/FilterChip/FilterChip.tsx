"use client";

import { cn } from "@/lib/utils";
import { X } from "@phosphor-icons/react";

type FilterChipSize = "sm" | "md" | "lg";

interface FilterChipProps {
  /** The label text displayed in the chip */
  label: string;
  /** Whether the chip is currently selected */
  selected?: boolean;
  /** Callback when the chip is clicked */
  onClick?: () => void;
  /** Whether to show a dismiss/remove button */
  dismissible?: boolean;
  /** Callback when the dismiss button is clicked */
  onDismiss?: () => void;
  /** Size variant of the chip */
  size?: FilterChipSize;
  /** Whether the chip is disabled */
  disabled?: boolean;
  /** Additional CSS classes */
  className?: string;
}

const sizeStyles: Record<FilterChipSize, string> = {
  sm: "px-3 py-1 text-sm gap-1.5",
  md: "px-4 py-1.5 text-base gap-2",
  lg: "px-6 py-2 text-lg gap-2.5 lg:text-xl lg:leading-9",
};

const iconSizes: Record<FilterChipSize, string> = {
  sm: "h-3 w-3",
  md: "h-4 w-4",
  lg: "h-5 w-5",
};

/**
 * A filter chip component for selecting/deselecting filter options.
 * Supports single and multi-select patterns with proper accessibility.
 */
export function FilterChip({
  label,
  selected = false,
  onClick,
  dismissible = false,
  onDismiss,
  size = "md",
  disabled = false,
  className,
}: FilterChipProps) {
  function handleDismiss(e: React.MouseEvent) {
    e.stopPropagation();
    onDismiss?.();
  }

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-pressed={selected}
      className={cn(
        // Base styles
        "inline-flex items-center justify-center rounded-full border font-medium transition-colors",
        // Focus styles
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-950 focus-visible:ring-offset-2 dark:focus-visible:ring-neutral-50",
        // Size styles
        sizeStyles[size],
        // State styles
        selected
          ? "border-neutral-900 bg-neutral-100 text-neutral-800 dark:border-neutral-100 dark:bg-neutral-800 dark:text-neutral-200"
          : "border-neutral-400 bg-transparent text-neutral-600 hover:bg-neutral-50 dark:border-neutral-500 dark:text-neutral-300 dark:hover:bg-neutral-800",
        // Disabled styles
        disabled && "pointer-events-none opacity-50",
        className,
      )}
    >
      <span>{label}</span>
      {dismissible && selected && (
        <span
          role="button"
          tabIndex={0}
          onClick={handleDismiss}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              handleDismiss(e as unknown as React.MouseEvent);
            }
          }}
          className="rounded-full p-0.5 hover:bg-neutral-200 dark:hover:bg-neutral-700"
          aria-label={`Remove ${label} filter`}
        >
          <X className={iconSizes[size]} weight="bold" />
        </span>
      )}
    </button>
  );
}
