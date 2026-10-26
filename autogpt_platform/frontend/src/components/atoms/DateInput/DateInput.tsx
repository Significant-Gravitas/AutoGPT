"use client";

import * as React from "react";
import { Calendar as CalendarIcon } from "lucide-react";
import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { Calendar } from "@/components/atoms/Calendar/Calendar";

function toLocalISODateString(d: Date) {
  const year = d.getFullYear();
  const month = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function parseISODateString(s?: string): Date | undefined {
  if (!s) return undefined;
  // Expecting "YYYY-MM-DD"
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(s);
  if (!m) return undefined;
  const [_, y, mo, d] = m;
  const date = new Date(Number(y), Number(mo) - 1, Number(d));
  return isNaN(date.getTime()) ? undefined : date;
}

export interface DateInputProps {
  value?: string;
  onChange?: (value?: string) => void;
  disabled?: boolean;
  readonly?: boolean;
  placeholder?: string;
  autoFocus?: boolean;
  className?: string;
  label?: string;
  hideLabel?: boolean;
  error?: string;
  id?: string;
  size?: "default" | "small";
  "aria-label"?: string;
  "aria-labelledby"?: string;
  "aria-describedby"?: string;
}

export const DateInput = ({
  value,
  onChange,
  disabled,
  readonly,
  placeholder,
  autoFocus,
  className,
  label,
  hideLabel = false,
  error,
  id,
  size = "default",
  "aria-label": ariaLabel,
  "aria-labelledby": ariaLabelledBy,
  "aria-describedby": ariaDescribedBy,
}: DateInputProps) => {
  const selected = React.useMemo(() => parseISODateString(value), [value]);
  const [open, setOpen] = React.useState(false);

  const setDate = (d?: Date) => {
    onChange?.(d ? toLocalISODateString(d) : undefined);
    setOpen(false);
  };

  const buttonText =
    selected?.toLocaleDateString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
    }) ||
    placeholder ||
    "Pick a date";

  const isDisabled = disabled || readonly;

  const triggerStyles = cn(
    // Base styles matching other form components
    "rounded-3xl border border-border-base bg-surface px-4 shadow-none",
    "font-normal text-foreground-base w-full text-body leading-5",
    "placeholder:font-normal !placeholder:text-placeholder",
    // Focus and hover states
    "focus:border-foreground-subtle focus:shadow-none focus:outline-none focus:ring-1 focus:ring-foreground-subtle focus:ring-offset-0",
    // Error state
    error && "border-1.5 border-error focus:border-error focus:ring-error",
    // Placeholder styling
    !selected && "text-foreground-subtle",
    "justify-start text-left",
    // Size variants
    size === "default" && "h-[2.875rem] py-2.5",
    className,
    size === "small" && [
      "min-h-9", // 36px minimum
      "py-2",
      "text-body",
      "placeholder:text-body",
    ],
  );

  return (
    <div data-slot="date-input" className="flex flex-col gap-1">
      {label && !hideLabel && (
        // TODO(tokens): no role is gray-700 #374151.
        <label
          data-slot="date-input-label"
          htmlFor={id}
          className="text-body font-medium leading-5 text-gray-700"
        >
          {label}
        </label>
      )}
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            data-slot="date-input-trigger"
            type="button"
            variant="legacy-ghost"
            className={triggerStyles}
            disabled={isDisabled}
            autoFocus={autoFocus}
            id={id}
            aria-label={ariaLabel ?? (hideLabel && label ? label : undefined)}
            aria-labelledby={ariaLabelledBy}
            aria-describedby={ariaDescribedBy}
          >
            <CalendarIcon
              data-slot="date-input-icon"
              className={cn("mr-2", size === "default" ? "h-4 w-4" : "h-3 w-3")}
            />
            {buttonText}
          </Button>
        </PopoverTrigger>
        {/* TODO(tokens): today's popover look. No exact role: border-neutral-200
            #E5E5E5; text-neutral-950 #0A0A0A is the `focus` role, a ring colour,
            not a text colour (SECRT-3154). */}
        <PopoverContent
          data-slot="date-input-popup"
          className="w-auto rounded-medium border-neutral-200 p-0 text-neutral-950 dark:border-neutral-800 dark:bg-neutral-950 dark:text-neutral-50"
          sideOffset={6}
        >
          <Calendar
            data-slot="date-input-calendar"
            mode="single"
            selected={selected}
            onSelect={setDate}
            showOutsideDays
            // Prevent selection when disabled/readonly
            modifiersClassNames={{
              disabled: "pointer-events-none opacity-50",
            }}
          />
        </PopoverContent>
      </Popover>
      {error && (
        <span
          data-slot="date-input-error"
          className="text-body leading-5 text-error"
        >
          {error}
        </span>
      )}
    </div>
  );
};
