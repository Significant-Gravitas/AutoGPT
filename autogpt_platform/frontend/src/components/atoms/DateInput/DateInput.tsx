"use client";

import * as React from "react";
import { Calendar as CalendarIcon } from "lucide-react";
import { Button } from "@/components/__legacy__/ui/button";
import { cn } from "@/lib/utils";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/__legacy__/ui/popover";
import { Calendar } from "@/components/__legacy__/ui/calendar";

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
    "rounded-3xl border border-zinc-200 bg-white px-4 shadow-none",
    "font-normal text-black w-full text-sm",
    "placeholder:font-normal !placeholder:text-zinc-400",
    // Focus and hover states
    "focus:border-zinc-400 focus:shadow-none focus:outline-none focus:ring-1 focus:ring-zinc-400 focus:ring-offset-0",
    // Error state
    error &&
      "border-1.5 border-red-500 focus:border-red-500 focus:ring-red-500",
    // Placeholder styling
    !selected && "text-zinc-400",
    "justify-start text-left",
    // Size variants
    size === "default" && "h-[2.875rem] py-2.5",
    className,
    size === "small" && [
      "min-h-[2.25rem]", // 36px minimum
      "py-2",
      "text-sm leading-[22px]",
      "placeholder:text-sm placeholder:leading-[22px]",
    ],
  );

  return (
    <div className="flex flex-col gap-1">
      {label && !hideLabel && (
        <label htmlFor={id} className="text-sm font-medium text-gray-700">
          {label}
        </label>
      )}
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            type="button"
            variant="ghost"
            className={triggerStyles}
            disabled={isDisabled}
            autoFocus={autoFocus}
            id={id}
            {...(hideLabel && label ? { "aria-label": label } : {})}
          >
            <CalendarIcon
              className={cn("mr-2", size === "default" ? "h-4 w-4" : "h-3 w-3")}
            />
            {buttonText}
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-auto p-0" sideOffset={6}>
          <Calendar
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
      {error && <span className="text-sm text-red-500">{error}</span>}
    </div>
  );
};
