"use client";

import * as React from "react";
import { Calendar as CalendarIcon, Clock } from "lucide-react";
import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";

import { Text } from "../Text/Text";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/__legacy__/ui/popover";
import { Calendar } from "@/components/__legacy__/ui/calendar";

function toLocalISODateTimeString(d: Date) {
  const year = d.getFullYear();
  const month = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  const hours = String(d.getHours()).padStart(2, "0");
  const minutes = String(d.getMinutes()).padStart(2, "0");
  return `${year}-${month}-${day}T${hours}:${minutes}`;
}

function parseISODateTimeString(s?: string): Date | undefined {
  if (!s) return undefined;
  // Expecting "YYYY-MM-DDTHH:MM" or "YYYY-MM-DD HH:MM"
  const normalized = s.replace(" ", "T");
  const date = new Date(normalized);
  return isNaN(date.getTime()) ? undefined : date;
}

export interface DateTimeInputProps {
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
  hint?: React.ReactNode;
  id?: string;
  size?: "default" | "small";
  wrapperClassName?: string;
}

export const DateTimeInput = ({
  value,
  onChange,
  disabled = false,
  readonly = false,
  placeholder,
  autoFocus,
  className,
  label,
  hideLabel = false,
  error,
  hint,
  id,
  size = "default",
  wrapperClassName,
}: DateTimeInputProps) => {
  const selected = React.useMemo(() => parseISODateTimeString(value), [value]);
  const [open, setOpen] = React.useState(false);
  const [timeValue, setTimeValue] = React.useState("");

  // Update time value when selected date changes
  React.useEffect(() => {
    if (selected) {
      const hours = String(selected.getHours()).padStart(2, "0");
      const minutes = String(selected.getMinutes()).padStart(2, "0");
      setTimeValue(`${hours}:${minutes}`);
    } else {
      setTimeValue("");
    }
  }, [selected]);

  const setDate = (d?: Date) => {
    if (!d) {
      onChange?.(undefined);
      setOpen(false);
      return;
    }

    // If we have a time value, apply it to the selected date
    if (timeValue) {
      const [hours, minutes] = timeValue.split(":").map(Number);
      if (!isNaN(hours) && !isNaN(minutes)) {
        d.setHours(hours, minutes, 0, 0);
      }
    }

    onChange?.(toLocalISODateTimeString(d));
    setOpen(false);
  };

  const handleTimeChange = (time: string) => {
    setTimeValue(time);

    if (selected && time) {
      const [hours, minutes] = time.split(":").map(Number);
      if (!isNaN(hours) && !isNaN(minutes)) {
        const newDate = new Date(selected);
        newDate.setHours(hours, minutes, 0, 0);
        onChange?.(toLocalISODateTimeString(newDate));
      }
    }
  };

  const buttonText = selected
    ? selected.toLocaleDateString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
      }) +
      " " +
      selected.toLocaleTimeString(undefined, {
        hour: "2-digit",
        minute: "2-digit",
      })
    : placeholder || "Pick date and time";

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
    size === "small" && [
      "min-h-[2.25rem]", // 36px minimum
      "py-2",
      "text-sm leading-[22px]",
      "placeholder:text-sm placeholder:leading-[22px]",
    ],
    className,
  );

  const timeInputStyles = cn(
    // Base styles
    "rounded-3xl border border-zinc-200 bg-white px-4 shadow-none",
    "font-normal text-black w-full",
    "placeholder:font-normal placeholder:text-zinc-400",
    // Focus and hover states
    "focus:border-zinc-400 focus:shadow-none focus:outline-none focus:ring-1 focus:ring-zinc-400 focus:ring-offset-0",
    // Size variants
    size === "small" && [
      "h-[2.25rem]", // 36px
      "py-2",
      "text-sm leading-[22px]", // 14px font, 22px line height
      "placeholder:text-sm placeholder:leading-[22px]",
    ],
    size === "default" && [
      "h-[2.875rem]", // 46px
      "py-2.5",
    ],
  );

  const inputWithError = (
    <div className={cn("relative", error ? "mb-6" : "", wrapperClassName)}>
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
            <Clock
              className={cn("mr-2", size === "default" ? "h-4 w-4" : "h-3 w-3")}
            />
            {buttonText}
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-auto p-0" sideOffset={6}>
          <div className="p-3">
            <Calendar
              mode="single"
              selected={selected}
              onSelect={setDate}
              showOutsideDays
              modifiersClassNames={{
                disabled: "pointer-events-none opacity-50",
              }}
            />
            <div className="mt-3 border-t pt-3">
              <label className="mb-2 block text-sm font-medium text-gray-700">
                Time
              </label>
              <input
                type="time"
                value={timeValue}
                onChange={(e) => handleTimeChange(e.target.value)}
                className={timeInputStyles}
                disabled={isDisabled}
                placeholder="HH:MM"
              />
            </div>
          </div>
        </PopoverContent>
      </Popover>
      {error && (
        <Text
          variant="small-medium"
          as="span"
          className={cn(
            "absolute left-0 top-full mt-1 !text-red-500 transition-opacity duration-200",
            error ? "opacity-100" : "opacity-0",
          )}
        >
          {error || " "}
        </Text>
      )}
    </div>
  );

  return hideLabel || !label ? (
    inputWithError
  ) : (
    <label htmlFor={id} className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <Text variant="body-medium" as="span" className="text-black">
          {label}
        </Text>
        {hint ? (
          <Text variant="small" as="span" className="!text-zinc-400">
            {hint}
          </Text>
        ) : null}
      </div>
      {inputWithError}
    </label>
  );
};
