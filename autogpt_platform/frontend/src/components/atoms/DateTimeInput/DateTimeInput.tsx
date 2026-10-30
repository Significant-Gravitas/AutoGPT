"use client";

import * as React from "react";
import { Calendar03Icon, Clock01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";

import { Text } from "../Text/Text";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { Calendar } from "@/components/atoms/Calendar/Calendar";

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
  "aria-label"?: string;
  "aria-labelledby"?: string;
  "aria-describedby"?: string;
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
  "aria-label": ariaLabel,
  "aria-labelledby": ariaLabelledBy,
  "aria-describedby": ariaDescribedBy,
}: DateTimeInputProps) => {
  const selected = React.useMemo(() => parseISODateTimeString(value), [value]);
  const [open, setOpen] = React.useState(false);
  const [timeValue, setTimeValue] = React.useState("");
  const timeInputId = React.useId();

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
    "rounded-3xl border border-border-base bg-surface px-4 shadow-none",
    "font-normal text-foreground-base w-full text-body leading-5",
    "placeholder:font-normal",
    // Focus and hover states
    "focus:border-foreground-subtle focus:shadow-none focus:outline-none focus:ring-1 focus:ring-foreground-subtle focus:ring-offset-0",
    // Error state
    error && "border-error focus:border-error focus:ring-error",
    // Placeholder styling
    !selected && "text-foreground-subtle",
    "justify-start text-left",
    // Size variants
    size === "default" && "h-[2.875rem] py-2.5",
    size === "small" && [
      "min-h-9", // 36px minimum
      "py-2",
      "text-body",
      "placeholder:text-body",
    ],
    className,
  );

  const timeInputStyles = cn(
    // Base styles
    "rounded-3xl border border-border-base bg-surface px-4 shadow-none",
    "font-normal text-foreground-base w-full",
    "placeholder:font-normal placeholder:text-placeholder",
    // Focus and hover states
    "focus:border-foreground-subtle focus:shadow-none focus:outline-none focus:ring-1 focus:ring-foreground-subtle focus:ring-offset-0",
    // Size variants
    size === "small" && [
      "h-9", // 36px
      "py-2",
      "text-body", // 14px font, 22px line height
      "placeholder:text-body",
    ],
    size === "default" && [
      "h-[2.875rem]", // 46px
      "py-2.5",
    ],
  );

  const inputWithError = (
    <div
      data-slot="date-time-input-field"
      className={cn("relative", error ? "mb-6" : "", wrapperClassName)}
    >
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            data-slot="date-time-input-trigger"
            type="button"
            variant="ghost"
            className={triggerStyles}
            disabled={isDisabled}
            autoFocus={autoFocus}
            id={id}
            aria-label={ariaLabel ?? (hideLabel && label ? label : undefined)}
            aria-labelledby={ariaLabelledBy}
            aria-describedby={ariaDescribedBy}
          >
            <Icon
              aria-hidden
              icon={Calendar03Icon}
              data-slot="date-time-input-date-icon"
              className={cn("mr-2", size === "default" ? "h-4 w-4" : "h-3 w-3")}
            />
            <Icon
              aria-hidden
              icon={Clock01Icon}
              data-slot="date-time-input-time-icon"
              className={cn("mr-2", size === "default" ? "h-4 w-4" : "h-3 w-3")}
            />
            {buttonText}
          </Button>
        </PopoverTrigger>
        <PopoverContent
          data-slot="date-time-input-popup"
          className="w-auto rounded-medium border-border-base p-0 text-foreground-base"
          sideOffset={6}
        >
          <div data-slot="date-time-input-popup-body" className="p-3">
            <Calendar
              data-slot="date-time-input-calendar"
              mode="single"
              selected={selected}
              onSelect={setDate}
              showOutsideDays
              modifiersClassNames={{
                disabled: "pointer-events-none opacity-50",
              }}
            />
            <div
              data-slot="date-time-input-time"
              className="mt-3 border-t pt-3"
            >
              <label
                data-slot="date-time-input-time-label"
                htmlFor={timeInputId}
                className="mb-2 block text-body font-medium leading-5 text-foreground-body"
              >
                Time
              </label>
              <input
                data-slot="date-time-input-time-control"
                id={timeInputId}
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
          data-slot="date-time-input-error"
          variant="small-medium"
          as="span"
          tone="error"
          className={cn(
            "absolute left-0 top-full mt-1 transition-opacity duration-200",
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
    <label
      data-slot="date-time-input"
      htmlFor={id}
      className="flex flex-col gap-2"
    >
      <div
        data-slot="date-time-input-header"
        className="flex items-center justify-between"
      >
        <Text data-slot="date-time-input-label" variant="body-medium" as="span">
          {label}
        </Text>
        {hint ? (
          <Text
            data-slot="date-time-input-hint"
            variant="small"
            as="span"
            tone="subtle"
          >
            {hint}
          </Text>
        ) : null}
      </div>
      {inputWithError}
    </label>
  );
};
