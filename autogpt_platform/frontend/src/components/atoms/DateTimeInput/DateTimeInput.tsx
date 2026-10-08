"use client";

import * as React from "react";
import { Calendar03Icon, Clock01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { cn } from "@/lib/utils";
import { DatePickerCalendar } from "../DateInput/components/DatePickerCalendar";
import { fieldVariants, type FieldSize } from "../Input/fieldVariants";
import { Text } from "../Text/Text";

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

function toTimeString(d?: Date) {
  if (!d) return "";
  const hours = String(d.getHours()).padStart(2, "0");
  const minutes = String(d.getMinutes()).padStart(2, "0");
  return `${hours}:${minutes}`;
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
  size?: FieldSize;
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
  size = "lg",
  wrapperClassName,
  "aria-label": ariaLabel,
  "aria-labelledby": ariaLabelledBy,
  "aria-describedby": ariaDescribedBy,
}: DateTimeInputProps) => {
  const selected = parseISODateTimeString(value);
  const [open, setOpen] = React.useState(false);
  // Time typed before a date is picked; once a date exists the value holds it.
  const [pendingTime, setPendingTime] = React.useState("");
  const timeValue = selected ? toTimeString(selected) : pendingTime;
  const timeInputId = React.useId();

  const setDate = (d?: Date) => {
    if (!d) {
      onChange?.(undefined);
      setOpen(false);
      return;
    }

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
    setPendingTime(time);
    if (!selected || !time) return;

    const [hours, minutes] = time.split(":").map(Number);
    if (!isNaN(hours) && !isNaN(minutes)) {
      const newDate = new Date(selected);
      newDate.setHours(hours, minutes, 0, 0);
      onChange?.(toLocalISODateTimeString(newDate));
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
  const iconSize = size === "sm" ? 14 : 16;

  const inputWithError = (
    <div className={cn("relative", error ? "mb-6" : "", wrapperClassName)}>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger
          type="button"
          className={cn(
            fieldVariants({ size, invalid: Boolean(error) }),
            "inline-flex min-w-0 items-center justify-start gap-2 text-left",
            !selected && "text-muted-foreground",
            className,
          )}
          disabled={isDisabled}
          autoFocus={autoFocus}
          id={id}
          aria-label={ariaLabel ?? (hideLabel && label ? label : undefined)}
          aria-labelledby={ariaLabelledBy}
          aria-describedby={ariaDescribedBy}
          aria-invalid={error ? true : undefined}
        >
          <Icon icon={Calendar03Icon} size={iconSize} aria-hidden />
          <Icon icon={Clock01Icon} size={iconSize} aria-hidden />
          {buttonText}
        </PopoverTrigger>
        <PopoverContent
          className="w-auto p-0"
          sideOffset={6}
          aria-label="Choose a date and time"
        >
          <DatePickerCalendar selected={selected} onSelect={setDate} />
          <div className="border-t border-border p-3">
            <label htmlFor={timeInputId} className="mb-2 block">
              <Text variant="body-medium" as="span">
                Time
              </Text>
            </label>
            <input
              id={timeInputId}
              type="time"
              value={timeValue}
              onChange={(e) => handleTimeChange(e.target.value)}
              className={fieldVariants({ size })}
              disabled={isDisabled}
              placeholder="HH:MM"
            />
          </div>
        </PopoverContent>
      </Popover>
      {error && (
        <Text
          variant="small-medium"
          as="span"
          tone="danger"
          className="absolute top-full left-0 mt-1"
        >
          {error}
        </Text>
      )}
    </div>
  );

  return hideLabel || !label ? (
    inputWithError
  ) : (
    <label htmlFor={id} className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <Text variant="body-medium" as="span">
          {label}
        </Text>
        {hint ? (
          <Text variant="small" as="span" tone="muted">
            {hint}
          </Text>
        ) : null}
      </div>
      {inputWithError}
    </label>
  );
};
