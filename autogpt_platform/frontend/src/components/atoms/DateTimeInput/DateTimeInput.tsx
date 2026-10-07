"use client";

import * as React from "react";
import { Calendar03Icon, Clock01Icon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { fieldVariants } from "../Input/fieldVariants";

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
  size?: "sm" | "md" | "lg";
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
    fieldVariants({ size, invalid: Boolean(error) }),
    "min-w-0 justify-start gap-2 text-left",
    !selected && "text-muted-foreground",
    className,
  );
  const iconSize = size === "sm" ? 14 : 16;

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
            aria-label={ariaLabel ?? (hideLabel && label ? label : undefined)}
            aria-labelledby={ariaLabelledBy}
            aria-describedby={ariaDescribedBy}
            aria-invalid={error ? true : undefined}
          >
            <Icon icon={Calendar03Icon} size={iconSize} aria-hidden />
            <Icon icon={Clock01Icon} size={iconSize} aria-hidden />
            {buttonText}
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-auto p-0" sideOffset={6}>
          <div className="p-3">
            <Calendar
              mode="single"
              selected={selected}
              defaultMonth={selected}
              onSelect={setDate}
              showOutsideDays
              modifiersClassNames={{
                disabled: "pointer-events-none opacity-50",
              }}
            />
            <div className="mt-3 border-t pt-3">
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
          </div>
        </PopoverContent>
      </Popover>
      {error && (
        <Text
          variant="small-medium"
          as="span"
          tone="danger"
          className={cn(
            "absolute top-full left-0 mt-1 transition-opacity duration-200",
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
