"use client";

import * as React from "react";
import { Calendar03Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { cn } from "@/lib/utils";
import { DatePickerCalendar } from "./components/DatePickerCalendar";
import { fieldVariants, type FieldSize } from "../Input/fieldVariants";

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
  size?: FieldSize;
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
  size = "lg",
  "aria-label": ariaLabel,
  "aria-labelledby": ariaLabelledBy,
  "aria-describedby": ariaDescribedBy,
}: DateInputProps) => {
  const selected = parseISODateString(value);
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

  return (
    <div className="flex flex-col gap-1">
      {label && !hideLabel && (
        <label htmlFor={id}>
          <Text variant="body-medium" as="span">
            {label}
          </Text>
        </label>
      )}
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
          <Icon
            icon={Calendar03Icon}
            size={size === "sm" ? 14 : 16}
            aria-hidden
          />
          {buttonText}
        </PopoverTrigger>
        <PopoverContent
          className="w-auto p-0"
          sideOffset={6}
          aria-label="Choose a date"
        >
          <DatePickerCalendar selected={selected} onSelect={setDate} />
        </PopoverContent>
      </Popover>
      {error && (
        <Text variant="small-medium" as="span" tone="danger">
          {error}
        </Text>
      )}
    </div>
  );
};
